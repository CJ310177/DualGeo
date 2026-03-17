import os
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from tqdm import tqdm
import warnings
from sklearn.cluster import DBSCAN
from math import radians, asin, sin, cos, sqrt

warnings.filterwarnings('ignore')


top_n = 20
root_path = './data'
I_pos_path = 'index/I_pos_dvseg_dual_im2gps3k.npy'
rgb_csv_name = 'MP16_Pro_places365.csv'
seg_csv_name = 'mp16-seg-png.csv'

EPS_KM = 50
MIN_SAMPLES = 2

output_csv = 'rerank_geo_cluster_wide_results.csv'

print(f"Geo-Cluster Rerank (Wide Format): eps={EPS_KM}km, top_n={top_n}")

def haversine_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * 6371

def geo_cluster_rerank_full(candidate_coords, eps_km=50, min_samples=2):
    if len(candidate_coords) == 0:
        return []
    coords_array = np.array(candidate_coords)
    coords_rad = np.radians(coords_array)
    db = DBSCAN(
        eps=eps_km / 6371.0,
        min_samples=min_samples,
        algorithm='ball_tree',
        metric='haversine'
    )
    labels = db.fit_predict(coords_rad)
    unique_labels, counts = np.unique(labels, return_counts=True)
    noise_mask = unique_labels != -1
    if np.any(noise_mask):
        valid_labels = unique_labels[noise_mask]
        valid_counts = counts[noise_mask]
        main_label = valid_labels[np.argmax(valid_counts)]
        main_mask = labels == main_label
        cluster_center = np.mean(coords_array[main_mask], axis=0)
    else:
        cluster_center = np.mean(coords_array, axis=0)
    distances = np.array([haversine_distance(coord, cluster_center) for coord in candidate_coords])
    reranked_local_indices = np.argsort(distances).tolist()
    return reranked_local_indices

if __name__ == "__main__":
    print("Loading retrieval results...")
    if not os.path.exists(I_pos_path):
        raise FileNotFoundError(f" {I_pos_path}")
    I_pos = np.load(I_pos_path)
    num_test = I_pos.shape[0]

    print("Loading test set info...")
    test_rgb_csv = os.path.join(root_path, 'im2gps3k_places365.csv')
    test_df = pd.read_csv(test_rgb_csv)
    test_df = test_df.rename(columns={'LON': 'LON', 'LAT': 'LAT'})

    print("Loading MP16 database...")
    rgb_csv_path = os.path.join(root_path, rgb_csv_name)
    seg_csv_path = os.path.join(root_path, seg_csv_name)
    rgb_df = pd.read_csv(rgb_csv_path)
    seg_df = pd.read_csv(seg_csv_path)

    rgb_df = rgb_df[rgb_df['country'].notnull()].reset_index(drop=True)
    rgb_df['COMMON_ID'] = rgb_df['IMG_ID'].str.replace('/', '_', regex=False).str.replace('.jpg', '', regex=False)
    seg_df['COMMON_ID'] = seg_df['IMG_ID'].apply(lambda x: x.replace('.png', '').replace('/', '_'))
    merged_df = pd.merge(
        rgb_df[['IMG_ID', 'COMMON_ID', 'LON', 'LAT']],
        seg_df[['IMG_ID', 'COMMON_ID']].rename(columns={'IMG_ID': 'SEG_IMG_ID'}),
        on='COMMON_ID',
        how='inner'
    ).reset_index(drop=True)

    columns = ['IMG_ID', 'LAT', 'LON']
    for i in range(top_n):
        columns.extend([f'pred_lat_{i}', f'pred_lon_{i}'])
    columns.append('geodesic_km')

    results = []

    print(f"Starting wide-format GEO-CLUSTER reranking (eps={EPS_KM}km)...")
    for test_idx in tqdm(range(num_test), desc="Processing"):
        test_row = test_df.iloc[test_idx]
        img_id = test_row['IMG_ID']
        true_lat = float(test_row['LAT'])
        true_lon = float(test_row['LON'])

        orig_cand_indices = I_pos[test_idx, :top_n].astype(int)
        candidate_coords = []
        valid_orig_indices = []

        for idx in orig_cand_indices:
            if idx < 0 or idx >= len(merged_df):
                continue
            row = merged_df.iloc[idx]
            if pd.isna(row['LON']) or pd.isna(row['LAT']):
                continue
            candidate_coords.append((float(row['LAT']), float(row['LON'])))
            valid_orig_indices.append(idx)

        pred_lats = [np.nan] * top_n
        pred_lons = [np.nan] * top_n

        if candidate_coords:
            reranked_order = geo_cluster_rerank_full(candidate_coords, eps_km=EPS_KM, min_samples=MIN_SAMPLES)
            for rank, local_idx in enumerate(reranked_order):
                if rank >= top_n:
                    break
                lat, lon = candidate_coords[local_idx]
                pred_lats[rank] = lat
                pred_lons[rank] = lon

        geodesic_km = np.inf
        if not (np.isnan(pred_lats[0]) or np.isnan(pred_lons[0])):
            try:
                geodesic_km = geodesic((true_lat, true_lon), (pred_lats[0], pred_lons[0])).kilometers
            except:
                geodesic_km = np.inf

        row_data = [img_id, true_lat, true_lon]
        for i in range(top_n):
            row_data.extend([pred_lats[i], pred_lons[i]])
        row_data.append(geodesic_km)

        results.append(row_data)

    result_df = pd.DataFrame(results, columns=columns)
    result_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Wide-format results saved to: {output_csv}")

    valid_top1 = result_df[result_df['geodesic_km'] != np.inf]
    total_valid = len(valid_top1)
    total_test = len(result_df)
    print(f"Top-1 Accuracy (based on pred_lat_0 / pred_lon_0):")
    print(f"Total: {total_test}, Valid: {total_valid}")
    for th in [1, 25, 200, 750, 2500]:
        acc = (valid_top1['geodesic_km'] < th).sum() / total_valid if total_valid > 0 else 0.0
        print(f"   Acc@{th}km: {acc:.4f}")

    print("Wide-format Geo-Cluster Reranking completed!")