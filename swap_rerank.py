import os
import numpy as np
import pandas as pd
from pathlib import Path
from geopy.distance import geodesic
from tqdm import tqdm
import warnings
from sklearn.cluster import DBSCAN
from math import radians, sin, cos, sqrt, asin

warnings.filterwarnings('ignore')


top_n = 20 
root_path = './data'
D_pos_path = 'index/D_pos_dvseg_dual_im2gps.npy'  
I_pos_path = 'index/I_pos_dvseg_dual_im2gps.npy'  
rgb_csv_name = 'MP16_Pro_places365.csv'
seg_csv_name = 'mp16-seg-png.csv'


EPS_LIST = [5, 10, 20, 30, 50, 100]
MIN_SAMPLES = 2  


DIST_THRESHOLDS = [1, 25, 200, 750, 2500]

OUTPUT_SUMMARY_CSV = 'sweep_eps_geo_cluster_summary.csv'
OUTPUT_DETAIL_DIR = 'sweep_results'

os.makedirs(OUTPUT_DETAIL_DIR, exist_ok=True)

def haversine_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * 6371

def geo_cluster_rerank(candidate_coords, eps_km=50, min_samples=2):
    if len(candidate_coords) == 0:
        return None
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
    distances = [haversine_distance(coord, cluster_center) for coord in candidate_coords]
    return int(np.argmin(distances))

def run_rerank_for_eps(eps_km, test_df, merged_df, I_pos, top_n):
    num_test = I_pos.shape[0]
    rerank_results = []
    for test_idx in range(num_test):
        test_row = test_df.iloc[test_idx]
        test_img_id = test_row['IMG_ID']
        test_lon = float(test_row['test_lon'])
        test_lat = float(test_row['test_lat'])

        cand_indices = I_pos[test_idx, :top_n].astype(int)
        candidate_coords = []
        for idx in cand_indices:
            if idx < 0 or idx >= len(merged_df):
                continue
            row = merged_df.iloc[idx]
            if pd.isna(row['LON']) or pd.isna(row['LAT']):
                continue
            candidate_coords.append((float(row['LAT']), float(row['LON'])))

        if not candidate_coords:
            rerank_results.append({
                'test_img_id': test_img_id,
                'rerank_top1_lat': None,
                'rerank_top1_lon': None,
                'GEODESIC': np.inf
            })
            continue

        best_idx = geo_cluster_rerank(candidate_coords, eps_km=eps_km, min_samples=MIN_SAMPLES)
        best_lat, best_lon = candidate_coords[best_idx] if best_idx is not None else candidate_coords[0]

        try:
            distance = geodesic((test_lat, test_lon), (best_lat, best_lon)).kilometers
        except:
            distance = np.inf

        rerank_results.append({
            'test_img_id': test_img_id,
            'rerank_top1_lat': best_lat,
            'rerank_top1_lon': best_lon,
            'GEODESIC': distance
        })

    rerank_df = pd.DataFrame(rerank_results)
    final_df = pd.merge(
        test_df[['IMG_ID', 'test_lon', 'test_lat']],
        rerank_df,
        left_on='IMG_ID',
        right_on='test_img_id',
        how='left'
    ).drop(columns=['test_img_id'])

    return final_df

def main():

    if not os.path.exists(I_pos_path):
        raise FileNotFoundError(f"{I_pos_path}")
    I_pos = np.load(I_pos_path)
    num_test = I_pos.shape[0]


    test_rgb_csv = os.path.join(root_path, 'im2gps.csv')
    test_df = pd.read_csv(test_rgb_csv)
    test_df = test_df.rename(columns={'LON': 'test_lon', 'LAT': 'test_lat'})

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

    summary_rows = []

    for eps in EPS_LIST:
        result_df = run_rerank_for_eps(eps, test_df, merged_df, I_pos, top_n)

        detail_path = os.path.join(OUTPUT_DETAIL_DIR, f'results_eps_{eps}km.csv')
        result_df.to_csv(detail_path, index=False, encoding='utf-8')

        valid_samples = result_df[result_df['GEODESIC'] != np.inf]
        total_valid = len(valid_samples)
        total_test = len(result_df)

        acc_dict = {'eps_km': eps, 'total_test': total_test, 'valid_reranked': total_valid}
        for th in DIST_THRESHOLDS:
            acc = (valid_samples['GEODESIC'] < th).sum() / total_valid if total_valid > 0 else 0.0
            acc_dict[f'acc@{th}km'] = acc

        summary_rows.append(acc_dict)
        print(f"  → Valid: {total_valid}/{total_test}")
        for th in DIST_THRESHOLDS:
            print(f"    Acc@{th}km: {acc_dict[f'acc@{th}km']:.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    print(f"\n💾 Summary saved to: {OUTPUT_SUMMARY_CSV}")

    best_row = summary_df.loc[summary_df['acc@25km'].idxmax()]
    print("\n🎉 Best configuration:")
    print(f"  eps_km = {best_row['eps_km']} km")
    for th in DIST_THRESHOLDS:
        print(f"  Acc@{th}km = {best_row[f'acc@{th}km']:.4f}")

if __name__ == "__main__":
    main()