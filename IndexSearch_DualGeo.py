# IndexSearch_DV.py
import faiss
import torch
import numpy as np
import os
import argparse
import pandas as pd
from tqdm import tqdm
from geopy.distance import geodesic
from torch.utils.data import DataLoader, Dataset, get_worker_info
import pickle
import tarfile
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------
# Only import Accelerator for index building (multi-GPU)
# Evaluation runs on single GPU without Accelerate
# -----------------------------

# -----------------------------
# Dataset for Index Building (RGB + Seg)
# -----------------------------
class MP16DualIndexDataset(Dataset):
    def __init__(
        self,
        merged_df,
        root_path='./data/',
        tar_path='mp-16-images.tar',
        seg_dir='mp16-seg-pngs',
        tar_index_path='tar_index.pkl',
        png_index_path='png_index.pkl',
        vision_processor_rgb=None,
        target_size=224
    ):
        self.merged_df = merged_df.reset_index(drop=True)
        self.root_path = root_path
        self.tar_path = os.path.join(root_path, tar_path)
        self.seg_dir = os.path.join(root_path, seg_dir)
        self.vision_processor_rgb = vision_processor_rgb
        self.target_size = target_size

        with open(os.path.join(root_path, tar_index_path), 'rb') as f:
            self.tar_index = pickle.load(f)
        with open(os.path.join(root_path, png_index_path), 'rb') as f:
            self.png_index = pickle.load(f)

        self.rgb_keys = []
        self.seg_keys = []
        for _, row in self.merged_df.iterrows():
            rgb_key = row['IMG_ID'].replace('/', '_')
            self.rgb_keys.append(rgb_key)
            self.seg_keys.append(row['SEG_IMG_ID'])

    def __len__(self):
        return len(self.merged_df)

    def __getitem__(self, idx):
        rgb_key = self.rgb_keys[idx]
        seg_key = self.seg_keys[idx]

        worker = get_worker_info()
        worker_id = worker.id if worker else None
        if not hasattr(self, 'tar_objs'):
            self.tar_objs = {}
        if worker_id not in self.tar_objs:
            self.tar_objs[worker_id] = tarfile.open(self.tar_path, 'r')

        try:
            member = self.tar_index[rgb_key]
            image_file = self.tar_objs[worker_id].extractfile(member)
            rgb_img = Image.open(image_file).convert('RGB')
        except Exception as e:
            print(f"⚠️ Failed to load RGB {rgb_key}: {e}")
            rgb_img = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.vision_processor_rgb:
            rgb_tensor = self.vision_processor_rgb(images=rgb_img, return_tensors='pt')['pixel_values'].squeeze(0)
        else:
            from torchvision import transforms as T
            transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
            rgb_tensor = transform(rgb_img)

        try:
            seg_path = self.png_index[seg_key]
            seg_img = Image.open(seg_path).convert('L')
        except Exception as e:
            print(f"⚠️ Failed to load Seg {seg_key}: {e}")
            seg_img = Image.new('L', (224, 224), 0)

        from torchvision import transforms as T
        seg_transform = T.Compose([
            T.Resize((self.target_size, self.target_size)),
            T.ToTensor()
        ])
        seg_tensor = seg_transform(seg_img)

        return rgb_tensor, seg_tensor


# -----------------------------
# 2. Build Index with Multi-GPU (using Accelerate)
# -----------------------------
def build_mp16_dual_index(args):
    print("Building MP-16 dual (RGB+Seg) index using DataLoader with Accelerate...")
    from accelerate import Accelerator, DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    from utils.DualView_seg import DualViewG3 as DVseg
    model = DVseg(device=device).to(device)
    checkpoint = torch.load('./checkpoints/dvseg_9_.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.requires_grad_(False)
    model = accelerator.prepare(model)

    root_path = './data/'
    tar_path = 'mp-16-images.tar'
    seg_dir = 'mp16-seg-pngs'

    rgb_csv = pd.read_csv(os.path.join(root_path, 'MP16_Pro_places365.csv'))
    seg_csv = pd.read_csv(os.path.join(root_path, 'mp16-seg-png.csv'))
    rgb_csv = rgb_csv[rgb_csv['country'].notnull()].reset_index(drop=True)
    rgb_csv['COMMON_ID'] = rgb_csv['IMG_ID'].str.replace('/', '_', regex=False).str.replace('.jpg', '', regex=False)
    seg_csv['COMMON_ID'] = seg_csv['IMG_ID'].apply(lambda x: x.replace('.png', '').replace('/', '_'))

    merged = pd.merge(
        rgb_csv[['IMG_ID', 'COMMON_ID', 'LON', 'LAT']],
        seg_csv[['IMG_ID', 'COMMON_ID']].rename(columns={'IMG_ID': 'SEG_IMG_ID'}),
        on='COMMON_ID',
        how='inner'
    )
    print(f"✅ Aligned {len(merged)} samples for MP-16 index.")

    dataset = MP16DualIndexDataset(
        merged_df=merged,
        root_path=root_path,
        tar_path=tar_path,
        seg_dir=seg_dir,
        tar_index_path='tar_index.pkl',
        png_index_path='png_index.pkl',
        vision_processor_rgb=model.vision_processor_rgb,
        target_size=224
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=3,
        drop_last=False
    )
    dataloader = accelerator.prepare(dataloader)

    all_features = []
    d = 1536  # 768 + 768
    # d = 768

    for rgb_batch, seg_batch in tqdm(dataloader, desc="Extracting features", disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            rgb_feat = model.vision_model_rgb(rgb_batch)[1]  # [B, 768]
            rgb_feat = model.vision_projection_rgb(rgb_feat)
            rgb_feat = torch.nn.functional.normalize(rgb_feat, p=2, dim=-1)
            enhanced_rgb, enhanced_seg = model.extract_enhanced_features(rgb_batch, seg_batch)
            enhanced_rgb = torch.nn.functional.normalize(enhanced_rgb, p=2, dim=-1)
            # enhanced_seg = torch.nn.functional.normalize(enhanced_seg, p=2, dim=-1)
            joint_feat = torch.cat([enhanced_rgb, rgb_feat], dim=1)  # [B, 1536]
            # joint_feat = enhanced_rgb
            
 
            joint_feat_gathered = accelerator.gather(joint_feat)
            if accelerator.is_main_process:
                all_features.append(joint_feat_gathered.cpu().numpy())

    if accelerator.is_main_process:
        all_features = np.vstack(all_features).astype(np.float32)
        index_flat = faiss.IndexFlatIP(d)
        index_flat.add(all_features)
        os.makedirs('./index', exist_ok=True)
        faiss.write_index(index_flat, './index/dvseg_dual.index')
        print(f"✅ Index saved with {index_flat.ntotal} vectors.")

    accelerator.wait_for_everyone()


# -----------------------------
# 3. Search & Evaluate on Single GPU (no Accelerate)
# -----------------------------
def search_and_evaluate_single_gpu(args, index, topk=20):
    print(f"Searching on {args.dataset} with dual features (single GPU)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from utils.DualView_seg import DualViewG3 as DVseg
    from utils.utils import Im2gpsDualDataset, Im2gps3kDualDataset, Yfcc4kDualDataset

    model = DVseg(device=device).to(device)
    checkpoint = torch.load('./checkpoints/dvseg_9_.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.requires_grad_(False)

    if args.dataset == 'im2gps':
        dataset = Im2gpsDualDataset(vision_processor_rgb=model.vision_processor_rgb)
    elif args.dataset == 'im2gps3k':
        dataset = Im2gps3kDualDataset(vision_processor_rgb=model.vision_processor_rgb)
    elif args.dataset == 'yfcc4k':
        dataset = Yfcc4kDualDataset(vision_processor_rgb=model.vision_processor_rgb)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=5
    )

    test_embeds = []
    for rgb_imgs, seg_imgs, img_ids, lons, lats in tqdm(dataloader, desc="Extracting test features"):
        rgb_imgs = rgb_imgs.to(device, non_blocking=True)
        seg_imgs = seg_imgs.to(device, non_blocking=True)
        with torch.no_grad():
            enhanced_rgb, enhanced_seg = model.extract_enhanced_features(rgb_imgs, seg_imgs)
            enhanced_rgb = torch.nn.functional.normalize(enhanced_rgb, p=2, dim=-1)
            rgb_feat = model.vision_model_rgb(rgb_imgs)[1]
            rgb_feat = model.vision_projection_rgb(rgb_feat)
            rgb_feat = torch.nn.functional.normalize(rgb_feat, p=2, dim=-1)
            # enhanced_seg = torch.nn.functional.normalize(enhanced_seg, p=2, dim=-1)
            # joint_feat = torch.cat([enhanced_rgb, enhanced_seg], dim=1)
            # joint_feat = enhanced_rgb
            joint_feat = torch.cat([enhanced_rgb, rgb_feat], dim=1)
            test_embeds.append(joint_feat.cpu().numpy())

    test_embeds = np.vstack(test_embeds).astype(np.float32)
    print(f"Test embeddings shape: {test_embeds.shape}")

    D_pos, I_pos = index.search(test_embeds, topk)
    D_neg, I_neg = index.search(-test_embeds, topk)

    # Save retrieval results
    np.save(f'./index/D_pos_dvseg_dual_{args.dataset}.npy', D_pos)
    np.save(f'./index/I_pos_dvseg_dual_{args.dataset}.npy', I_pos)
    np.save(f'./index/D_neg_dvseg_dual_{args.dataset}.npy', D_neg)
    np.save(f'./index/I_neg_dvseg_dual_{args.dataset}.npy', I_neg)
    print("✅ Saved positive and negative retrieval results.")

    return I_pos


def evaluate(args, I_pos):
    print('Evaluating retrieval results...')
    root_path = './data/'

    rgb_csv = pd.read_csv(os.path.join(root_path, 'MP16_Pro_places365.csv'))
    seg_csv = pd.read_csv(os.path.join(root_path, 'mp16-seg-png.csv'))
    rgb_csv = rgb_csv[rgb_csv['country'].notnull()].reset_index(drop=True)
    rgb_csv['COMMON_ID'] = rgb_csv['IMG_ID'].str.replace('/', '_', regex=False).str.replace('.jpg', '', regex=False)
    seg_csv['COMMON_ID'] = seg_csv['IMG_ID'].apply(lambda x: x.replace('.png', '').replace('/', '_'))

    database_df = pd.merge(
        rgb_csv[['IMG_ID', 'COMMON_ID', 'LON', 'LAT']],
        seg_csv[['IMG_ID', 'COMMON_ID']].rename(columns={'IMG_ID': 'SEG_IMG_ID'}),
        on='COMMON_ID',
        how='inner'
    ).reset_index(drop=True)

    if args.dataset == 'im2gps':
        test_df = pd.read_csv('./data/im2gps.csv')
        test_df = test_df.rename(columns={'IMG_ID': 'QUERY_IMG_ID'})
    elif args.dataset == 'im2gps3k':
        test_df = pd.read_csv('./data/im2gps3k_places365.csv')
        test_df = test_df.rename(columns={'IMG_ID': 'QUERY_IMG_ID'})
    elif args.dataset == 'yfcc4k':
        test_df = pd.read_csv('./data/yfcc4k_places365.csv')
        test_df = test_df.rename(columns={'IMG_ID': 'QUERY_IMG_ID'})
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    max_index = len(database_df) - 1
    I_pos = np.clip(I_pos, 0, max_index)

    test_df['NN_IDX'] = I_pos[:, 0]
    test_df['LAT_PRED'] = test_df['NN_IDX'].apply(lambda x: database_df.loc[x, 'LAT'])
    test_df['LON_PRED'] = test_df['NN_IDX'].apply(lambda x: database_df.loc[x, 'LON'])
    test_df['GEODESIC'] = test_df.apply(
        lambda row: geodesic((row['LAT'], row['LON']), (row['LAT_PRED'], row['LON_PRED'])).km,
        axis=1
    )

    save_path = f'./data/{args.dataset}_dvseg_dual_results.csv'
    test_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

    total = len(test_df)
    for th in [2500, 750, 200, 25, 1]:
        acc = (test_df['GEODESIC'] < th).sum() / total
        print(f"{th}km level: {acc:.4f}")

# -----------------------------
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_name', type=str, default='dvseg_dual')
    parser.add_argument('--dataset', type=str, default='im2gps3k',
                        choices=['im2gps', 'im2gps3k', 'yfcc4k'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--topk', type=int, default=20)
    args = parser.parse_args()
    
   
    from accelerate import Accelerator
    accelerator = Accelerator()
    
    os.makedirs('./index', exist_ok=True)
    index_path = f'./index/{args.index_name}.index'

    if not os.path.exists(index_path):
        if accelerator.is_main_process:
            print(f"Index not found. Building {index_path}...")
   
        build_mp16_dual_index(args)
    else:
        if accelerator.is_main_process:
            print(f"Index already exists at {index_path}")
    
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone() 
        print(f"Process {accelerator.process_index} exiting after index building...")
        exit(0)

    index = faiss.read_index(index_path)
    pos_I_path = f'./index/I_pos_{args.index_name}_{args.dataset}.npy'
    
    if not os.path.exists(pos_I_path):
        I_pos = search_and_evaluate_single_gpu(args, index, args.topk)
    else:
        print("Loading existing retrieval results...")
        I_pos = np.load(pos_I_path)
    
    evaluate(args, I_pos)
    accelerator.wait_for_everyone()

# accelerate launch --multi_gpu --num_processes=2 IndexSearch_DV_AT.py --dataset im2gps3k
# python IndexSearch_DV_AT.py --dataset im2gps3k