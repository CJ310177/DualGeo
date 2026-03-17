import os
import pickle
from pathlib import Path
from typing import Optional
from PIL import Image, ImageFile, ImageFilter
from torch.utils.data import DataLoader, get_worker_info
from torchvision.datasets import VisionDataset
from torchvision import transforms as T
from torchvision.transforms.functional import to_tensor
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import tarfile
from torch.utils.data import Dataset as TorchDataset


ImageFile.LOAD_TRUNCATED_IMAGES = True


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

class MP16DualTrainDataset(TorchDataset):
    def __init__(self, root_path='./data/', 
                 rgb_csv='MP16_Pro_places365.csv',
                 seg_csv='mp16-seg-png.csv',
                 target_size=224):
        self.root_path = Path(root_path)
        self.target_size = target_size
        
        # Load and align CSVs
        rgb_df = pd.read_csv(self.root_path / rgb_csv)
        seg_df = pd.read_csv(self.root_path / seg_csv)

        rgb_df = rgb_df[rgb_df['country'].notnull()].reset_index(drop=True)
        rgb_df['COMMON_ID'] = rgb_df['IMG_ID'].str.replace('/', '_', regex=False).str.replace('.jpg', '', regex=False)
        seg_df['COMMON_ID'] = seg_df['IMG_ID'].apply(lambda x: x.replace('.png', '').replace('/', '_'))

        merged = pd.merge(
            rgb_df[['IMG_ID', 'COMMON_ID', 'LON', 'LAT']],
            seg_df[['IMG_ID', 'COMMON_ID']].rename(columns={'IMG_ID': 'SEG_IMG_ID'}),
            on='COMMON_ID',
            how='inner'
        ).reset_index(drop=True)

        self.data = merged
        print(f"Aligned {len(self.data)} training samples.")

        # Load indices
        with open(self.root_path / 'tar_index.pkl', 'rb') as f:
            self.tar_index = pickle.load(f)
        with open(self.root_path / 'png_index.pkl', 'rb') as f:
            self.png_index = pickle.load(f)

        self.tar_path = self.root_path / 'mp-16-images.tar'

        # Transforms
        self.resize = T.Resize((target_size, target_size))
        self.normalize_rgb = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        rgb_key = row['IMG_ID']
        seg_key = row['SEG_IMG_ID']
        lon = float(row['LON'])
        lat = float(row['LAT'])

        # Load RGB
        with tarfile.open(self.tar_path, 'r') as tar:
            member = self.tar_index[rgb_key.replace('/', '_')]
            rgb_file = tar.extractfile(member)
            rgb_img = Image.open(rgb_file).convert('RGB')

        # Load Seg
        seg_path = self.png_index[seg_key]
        seg_img = Image.open(seg_path).convert('L')

        # Random horizontal flip
        if torch.rand(1) < 0.5:
            rgb_img = T.functional.hflip(rgb_img)
            seg_img = T.functional.hflip(seg_img)

        # Random resized crop
        i, j, h, w = T.RandomResizedCrop.get_params(
            rgb_img, scale=(0.8, 1.0), ratio=(0.9, 1.1)
        )
        rgb_img = T.functional.crop(rgb_img, i, j, h, w)
        seg_img = T.functional.crop(seg_img, i, j, h, w)

        # Resize to target (e.g., 224x224)
        rgb_img = self.resize(rgb_img)
        seg_img = self.resize(seg_img)

        # Color augmentations (RGB only)
        if torch.rand(1) < 0.8:
            rgb_img = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)(rgb_img)
        if torch.rand(1) < 0.2:
            rgb_img = T.Grayscale(num_output_channels=3)(rgb_img)
        if torch.rand(1) < 0.5:
            rgb_img = rgb_img.filter(ImageFilter.GaussianBlur(radius=1))

        # Convert to tensor
        rgb_tensor = to_tensor(rgb_img)      # [3, H, W], range [0,1]
        seg_tensor = to_tensor(seg_img)      # [1, H, W], range [0,1]

        # Apply CLIP normalization to RGB (after [0,1])
        rgb_tensor = self.normalize_rgb(rgb_tensor)

        return rgb_tensor, seg_tensor, lon, lat

class MP16Dataset(VisionDataset):
    def __init__(
        self,
        root_path='./data/',
        text_data_path='MP16_Pro_places365.csv',
        image_data_path='mp-16-images.tar',
        member_info_path='tar_index.pkl',
        vision_processor=None,
        seg_csv_path='mp16-seg-png.csv',
        text_processor=None  
    ):
        super().__init__(root_path)
        self.root_path = Path(root_path)
        self.text_data_path = text_data_path
        self.image_data_path = image_data_path
        self.seg_csv_path = seg_csv_path

        # Load RGB CSV
        rgb_csv = pd.read_csv(self.root_path / self.text_data_path)
        print(f"Loaded {len(rgb_csv)} entries from RGB CSV.")

        rgb_csv['IMG_ID'] = rgb_csv['IMG_ID'].str.replace('/', '_', regex=False)

        rgb_csv = rgb_csv[rgb_csv['country'].notnull()].reset_index(drop=True)

        worker = get_worker_info()
        worker_id = worker.id if worker else None
        self.tar_obj = {worker_id: tarfile.open(self.root_path / image_data_path)}

        tar_index_full_path = self.root_path / member_info_path
        if tar_index_full_path.exists():
            with open(tar_index_full_path, 'rb') as f:
                self.tar_index = pickle.load(f)
            all_image_names = set(self.tar_index.keys())
            print('Loaded tar index successfully.')
        else:
            print('Building tar index...')
            self.tar_index = {}
            all_image_names = []
            for member in tqdm(self.tar_obj[worker_id], desc="Indexing tar"):
                if not (member.isfile() and member.name.endswith('.jpg') and member.size > 5120):
                    continue
                parts = member.name.split('/')
                if len(parts) < 3:
                    continue
                # Convert 'xx/yy/zzz.jpg' → 'xx_yy_zzz.jpg'
                img_id = f"{parts[-3]}_{parts[-2]}_{parts[-1]}"
                self.tar_index[img_id] = member
                all_image_names.append(img_id)
            print(f'Tar index built with {len(all_image_names)} images.')
            with open(tar_index_full_path, 'wb') as f:
                pickle.dump(self.tar_index, f)
            all_image_names = set(all_image_names)

        # Filter using consistent format
        rgb_csv = rgb_csv[rgb_csv['IMG_ID'].isin(all_image_names)].reset_index(drop=True)
        print(f"Filtered to {len(rgb_csv)} RGB images present in tar.")

        # Align with Seg dataset
        seg_csv = pd.read_csv(self.root_path / self.seg_csv_path)
        # Convert seg IMG_ID: 'xx/yy/zzz.png' → 'xx_yy_zzz.jpg'
        seg_rgb_ids = set(
            seg_csv['IMG_ID'].apply(
                lambda x: x.replace('.png', '').replace('/', '_') + '.jpg'
            )
        )
        aligned_rgb = rgb_csv[rgb_csv['IMG_ID'].isin(seg_rgb_ids)].reset_index(drop=True)
        print(f"After alignment with Seg dataset: {len(aligned_rgb)} RGB images.")

        self.text_data = aligned_rgb
        self.text_data['LON'] = self.text_data['LON'].astype(float)
        self.text_data['LAT'] = self.text_data['LAT'].astype(float)
        self.vision_processor = vision_processor

        # Contrastive transforms
        self.contrast_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(size=224),
            T.RandomApply([
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=9),
            T.ToTensor()
        ])

    def __getitem__(self, index):
        row = self.text_data.iloc[index]
        img_key = row['IMG_ID']  
        location_elements = [
            elem for elem in [row['city'], row['state'], row['country']]
            if pd.notna(elem) and str(elem).strip() != ''
        ]
        text = 'A street view photo taken in ' + ', '.join(location_elements) if location_elements else 'A street view photo.'
        longitude = float(row['LON'])
        latitude = float(row['LAT'])

        worker = get_worker_info()
        worker_id = worker.id if worker else None

        if worker_id not in self.tar_obj:
            self.tar_obj[worker_id] = tarfile.open(self.root_path / self.image_data_path)

        image_file = self.tar_obj[worker_id].extractfile(self.tar_index[img_key])
        image = Image.open(image_file).convert('RGB')

        if self.vision_processor:
            image = self.vision_processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
        else:
            image = self.contrast_transforms(image)

        return image, text, longitude, latitude

    def __len__(self):
        return len(self.text_data)


class MP16SegDataset(VisionDataset):
    def __init__(
        self,
        root_path: str = './data/',
        ext_data_path: str = 'mp16-seg-png.csv',
        image_data_path: str = 'mp-16-pngs',
        member_info_path: str = 'png_index.pkl',
        vision_processor=None,
        target_size: int = 224,
        rgb_csv_path: str = 'MP16_Pro_places365.csv',
        text_processor=None  
    ):
        super().__init__(root_path)
        self.root_path = Path(root_path)
        self.image_dir = self.root_path / image_data_path
        self.csv_path = self.root_path / ext_data_path
        self.rgb_csv_path = rgb_csv_path
        self.member_info_path = self.root_path / member_info_path
        self.target_size = target_size
        self.transform = T.Compose([
            T.Resize((target_size, target_size)),
            T.ToTensor()
        ])

        seg_csv = pd.read_csv(self.csv_path)
        # Normalize seg IMG_ID to common format WITHOUT extension
        seg_csv['COMMON_ID'] = seg_csv['IMG_ID'].apply(lambda x: x.replace('.png', '').replace('/', '_'))
        print(f"Loaded {len(seg_csv)} entries from Seg CSV.")

        png_index_path = self.member_info_path
        if png_index_path.exists():
            with open(png_index_path, 'rb') as f:
                self.png_index = pickle.load(f)
            all_png_relpaths = set(self.png_index.keys())
            print("Loaded PNG index successfully.")
        else:
            print("Building PNG index...")
            self.png_index = {}
            all_png_relpaths = set()
            for img_path in tqdm(self.image_dir.rglob("*.png"), desc="Indexing PNGs"):
                rel_path = img_path.relative_to(self.image_dir).as_posix()
                self.png_index[rel_path] = img_path
                all_png_relpaths.add(rel_path)
            with open(png_index_path, 'wb') as f:
                pickle.dump(self.png_index, f)
            print(f"PNG index built and saved to {png_index_path}")

        seg_csv = seg_csv[seg_csv['IMG_ID'].isin(all_png_relpaths)].reset_index(drop=True)
        print(f"Filtered to {len(seg_csv)} valid Seg images.")

        # Load RGB CSV and normalize its IMG_ID for alignment
        rgb_csv = pd.read_csv(self.root_path / self.rgb_csv_path)
        rgb_csv = rgb_csv[rgb_csv['country'].notnull()]
        # RGB IMG_ID is like 'xx/yy/zzz.jpg' → convert to 'xx_yy_zzz'
        rgb_common_ids = set(rgb_csv['IMG_ID'].str.replace('/', '_', regex=False).str.replace('.jpg', '', regex=False))
        aligned_seg = seg_csv[seg_csv['COMMON_ID'].isin(rgb_common_ids)].reset_index(drop=True)
        print(f"After alignment with RGB dataset: {len(aligned_seg)} Seg images.")

        self.text_data = aligned_seg
        self.text_data['LAT'] = self.text_data['LAT'].astype(float)
        self.text_data['LON'] = self.text_data['LON'].astype(float)

    def __getitem__(self, index):
        row = self.text_data.iloc[index]
        img_id = row['IMG_ID']
        lat = float(row['LAT'])
        lon = float(row['LON'])
        img_path = self.png_index[img_id]
        image = Image.open(img_path).convert('L')
        image_tensor = self.transform(image)
        return image_tensor, img_id, lon, lat

    def __len__(self):
        return len(self.text_data)

class DualViewTestDataset(TorchDataset):
    def __init__(self, rgb_dir, seg_dir, rgb_csv, seg_csv,
                 vision_processor_rgb=None, target_size=224):
        self.rgb_dir = Path(rgb_dir)
        self.seg_dir = Path(seg_dir)
        self.vision_processor_rgb = vision_processor_rgb
        self.target_size = target_size
        
        # Load CSVs
        rgb_df = pd.read_csv(rgb_csv)
        seg_df = pd.read_csv(seg_csv)
        
        # Normalize IMG_ID for alignment
        # RGB: 'xxx.jpg' -> 'xxx'
        # Seg: 'xxx.png' -> 'xxx'
        rgb_df['COMMON_ID'] = rgb_df['IMG_ID'].apply(lambda x: os.path.splitext(x)[0])
        seg_df['COMMON_ID'] = seg_df['IMG_ID'].apply(lambda x: os.path.splitext(x)[0])
        
        # Align by COMMON_ID
        merged = pd.merge(rgb_df, seg_df, on='COMMON_ID', suffixes=('_rgb', '_seg'))
        print(f"Aligned {len(merged)} samples from RGB and Seg datasets.")
        
        self.data = merged
        self.transform_seg = T.Compose([
            T.Resize((target_size, target_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # RGB image
        rgb_path = self.rgb_dir / row['IMG_ID_rgb']
        rgb_img = Image.open(rgb_path).convert('RGB')
        if self.vision_processor_rgb:
            rgb_img = self.vision_processor_rgb(images=rgb_img, return_tensors='pt')['pixel_values'].squeeze(0)
        else:
            rgb_img = T.Compose([
                T.Resize((self.target_size, self.target_size)),
                T.ToTensor()
            ])(rgb_img)
        
        # Segmentation image
        seg_path = self.seg_dir / row['IMG_ID_seg']
        seg_img = Image.open(seg_path).convert('L')
        seg_img = self.transform_seg(seg_img)  # [1, H, W]
        
        # Coordinates (use RGB's or Seg's, they should be same)
        lon = float(row['LON_rgb']) if 'LON_rgb' in row else float(row['LON'])
        lat = float(row['LAT_rgb']) if 'LAT_rgb' in row else float(row['LAT'])
        
        return rgb_img, seg_img, str(row['IMG_ID_rgb']), lon, lat


# Specific datasets
class Im2gpsDualDataset(DualViewTestDataset):
    def __init__(self, root_path='./data', vision_processor_rgb=None):
        super().__init__(
            rgb_dir=os.path.join(root_path, 'im2gps_rgb_images'),
            seg_dir=os.path.join(root_path, 'im2gps_seg'),
            rgb_csv=os.path.join(root_path, 'im2gps.csv'),
            seg_csv=os.path.join(root_path, 'im2gps_seg.csv'),
            vision_processor_rgb=vision_processor_rgb
        )

class Im2gps3kDualDataset(DualViewTestDataset):
    def __init__(self, root_path='./data', vision_processor_rgb=None):
        super().__init__(
            rgb_dir=os.path.join(root_path, 'im2gps3ktest'),
            seg_dir=os.path.join(root_path, 'im2gps3k_seg'),
            rgb_csv=os.path.join(root_path, 'im2gps3k_places365.csv'),
            seg_csv=os.path.join(root_path, 'im2gps3k_seg.csv'),
            vision_processor_rgb=vision_processor_rgb
        )

class Yfcc4kDualDataset(DualViewTestDataset):
    def __init__(self, root_path='./data', vision_processor_rgb=None):
        super().__init__(
            rgb_dir=os.path.join(root_path, 'yfcc4k/yfcc4k'),
            seg_dir=os.path.join(root_path, 'yfcc4k_seg'),
            rgb_csv=os.path.join(root_path, 'yfcc4k_places365.csv'),
            seg_csv=os.path.join(root_path, 'yfcc4k_seg.csv'),
            vision_processor_rgb=vision_processor_rgb
        )