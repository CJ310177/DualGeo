import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.utils import MP16DualTrainDataset  
from utils.DualView_seg import DualViewG3 as DVseg
from accelerate import Accelerator, DistributedDataParallelKwargs
import warnings
warnings.filterwarnings('ignore')

def train_1epoch(dataloader, model, optimizer, accelerator):
    model.train()
    t = tqdm(dataloader, disable=not accelerator.is_local_main_process)
    for batch in t:
        rgb_images, seg_images, longitude, latitude = batch
        rgb_images = rgb_images.to(accelerator.device, non_blocking=True)
        seg_images = seg_images.to(accelerator.device, non_blocking=True)
        longitude = longitude.to(accelerator.device, non_blocking=True).float()
        latitude = latitude.to(accelerator.device, non_blocking=True).float()

        optimizer.zero_grad()
        output = model(
            rgb_images=rgb_images,
            seg_images=seg_images,
            longitude=longitude,
            latitude=latitude,
            return_loss=True
        )
        loss = output['loss']
        accelerator.backward(loss)
        optimizer.step()

        if accelerator.is_local_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            t.set_description(f'loss {loss.item():.4f}, lr {current_lr:.6f}')

def main():
    os.makedirs('checkpoints', exist_ok=True)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    if accelerator.is_local_main_process:
        print(f"Device: {accelerator.device}, Processes: {accelerator.num_processes}")

    model = DVseg(device=accelerator.device)

    # Load pre-trained location encoder (if available)
    if accelerator.is_local_main_process:
        loc_enc_path = 'your path'
        if os.path.exists(loc_enc_path):
            loc_state = torch.load(loc_enc_path, map_location=accelerator.device)
            model.location_encoder.load_state_dict(loc_state)
            print("Loaded pre-trained location encoder.")

    # Use the new unified dataset
    dataset = MP16DualTrainDataset(
        root_path='./data/',
        rgb_csv='MP16_Pro_places365.csv',
        seg_csv='mp16-seg-png.csv',
        target_size=224
    )
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=16,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=5
    )

    # Only train unfrozen parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=4e-5,
        weight_decay=1e-6
    )
    # Resume training
    resume_epoch = 0
    if accelerator.is_local_main_process:
        for epoch in range(9, -1, -1):
            cp_path = f'checkpoints/dvseg_{epoch}_.pth'
            if os.path.exists(cp_path):
                checkpoint = torch.load(cp_path, map_location=accelerator.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                resume_epoch = epoch + 1
                print(f"Resuming from epoch {resume_epoch}")
                break

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.87)

    model, optimizer, scheduler, dataloader = accelerator.prepare(
        model, optimizer, scheduler, dataloader
    )
    
    scheduler.last_epoch = resume_epoch - 1

    for epoch in range(resume_epoch, 10):
        current_lr = optimizer.param_groups[0]['lr']
        if accelerator.is_local_main_process:
            print(f"===== Epoch {epoch} (lr={current_lr:.2e}) =====")
        train_1epoch(dataloader, model, optimizer, accelerator)
        scheduler.step()

        accelerator.wait_for_everyone()

        if accelerator.is_local_main_process:
            unwrapped = accelerator.unwrap_model(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f'checkpoints/dvseg_{epoch}_.pth')
            print(f"Checkpoint saved for epoch {epoch}")
        
        accelerator.wait_for_everyone()

if __name__ == '__main__':
    main()
    # accelerate launch --num_processes=2 --mixed_precision=fp16 New_dvseg.py
    # accelerate launch --num_processes=2 New_dvseg.py