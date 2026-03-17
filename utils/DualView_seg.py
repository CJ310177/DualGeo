import torch
import torch.nn as nn
import torchvision.models as models
from transformers import CLIPTokenizer, CLIPImageProcessor, CLIPModel
from .rff.layers import GaussianEncoding
from pyproj import Proj, Transformer
from transformers import ViTModel, ViTConfig

local_model_dir = "clip-vit-large-patch14"
local_ViT_dir = "vit-base-patch16-224"


class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma):
        super(LocationEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(
            rff_encoding,
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x


class CustomLocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8]):
        super(CustomLocationEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)
        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s))
        proj_wgs84 = Proj('epsg:4326')
        proj_mercator = Proj('epsg:3857')
        self.transformer = Transformer.from_proj(proj_wgs84, proj_mercator, always_xy=True)

    def forward(self, input):
        lat = input[:, 0].float().detach().cpu().numpy()
        lon = input[:, 1].float().detach().cpu().numpy()
        projected_lon_lat = self.transformer.transform(lon, lat)
        location = []
        for coord in zip(*projected_lon_lat):
            location.append([coord[1], coord[0]])
        location = torch.Tensor(location).to('cuda')
        location = location / 20037508.3427892
        location_features = torch.zeros(location.shape[0], 512).to('cuda')
        for i in range(self.n):
            location_features += self._modules['LocEnc' + str(i)](location)
        return location_features

class SemanticEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])  
        self.projection = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.features(x)  # [B, 512, 1, 1] for 224x224 input
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.projection(x)   # [B, 768]
        return x


import torch.nn.functional as F  

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj_rgb = nn.Linear(embed_dim, embed_dim)
        self.k_proj_seg = nn.Linear(embed_dim, embed_dim)
        self.v_proj_seg = nn.Linear(embed_dim, embed_dim)
        
        self.q_proj_seg = nn.Linear(embed_dim, embed_dim)
        self.k_proj_rgb = nn.Linear(embed_dim, embed_dim)
        self.v_proj_rgb = nn.Linear(embed_dim, embed_dim)
        
        self.output_proj_rgb = nn.Linear(embed_dim, embed_dim)
        self.output_proj_seg = nn.Linear(embed_dim, embed_dim)
        
        self.norm_rgb = nn.LayerNorm(embed_dim)
        self.norm_seg = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, rgb_feat, seg_feat):
        B = rgb_feat.size(0)
        
        # RGB → SEG 注意力
        q_rgb = self.q_proj_rgb(rgb_feat).view(B, self.num_heads, self.head_dim).transpose(0, 1)
        k_seg = self.k_proj_seg(seg_feat).view(B, self.num_heads, self.head_dim).transpose(0, 1)
        v_seg = self.v_proj_seg(seg_feat).view(B, self.num_heads, self.head_dim).transpose(0, 1)
        
        attn_weights_rgb = F.softmax(torch.matmul(q_rgb, k_seg.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        rgb_enhanced = torch.matmul(attn_weights_rgb, v_seg).transpose(0, 1).contiguous().view(B, self.embed_dim)
        rgb_enhanced = self.output_proj_rgb(rgb_enhanced)
        
        # SEG → RGB 注意力
        q_seg = self.q_proj_seg(seg_feat).view(B, self.num_heads, self.head_dim).transpose(0, 1)
        k_rgb = self.k_proj_rgb(rgb_feat).view(B, self.num_heads, self.head_dim).transpose(0, 1)
        v_rgb = self.v_proj_rgb(rgb_feat).view(B, self.num_heads, self.head_dim).transpose(0, 1)
        
        attn_weights_seg = F.softmax(torch.matmul(q_seg, k_rgb.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        seg_enhanced = torch.matmul(attn_weights_seg, v_rgb).transpose(0, 1).contiguous().view(B, self.embed_dim)
        seg_enhanced = self.output_proj_seg(seg_enhanced)
        
        # 残差连接 + LayerNorm
        rgb_out = self.norm_rgb(rgb_feat + self.dropout(rgb_enhanced))
        seg_out = self.norm_seg(seg_feat + self.dropout(seg_enhanced))
        
        return rgb_out, seg_out


class DualViewG3(torch.nn.Module):
    def __init__(self, device):
        super(DualViewG3, self).__init__()
        self.device = device

        clip_model = CLIPModel.from_pretrained(local_model_dir)
        self.vision_model_rgb = clip_model.vision_model
        self.vision_projection_rgb = clip_model.visual_projection
        self.vision_processor_rgb = CLIPImageProcessor.from_pretrained(local_model_dir)

        self.semantic_encoder = SemanticEncoder(output_dim=768)
        self.vision_projection_seg = nn.Linear(768, 768, bias=False)

        self.cross_attn_fusion = CrossAttentionFusion(embed_dim=768, num_heads=8)

        self.logit_scale_rgb = nn.Parameter(torch.tensor(3.99))
        self.logit_scale_seg = nn.Parameter(torch.tensor(3.99))
        self.location_encoder = CustomLocationEncoder()
        self.rgb_projection_else = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )
        self.seg_projection_else = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )
        self.location_projection_else = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 768)
        )

        self.vision_model_rgb.requires_grad_(False)
        self.vision_projection_rgb.requires_grad_(False)

    def forward(self, rgb_images, seg_images, longitude, latitude, return_loss=True):
        vision_output_rgb = self.vision_model_rgb(rgb_images)[1]  # [B, 768]
        image_embeds_rgb = self.vision_projection_rgb(vision_output_rgb)

        seg_features = self.semantic_encoder(seg_images)  # [B, 768]
        image_embeds_seg = self.vision_projection_seg(seg_features)

        enhanced_rgb, enhanced_seg = self.cross_attn_fusion(image_embeds_rgb, image_embeds_seg)
        image_embeds_rgb = enhanced_rgb
        image_embeds_seg = enhanced_seg

        this_batch_locations = torch.stack((latitude, longitude), dim=1)
        location_embeds = self.location_encoder(this_batch_locations)

        image_embeds_rgb_proj = self.rgb_projection_else(image_embeds_rgb)
        image_embeds_seg_proj = self.seg_projection_else(image_embeds_seg)
        location_embeds_proj = self.location_projection_else(location_embeds)

        image_embeds_rgb_proj = image_embeds_rgb_proj / image_embeds_rgb_proj.norm(p=2, dim=-1, keepdim=True)
        image_embeds_seg_proj = image_embeds_seg_proj / image_embeds_seg_proj.norm(p=2, dim=-1, keepdim=True)
        location_embeds_proj = location_embeds_proj / location_embeds_proj.norm(p=2, dim=-1, keepdim=True)

        # Compute logits and losses
        logit_scale_rgb = self.logit_scale_rgb.exp()
        logits_per_rgb_with_gps = torch.matmul(image_embeds_rgb_proj, location_embeds_proj.t()) * logit_scale_rgb

        logit_scale_seg = self.logit_scale_seg.exp()
        logits_per_seg_with_gps = torch.matmul(image_embeds_seg_proj, location_embeds_proj.t()) * logit_scale_seg

        loss_rgb_gps = None
        loss_seg_gps = None
        total_loss = None

        if return_loss:
            loss_rgb_gps = self.clip_loss(logits_per_rgb_with_gps)
            loss_seg_gps = self.clip_loss(logits_per_seg_with_gps)
            total_loss = loss_rgb_gps + loss_seg_gps

        return {
            'logits_per_rgb_with_gps': logits_per_rgb_with_gps,
            'logits_per_seg_with_gps': logits_per_seg_with_gps,
            'loss_rgb_gps': loss_rgb_gps,
            'loss_seg_gps': loss_seg_gps,
            'loss': total_loss,
            'rgb_image_embeds': image_embeds_rgb,      
            'seg_image_embeds': image_embeds_seg,     
            'location_embeds': location_embeds
        }

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0
#---------------------------------------------------------------------------------------------
    def extract_enhanced_features(self, rgb_images, seg_images):
  
        # RGB branch
        rgb_feat = self.vision_model_rgb(rgb_images)[1]  # [B, 768]
        rgb_feat = self.vision_projection_rgb(rgb_feat)

        # Seg branch
        seg_feat = self.semantic_encoder(seg_images)  # [B, 768]
        seg_feat = self.vision_projection_seg(seg_feat)

        # Cross-attention fusion
        enhanced_rgb, enhanced_seg = self.cross_attn_fusion(rgb_feat, seg_feat)

        return enhanced_rgb, enhanced_seg