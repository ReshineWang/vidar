import torch
import torch.nn as nn
import os

from .resnet import *
from .unet import *


class IDM(nn.Module):

    def __init__(self, model_name, *args, **kwargs):
        super(IDM, self).__init__()
        match model_name:
            case "mask":
                self.model = Mask(*args, **kwargs)
            case "unet":
                self.model = Unet(*args, **kwargs)
            case "resnet":
                self.model = ResNet(*args, **kwargs)
            case "dino-resnet":
                self.model = DinoResNet(*args, **kwargs)
            case "dino-transformer":
                self.model = DinoTransformer(*args, **kwargs)
            case "resnet_plus":
                self.model = Resnet_plus(*args, **kwargs)
            case _:
                raise ValueError(f"Unsupported model name: {model_name}")
            
        train_mean = torch.tensor([-0.26866713, 0.83559588, 0.69520934, -0.29099351, 0.18849116, -0.01014598, 1.41953145, 0.35073715, 1.05651613, 0.8930193, -0.37493264, -0.18510782, -0.0272574, 1.35274259])
        train_std = torch.tensor([0.25945241, 0.65903812, 0.52147207, 0.42150272, 0.32029947, 0.28452226, 1.78270006, 0.29091741, 0.67675932, 0.58250554, 0.42399049, 0.28697442, 0.31100304, 1.67651926])
        
        self.register_buffer("train_mean", train_mean)
        self.register_buffer("train_std", train_std)

    def normalize(self, x):
        x = (x - self.train_mean) / self.train_std
        return x

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if isinstance(output, tuple):
            return output[0] * self.train_std + self.train_mean, *output[1:]
        else:
            return output * self.train_std + self.train_mean


class DinoResNet(nn.Module):
    def __init__(self, output_dim=14, dino_version="dinov3_vitl16", *args, **kwargs):
        super().__init__()
        self.output_dim = output_dim
        
        print(f"Loading DinoV3 model from torch.hub: facebookresearch/dinov3 - {dino_version}")
        # Note: This requires network access or cached model to get the structure.
        # We load the structure only (pretrained=False) then load local weights.
        # self.backbone = torch.hub.load('facebookresearch/dinov3', dino_version, pretrained=False)
        self.backbone = torch.hub.load(
            repo_or_dir="/data/dex/dinov3",
            model=dino_version,          # e.g. "dinov3_vitl16"
            pretrained=False,
            source="local",              # 关键！
        )  
        
        # Load local checkpoint
        ckpt_path = "/data/dex/vidar/vidar_ckpts/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        print(f"Loading local weights from: {ckpt_path}")
        
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            # Handle common DINO wrapper keys
            if "teacher" in state_dict:
                state_dict = state_dict["teacher"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            
            # Clean up state dictionary keys if necessary (e.g. remove 'backbone.' or 'module.' prefix)
            # This handles cases where the saved model was wrapped in DDP or a larger implementation
            new_state_dict = {}
            for k, v in state_dict.items():
                k_clean = k.replace("module.", "").replace("backbone.", "")
                new_state_dict[k_clean] = v
                # Also keep original just in case
                new_state_dict[k] = v
            
            msg = self.backbone.load_state_dict(new_state_dict, strict=False)
            print(f"Weights loaded. {msg}")
        else:
            print(f"WARNING: Local checkpoint not found at {ckpt_path}. Using random initialization!")

        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # Infer embed_dim from the model
        if hasattr(self.backbone, "embed_dim"):
            self.embed_dim = self.backbone.embed_dim
        else:
            # Fallback heuristics
            if "vitl" in dino_version: self.embed_dim = 1024
            elif "vitb" in dino_version: self.embed_dim = 768
            elif "vits" in dino_version: self.embed_dim = 384
            elif "vitg" in dino_version: self.embed_dim = 1536
            else: self.embed_dim = 1024 # Default guess
            
        print(f"Dino backbone loaded. Embed dim: {self.embed_dim}. Backbone is FROZEN.")

        # ResNet head
        # Input to ResNet will be the feature map from Dino
        # shape: (B, Embed_Dim, H_patch, W_patch)
        self.resnet_model = ResNet(output_dim, input_channels=self.embed_dim)
        
        # Print number of parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"DinoResNet initialized. Trainable parameters (ResNet only): {trainable_params}")

    def forward(self, images, return_mask=False, *args, **kwargs):
        # images: [B, 3, H, W]
        # Dino expects normalized images. Our processor does that.
        
        with torch.no_grad():
            # Get features from Dino
            # forward_features returns a dict with 'x_norm_patchtokens' typically for DINOv2/v3
            # x_norm_patchtokens: [B, N, C]
            features_dict = self.backbone.forward_features(images)
            patch_tokens = features_dict['x_norm_patchtokens']
            
            B, N, C = patch_tokens.shape
            # Assuming square image and patch size 14
            # e.g. 518 / 14 = 37. 37*37 = 1369.
            H_grid = W_grid = int(N ** 0.5)
            
            # Reshape to [B, C, H_grid, W_grid]
            feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H_grid, W_grid)
        
        # Pass feature map to ResNet
        outputs = self.resnet_model(feature_map)
        
        if return_mask:
            # This architecture does not predict masks, but IDM interface might ask for it
            return outputs, None
        else:
            return outputs


class Unet(nn.Module):
    def __init__(self, output_dim: int = 14, *args, **kwargs):

        super().__init__()
        self.output_dim = output_dim

        self.mask_model = UNet(3, 3)
        self.resnet_model = ResNet(14, 3)

        # Print number of parameters
        print(f"output_dim: {output_dim}, parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, images, *args, **kwargs):
        outputs = self.resnet_model(torch.tanh(self.mask_model(images)))
        return outputs


class Mask(nn.Module):
    def __init__(self, output_dim: int = 14, *args, **kwargs):

        super().__init__()
        self.output_dim = output_dim

        self.mask_model = UNet(3, 1)
        self.resnet_model = ResNet(14, *args, **kwargs)

        # Print number of parameters
        print(f"output_dim: {output_dim}, parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, images, return_mask=False, *args, **kwargs):
        mask = (1 + torch.tanh(self.mask_model(images))) / 2
        mask_hard = torch.where(mask < 0.5, 0.0, 1.0)

        ## 这里反向传播的时候允许梯度通过mask
        inputs = images * ((mask_hard - mask).detach() + mask)
        outputs = self.resnet_model(inputs)
        if return_mask:
            return outputs, mask
        else:
            return outputs      


class DinoTransformer(nn.Module):
    def __init__(self,
                 dino_version="dinov3_vitl16",
                 num_frames=2,
                 transformer_embed_dim=1024, 
                 output_dim=14, 
                 nhead=16, 
                 num_layers=6,
                 *args, **kwargs
                 ):      
        """
        Args:
            dino_version: DINOv3 model version string (e.g., "dinov3_vitl16")
            num_frames: 输入的时间窗口长度 (1, 2 或 3)
            dino_embed_dim: DINO 输出的特征维度 (ViT-L 为 1024)
            transformer_embed_dim: Head 的维度 (建议保持 1024 以保留信息)
            output_dim: 输出动作维度 (例如 14)
        """
        super().__init__()

        print(f"Loading DinoV3 model from torch.hub: facebookresearch/dinov3 - {dino_version}")
        # Note: This requires network access or cached model to get the structure.
        # We load the structure only (pretrained=False) then load local weights.
        # self.backbone = torch.hub.load('facebookresearch/dinov3', dino_version, pretrained=False)
        self.backbone = torch.hub.load(
            repo_or_dir="/data/dex/dinov3",
            model=dino_version,          # e.g. "dinov3_vitl16"
            pretrained=False,
            source="local",              # 关键！
        )  
        
        # Load local checkpoint
        ckpt_path = "/data/dex/vidar/vidar_ckpts/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        print(f"Loading local weights from: {ckpt_path}")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            # Handle common DINO wrapper keys
            if "teacher" in state_dict:
                state_dict = state_dict["teacher"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            
            # Clean up state dictionary keys if necessary (e.g. remove 'backbone.' or 'module.' prefix)
            # This handles cases where the saved model was wrapped in DDP or a larger implementation
            new_state_dict = {}
            for k, v in state_dict.items():
                k_clean = k.replace("module.", "").replace("backbone.", "")
                new_state_dict[k_clean] = v
                # Also keep original just in case
                new_state_dict[k] = v
            
            msg = self.backbone.load_state_dict(new_state_dict, strict=False)
            print(f"Weights loaded. {msg}")
        else:
            print(f"WARNING: Local checkpoint not found at {ckpt_path}. Using random initialization!")

        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # Infer embed_dim from the model
        if hasattr(self.backbone, "embed_dim"):
            self.embed_dim = self.backbone.embed_dim
        else:
            # Fallback heuristics
            if "vitl" in dino_version: self.embed_dim = 1024
            elif "vitb" in dino_version: self.embed_dim = 768
            elif "vits" in dino_version: self.embed_dim = 384
            elif "vitg" in dino_version: self.embed_dim = 1536
            else: self.embed_dim = 1024 # Default guess
            
        print(f"Dino backbone loaded. Embed dim: {self.embed_dim}. Backbone is FROZEN.")

        # Determine patch size and grid size
        # Assuming typical DINO naming convention (e.g. vitl16 -> patch size 16)
        if "16" in dino_version:
            self.patch_size = 16
        elif "14" in dino_version:
            self.patch_size = 14
        else:
            self.patch_size = 16 # Default fallback
            
        # Hardcoded image size matching DinoPreprocessor
        self.img_size = 512 
        self.grid_size = self.img_size // self.patch_size
        self.num_patches = self.grid_size ** 2
        print(f"Model Config: Img Size {self.img_size}, Patch Size {self.patch_size}, Num Patches {self.num_patches}")

        self.input_proj = nn.Linear(self.embed_dim, transformer_embed_dim)
        # 3. 时空位置编码 (Spatio-Temporal Embeddings)
        # 关键！这是模型区分 "上一帧的夹爪" 和 "这一帧的夹爪" 的唯一依据。
        
        # 空间编码: [1, 1, N_patches, D]
        # 作用：告诉模型 "这是图片的左上角还是右下角"
        self.pos_embed_spatial = nn.Parameter(torch.randn(1, 1, self.num_patches, transformer_embed_dim) * 0.02)
        
        # 时间编码: [1, T_frames, 1, D]
        # 作用：告诉模型 "这是 t-1 时刻 还是 t 时刻"
        self.pos_embed_temporal = nn.Parameter(torch.randn(1, num_frames, 1, transformer_embed_dim) * 0.02)
        
        # 4. 状态查询 Token (State Query)
        # 这是一个 "探针"，专门用来从混乱的像素特征中把关节角度 "吸" 出来
        self.state_query = nn.Parameter(torch.randn(1, 1, transformer_embed_dim) * 0.02)
        
        # 5. Transformer Encoder (核心交互层)
        # 这里发生了 "光流计算" 和 "特征对比"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim,
            nhead=nhead,                # 多头注意力 (16头能捕捉更细微的几何关系)
            dim_feedforward=transformer_embed_dim * 4, # MLP 宽一点好
            dropout=0.1,
            activation='gelu',
            batch_first=True,           # 必须为 True，符合 A800 优化习惯
            norm_first=True             # Pre-Norm 结构，训练收敛更稳
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 6. 回归头 (Regression Head)
        # 从 Query 的特征映射到 14 个关节角
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(transformer_embed_dim),
            nn.Linear(transformer_embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, *args, **kwargs):
        """
        Input:  [Batch, Frames, 3, H, W]  (例如: 32, 2, 3, 512, 512)
        Output: [Batch, 14]               (当前帧的关节角度)
        """
        B, T, C, H, W = x.shape
        
        # ------------------------------------------------------------------
        # Step 1: 冻结骨干提取特征 (Frozen Backbone Extraction)
        # ------------------------------------------------------------------
        # 物理意义：把图片变成高维语义向量。DINO 负责看懂 "这是夹爪"，"这是物体"。
        
        # 合并 Batch 和 Time 维度以便并行计算: [B*T, 3, H, W]
        x_flat = x.view(B * T, C, H, W)
        
        with torch.no_grad():
            # DINOv3 output dict
            features_dict = self.backbone.forward_features(x_flat)
            # 获取 Patch Tokens: [B*T, N_patches, 1024]
            patch_tokens = features_dict['x_norm_patchtokens']
            
        # 恢复维度: [B, T, N, 1024]
        # 这一步如果不做，后面就没法加时间编码
        _, N, D = patch_tokens.shape
        x_feats = patch_tokens.view(B, T, N, D)
        
        # ------------------------------------------------------------------
        # Step 2: 特征融合与位置注入 (Fusion & Embedding)
        # ------------------------------------------------------------------
        # 物理意义：给特征打上 "时空标签"。
        # 如果没有这一步，模型分不清 t 和 t-1，也就无法感知速度。
        
        # 投影到 Transformer 的维度 (如果是 1024->1024 则主要起非线性整理作用)
        x_feats = self.input_proj(x_feats)
        
        # 广播加法 (Broadcasting Add):
        # x_feats: [B, T, N, D]
        # spatial: [1, 1, N, D] -> 广播到每一帧 T
        # temporal:[1, T, 1, D] -> 广播到每一个 Patch N
        x_feats = x_feats + self.pos_embed_spatial + self.pos_embed_temporal
        
        # ------------------------------------------------------------------
        # Step 3: 序列展平与 Query 拼接 (Flatten & Append Query)
        # ------------------------------------------------------------------
        # 物理意义：打破帧的界限，把 t 和 t-1 的所有像素放在一个池子里。
        
        # [B, T, N, D] -> [B, T*N, D]
        # 现在序列长度变成了 2 * 1024 = 2048
        x_sequence = x_feats.view(B, T * N, -1)
        
        # 扩展 Query 到 Batch 维度: [B, 1, D]
        query_token = self.state_query.expand(B, -1, -1)
        
        # 拼接到最前面: [B, 1 + T*N, D]
        # Query 在第 0 位，后面跟着几千个像素特征
        transformer_input = torch.cat([query_token, x_sequence], dim=1)
        
        # ------------------------------------------------------------------
        # Step 4: 交叉注意力交互 (Transformer Interaction)
        # ------------------------------------------------------------------
        # 物理意义：核心步骤。
        # Query Token 会去计算它与 t 帧夹爪像素、t-1 帧夹爪像素的 Attention。
        # 如果 t 帧和 t-1 帧有差异（由于运动），Attention 权重会敏锐地捕捉到。
        
        x_out = self.transformer(transformer_input)
        
        # ------------------------------------------------------------------
        # Step 5: 状态读出 (Readout)
        # ------------------------------------------------------------------
        # 物理意义：只取 Query Token (索引 0) 的结果。
        # 因为只有它聚合了全局信息，其他的 tokens 只是被 "看" 的对象。
        
        state_feat = x_out[:, 0] # [B, D]
        
        # 回归出最终角度
        qpos_pred = self.mlp_head(state_feat) # [B, 14]
        
        return qpos_pred