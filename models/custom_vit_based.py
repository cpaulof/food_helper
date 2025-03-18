import torch
import torch.nn as nn
import torch.nn.functional as F

class DyT(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_features))
        self.lambda_param = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        # Apply dynamic tanh directly
        if len(x.shape) == 3:  # (B, N, C)
            x = self.lambda_param * torch.tanh(self.alpha * x) + self.beta
        else:  # (B, C, H, W)
            x = self.lambda_param.view(1, -1, 1, 1) * torch.tanh(self.alpha.view(1, -1, 1, 1) * x) + self.beta.view(1, -1, 1, 1)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192, use_dyt=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = DyT(embed_dim) if use_dyt else nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        x = self.norm(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, E = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, E)
        x = self.proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, use_dyt=True):
        super().__init__()
        self.norm1 = DyT(embed_dim) if use_dyt else nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = DyT(embed_dim) if use_dyt else nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class CustomViT(nn.Module):
    def __init__(self, 
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=192,
                 depth=8,
                 num_heads=6,
                 mlp_ratio=2.0,
                 use_dyt=True):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim, use_dyt)
        num_patches = self.patch_embed.n_patches
        
        # Position embedding and cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, use_dyt)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = DyT(embed_dim) if use_dyt else nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 5)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if use_dyt:
            self._initialize_dyt_parameters()
        
    def _initialize_dyt_parameters(self):
        # Initialize DyT parameters
        for m in self.modules():
            if isinstance(m, DyT):
                nn.init.ones_(m.alpha)
                nn.init.ones_(m.lambda_param)
                nn.init.zeros_(m.beta)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add cls token and position embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Use cls token for classification
        x = self.head(x)
        
        # Apply exponential to output
        x = torch.exp(x)
        
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test both normalization variants
    model_dyt = CustomViT(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=8,
        num_heads=6,
        mlp_ratio=2.0,
        use_dyt=True
    )
    print(f"DyT Model parameters: {count_parameters(model_dyt):,}")
    
    model_ln = CustomViT(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=8,
        num_heads=6,
        mlp_ratio=2.0,
        use_dyt=False
    )
    print(f"LayerNorm Model parameters: {count_parameters(model_ln):,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output_dyt = model_dyt(x)
    output_ln = model_ln(x)
    print(f"Output shape (DyT): {output_dyt.shape}")
    print(f"Output shape (LayerNorm): {output_ln.shape}") 