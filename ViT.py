import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PatchEmbedding(nn.Module):

    def __init__(self, in_channels=3, patch_size=16, embeding_dim=768):
        super().__init__()

        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=embeding_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):

        x = self.flatten(self.patcher(x)) #It is more efficient setting them on the same line for operation fusion

        return x.permute(0, 2, 1)
    
class MultiheadSelfAttentionBock(nn.Module):

    def __init__(self, embedding_dim=768, num_heads=12, attn_dropout=0):
        super().__init__()

        self.norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)


    def forward(self, x):
        x = self.norm(x)
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x, need_wheights=False)

        return attn_output
    
class MLPBlock(nn.Module):

    def __init__(self, embbeding_dim=768, mlp_size=3072, dropout=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(normalized_shape=embbeding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embbeding_dim, out_features=mlp_size)
            nn.GELU(),
            nn.Dropout(p=dropout)
            nn.Linear(in_features=mlp_size, out_features=embbeding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):

        return self.mlp(self.norm(x)) #Efficient for operator fusion

class TransformerEncoderBlock(nn.Module):

    def __init__(self, embbeding_dim=768, num_heads=12, mlp_size=3072, mlp_dropout=0.1, attn_dropout=0):
        super().__init__()

        self.msa_block = MultiheadSelfAttentionBock(embedding_dim=embbeding_dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.mlp_block = MLPBlock(embbeding_dim=embbeding_dim, mlp_size=mlp_size, dropout=mlp_dropout)

    def forward(self, x):

        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x

class ViT(nn.Module):

    def __init__(self, img_size=224, in_channels=3, patch_size=16, num_transformers_layers=12, embbeding_dim=768,
                 mlp_size=3072, num_heads=12, attn_dropout=0, mlp_dropout=0.1, embbeding_dropout=0.1, num_classes=1000):
        super().__init__()

        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        self.num_patches = (img_size * img_size) // patch_size**2

        self.class_embedding = nn.Parameter(torch.randn(1, 1, embbeding_dim), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches+1), requires_grad=True)

        self.embedding_dropout = nn.Dropout(p=embbeding_dropout)

        self.patch_embedding = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embeding_dim=embbeding_dim)

        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embbeding_dim=embbeding_dim, num_heads=num_heads,
                                                                           mlp_size=mlp_size, mlp_dropout=mlp_dropout, attn_dropout=attn_dropout)
                                                                           for _ in range(num_transformers_layers)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embbeding_dim),
            nn.Linear(in_features=embbeding_dim, out_features=num_classes)
        )

    def forward(self, x):

        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])

        return x