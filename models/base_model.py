from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    '''
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)
    '''
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)           # work with diff dim tensors
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    '''
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)
    '''
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    '''
    2D Image to Patch Embedding
    '''
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 input_c=3,
                 embed_dim=768,
                 norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_pathches = self.grid_size[0] * self.grid_size[1]
        self.project_conv = nn.Conv2d(input_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]

        # flatten: [B, C, H, W] -> [B, C, H*W]
        # transpose: [B, C, H*W] -> [B, H*W, C]
        out = self.project_conv(x)
        out = out.flatten(2)
        out = out.transpose(1, 2)
        out = self.norm(out)

        return out



class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.0,
                 proj_drop_ratio=0.0):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attention_drop = nn.Dropout(attn_drop_ratio)
        self.project_conv = nn.Linear(dim, dim)
        self.project_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape


        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_head, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_head, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]


        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads,  num_patches + 1, num_patches + 1]
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attention_drop(attention)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attention @ v).transpose(1, 2).reshape(B, N, C)
        out = self.project_conv(x)
        out = self.project_drop(out)

        return out



class MLP(nn.Module):
    '''
    MLP block module
    '''
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 activation_layer=nn.GELU,
                 drop_rate=0.0):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.ac = activation_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        out = self.fc1(x)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.dropout(out)

        return out



class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.0,
                 attention_drop_ratio=0.0,
                 drop_path_ratio=0.0,
                 activation_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attention = Attention(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   qk_scale=qk_scale,
                                   attn_drop_ratio=attention_drop_ratio,
                                   proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_block = MLP(in_features=dim,
                             hidden_features=mlp_hidden_dim,
                             activation_layer=activation_layer,
                             drop_rate=drop_ratio)

    def forward(self, x):
        out = self.norm1(x)
        out = self.attention(out)
        out = self.drop_path(out)
        out = x + out
        out = out + self.drop_path(self.mlp_block(self.norm2(out)))

        return out



class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 input_c=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 representation_size=None,
                 distilled=False,
                 drop_ratio=0.0,
                 attention_drop_ratio=0.0,
                 drop_path_ratio=0.0,
                 embed_layer=PatchEmbed,
                 norm_layer=None,
                 ac_layer=None):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        ac_layer = ac_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, input_c=input_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_pathches

        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.position_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.position_drop = nn.Dropout(p=drop_ratio)

        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attention_drop_ratio=attention_drop_ratio, drop_path_ratio=drop_path_rate[i],
                  norm_layer=norm_layer, activation_layer=ac_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("ac", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # classifier heads
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.position_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.class_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        out = self.patch_embed(x)
        # [1, 1, 768] -> [B, 1, 768]
        class_token = self.class_token.expand(out.shape[0], -1, -1)
        if self.dist_token is None:
            out = torch.cat((class_token, out), dim=1)  # [B, 197, 768]
        else:
            out = torch.cat((class_token, self.dist_token.expand(out.shape[0], -1, -1), out), dim=1)

        out += self.position_embed
        out = self.position_drop(out)
        out = self.blocks(out)
        out = self.norm(out)
        if self.dist_token is None:
            return self.pre_logits(out[:, 0])
        else:
            return out[:, 0], out[:, 1]

    def forward(self, x):
        out = self.forward_features(x)
        if self.head_dist is not None:
            out, out_dist = self.head(out[0]), self.head_dist(out[1])
            if self.training and not torch.jit.is_scripting():
               return out, out_dist
            else:
                return (out + out_dist) / 2
        else:
            out = self.head(out)

        return out




def _init_vit_weights(m):
    '''
    Vision Transformer weight initialization
    :param m: module
    '''
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)



def vit_base_patch16_224(num_classes: int == 21843, has_logits: bool = True):
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int == 21843, has_logits: bool = True):
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int == 21843, has_logits: bool = True):
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224(num_classes: int == 21843, has_logits: bool = True):
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model

def vit_huge_patch16_224(num_classes: int == 21843, has_logits: bool = True):
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16 ,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model