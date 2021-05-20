import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class Patches(nn.Module):
    
    def __init__(self, in_channels = 3, emd_size = 768, patch_size = 16, image_size = 224):
        self.patch_size = patch_size
        super().__init__()
        self.linear_projection = nn.Sequential(
            nn.Conv2d(in_channels, emd_size, patch_size, patch_size),
            Rearrange('b e h w -> b (h w) e')
        )
        # class token and position embeding are learnable parameter
        self.class_token = nn.Parameter(torch.randn(1,1,emd_size))
        # give postion to each patches including class token.
        self.position = nn.Parameter(torch.randn((image_size // patch_size) ** 2 + 1, emd_size))
    
    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.linear_projection(x)
        class_token = repeat(self.class_token, '() n e -> b n e', b = b)
        x = torch.cat([class_token, x], dim=1)
        x += self.position
        return x

class Normalize(nn.Module):
    def __init__(self, emd_size, fnc):
        super().__init__()
        self.norm = nn.LayerNorm(emd_size)
        self.fnc = fnc
    
    def forward(self, x):
        return self.fnc(self.norm(x))


class MultiHeadAttention(nn.Module):

    def __init__(self, emd_size = 768, no_heads = 8, dropout = 0.):
        super().__init__()
        self.emd_size = emd_size
        self.no_heads = no_heads
        self.keys = nn.Linear(emd_size, emd_size)
        self.queries = nn.Linear(emd_size, emd_size)
        self.values = nn.Linear(emd_size, emd_size)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(emd_size, emd_size)

    def forward(self, x, mask = None):
        keys = self.keys(x)
        keys = rearrange(keys, "b n (h d) -> b h n d", h = self.no_heads)
        queries = self.queries(x)
        queries = rearrange(queries, "b n (h d) -> b h n d", h = self.no_heads)
        values = self.values(x)
        values = rearrange(values, "b n (h d) -> b h n d", h = self.no_heads)

        # dot product of queries and keys
        # b = batch_size, h = no. heads, q = queries length, k = keys length
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            # fill the min value where mask is False
            energy.mask_fill(~mask, fill_value)
        # dividing the result by sqrt(N) N = emd_size
        scaling = self.emd_size ** 0.5
        attention = F.softmax(energy, dim=1)/scaling
        attention = self.dropout(attention)
    
        # dot product of result and values
        out = torch.einsum('bhnd, bhdk -> bhnk', attention, values)
        out = rearrange(out, "b h n k -> b n (h k)")
        out = self.projection(out)
        return out

class Residual(nn.Module):
    def __init__(self, fnc):
        super().__init__()
        self.fnc = fnc
    
    def forward(self, x, **kwargs):
        temp = x.clone()
        x = self.fnc(temp, **kwargs)
        x += temp
        return x



class MLP(nn.Module):
    def __init__(self, emd_size, expansion = 4, dropout = 0.):
        super().__init__()
        self.fc1 = nn.Linear(emd_size, expansion * emd_size)
        self.fc2 = nn.Linear(expansion*emd_size, emd_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Transformer(nn.Module):
    def __init__(self, n_transform = 12, emd_size = 768, num_head = 8, dropout = 0., expansion = 4):
        super().__init__()
        self.net = nn.ModuleList([])
        for _ in range(n_transform):
            self.net.append(nn.ModuleList([
                Residual(Normalize(emd_size, MultiHeadAttention(emd_size, num_head, dropout))),
                Residual(Normalize(emd_size, MLP(emd_size, expansion, dropout)))
            ]))
    def forward(self, x):
        for attention, fc in self.net:
            x = attention(x)
            x = fc(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, emd_size = 768, n_classes = 100):
        super().__init__()
        self.net = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emd_size, n_classes)
        )
    def forward(self, x):
        return self.net(x)

class VIT(nn.Module):
    def __init__(self, in_channels = 3, patch_size = 16, emd_size = 768, img_size = 224, n_classes = 100, n_transformer = 12, num_heads = 12, dropout = 0., expansion = 4):
        super().__init__()

        self.net = nn.Sequential(
            Patches(in_channels, emd_size, patch_size, img_size),
            Transformer(n_transformer, emd_size, num_heads, dropout, expansion),
            ClassificationHead(emd_size, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    # x = torch.randn(16, 3, 224, 224)
    summary(VIT(), (3, 224, 224), device='cuda')