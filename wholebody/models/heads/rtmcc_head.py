import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple, Union

from wholebody.core.registry import HEADS
from wholebody.models.backbones.cspnext import ConvModule

def rope(x, dim):
    shape = x.shape
    if isinstance(dim, int):
        dim = [dim]

    spatial_shape = [shape[i] for i in dim]
    total_len = 1
    for i in spatial_shape:
        total_len *= i

    position = torch.reshape(
        torch.arange(total_len, dtype=torch.int, device=x.device),
        spatial_shape)

    for i in range(dim[-1] + 1, len(shape) - 1, 1):
        position = torch.unsqueeze(position, dim=-1)

    half_size = shape[-1] // 2
    freq_seq = -torch.arange(
        half_size, dtype=torch.float, device=x.device) / float(half_size)
    inv_freq = 10000**-freq_seq

    sinusoid = position[..., None] * inv_freq[None, None, :]

    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    x1, x2 = torch.chunk(x, 2, dim=-1)

    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        from wholebody.models.losses.kl_discret_loss import KLDiscretLoss
        self.loss_module = KLDiscretLoss(use_target_weight=True, beta=10.0, label_softmax=True)
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor

class Scale(nn.Module):
    def __init__(self, dim, init_value=1., trainable=True):
        super().__init__()
        from wholebody.models.losses.kl_discret_loss import KLDiscretLoss
        self.loss_module = KLDiscretLoss(use_target_weight=True, beta=10.0, label_softmax=True)
        self.scale = nn.Parameter(
            init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        from wholebody.models.losses.kl_discret_loss import KLDiscretLoss
        self.loss_module = KLDiscretLoss(use_target_weight=True, beta=10.0, label_softmax=True)
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        norm = norm * self.scale
        return x / norm.clamp(min=self.eps) * self.g

class RTMCCBlock(nn.Module):
    def __init__(self,
                 num_token,
                 in_token_dims,
                 out_token_dims,
                 expansion_factor=2,
                 s=128,
                 eps=1e-5,
                 dropout_rate=0.,
                 drop_path=0.,
                 attn_type='self-attn',
                 act_fn='SiLU',
                 bias=False,
                 use_rel_bias=True,
                 pos_enc=False):

        super().__init__()
        from wholebody.models.losses.kl_discret_loss import KLDiscretLoss
        self.loss_module = KLDiscretLoss(use_target_weight=True, beta=10.0, label_softmax=True)
        self.s = s
        self.num_token = num_token
        self.use_rel_bias = use_rel_bias
        self.attn_type = attn_type
        self.pos_enc = pos_enc
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.e = int(in_token_dims * expansion_factor)
        if use_rel_bias:
            if attn_type == 'self-attn':
                self.w = nn.Parameter(
                    torch.rand([2 * num_token - 1], dtype=torch.float))
            else:
                self.a = nn.Parameter(torch.rand([1, s], dtype=torch.float))
                self.b = nn.Parameter(torch.rand([1, s], dtype=torch.float))
        self.o = nn.Linear(self.e, out_token_dims, bias=bias)

        if attn_type == 'self-attn':
            self.uv = nn.Linear(in_token_dims, 2 * self.e + self.s, bias=bias)
            self.gamma = nn.Parameter(torch.rand((2, self.s)))
            self.beta = nn.Parameter(torch.rand((2, self.s)))
        else:
            self.uv = nn.Linear(in_token_dims, self.e + self.s, bias=bias)
            self.k_fc = nn.Linear(in_token_dims, self.s, bias=bias)
            self.v_fc = nn.Linear(in_token_dims, self.e, bias=bias)

        self.ln = ScaleNorm(in_token_dims, eps=eps)
        
        self.act_fn = nn.SiLU(True) if act_fn == 'SiLU' else nn.ReLU(True)

        if in_token_dims == out_token_dims:
            self.shortcut = True
            self.res_scale = Scale(in_token_dims)
        else:
            self.shortcut = False

        self.sqrt_s = math.sqrt(s)
        self.dropout_rate = dropout_rate
        if dropout_rate > 0.:
            self.dropout = nn.Dropout(dropout_rate)

    def rel_pos_bias(self, seq_len, k_len=None):
        if self.attn_type == 'self-attn':
            t = F.pad(self.w[:2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        else:
            a = rope(self.a.repeat(seq_len, 1), dim=0)
            b = rope(self.b.repeat(k_len, 1), dim=0)
            t = torch.bmm(a, b.permute(0, 2, 1))
        return t

    def _forward(self, inputs):
        if self.attn_type == 'self-attn':
            x = inputs
        else:
            x, k, v = inputs

        x = self.ln(x)
        uv = self.uv(x)
        uv = self.act_fn(uv)

        if self.attn_type == 'self-attn':
            u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
            base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta
            if self.pos_enc:
                base = rope(base, dim=1)
            q, k = torch.unbind(base, dim=2)
        else:
            u, q = torch.split(uv, [self.e, self.s], dim=2)
            k = self.k_fc(k)
            v = self.v_fc(v)
            if self.pos_enc:
                q = rope(q, 1)
                k = rope(k, 1)

        qk = torch.bmm(q, k.permute(0, 2, 1))

        if self.use_rel_bias:
            if self.attn_type == 'self-attn':
                bias = self.rel_pos_bias(q.size(1))
            else:
                bias = self.rel_pos_bias(q.size(1), k.size(1))
            qk += bias[:, :q.size(1), :k.size(1)]
            
        kernel = torch.square(F.relu(qk / self.sqrt_s))

        if self.dropout_rate > 0.:
            kernel = self.dropout(kernel)
            
        x = u * torch.bmm(kernel, v)
        x = self.o(x)
        return x

    def forward(self, x):
        if self.shortcut:
            res_shortcut = x[0] if self.attn_type == 'cross-attn' else x
            main_branch = self.drop_path(self._forward(x))
            return self.res_scale(res_shortcut) + main_branch
        else:
            return self.drop_path(self._forward(x))

@HEADS.register("RTMCCHead")
class RTMCCHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        final_layer_kernel_size: int = 1,
        gau_cfg: dict = dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        codec: dict = None,
    ):
        super().__init__()
        from wholebody.models.losses.kl_discret_loss import KLDiscretLoss
        self.loss_module = KLDiscretLoss(use_target_weight=True, beta=10.0, label_softmax=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio
        
        if codec is not None:
            from wholebody.core.registry import CODECS
            self.codec = CODECS.build(codec)
        else:
            self.codec = None

        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]
        self.final_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2,
            act=nn.ReLU(inplace=True)
        )

        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False)
        )

        W = int(self.input_size[1] * self.simcc_split_ratio) # input_size is [H, W], so W is input_size[1]
        H = int(self.input_size[0] * self.simcc_split_ratio) # H is input_size[0]

        self.gau = RTMCCBlock(
            self.out_channels,
            gau_cfg['hidden_dims'],
            gau_cfg['hidden_dims'],
            s=gau_cfg['s'],
            expansion_factor=gau_cfg['expansion_factor'],
            dropout_rate=gau_cfg['dropout_rate'],
            drop_path=gau_cfg['drop_path'],
            attn_type='self-attn',
            act_fn=gau_cfg['act_fn'],
            use_rel_bias=gau_cfg['use_rel_bias'],
            pos_enc=gau_cfg['pos_enc'])

        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)

    def forward(self, feats):
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
            
        feats = self.final_layer(feats)
        feats = torch.flatten(feats, 2)
        feats = self.mlp(feats)
        feats = self.gau(feats)
        
        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)
        
        return pred_x, pred_y

    def loss(self, feats, batch_data_samples):
        pred_x, pred_y = self.forward(feats)
        
        # SimCC codec uses keypoints directly to create ground truth SimCC
        # Our pipeline should theoretically generate gt_x and gt_y, but currently
        # it just produces heatmaps or keypoints. We will extract keypoints and target_weight.
        device = pred_x.device
        keypoints = torch.stack([s.gt_instances.keypoints for s in batch_data_samples]).to(device)
        weights = torch.stack([s.gt_instances.keypoint_weights for s in batch_data_samples]).to(device)
        
        # The codec should have a method to generate target simcc from keypoints
        gt_x, gt_y, gt_weight = self.codec.encode(keypoints.cpu().numpy(), weights.cpu().numpy())
        gt_x = torch.from_numpy(gt_x).to(device)
        gt_y = torch.from_numpy(gt_y).to(device)
        gt_weight = torch.from_numpy(gt_weight).to(device)
        
        loss_val = self.loss_module((pred_x, pred_y), (gt_x, gt_y), gt_weight)
        
        return {"loss_kpt": loss_val}
