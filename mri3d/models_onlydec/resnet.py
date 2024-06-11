# start bulid 3d vae block based on tune a video 
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# class pseudoConv3d(nn.Module):
#     def __init__(
#         self,
#         dim,
#         dim_out = None,
#         kernel_size = 3,
#         padding = 1,
#         *,
#         temporal_kernel_size = None,
#         **kwargs
#     ):
#         super().__init__()
#         dim_out = default(dim_out, dim)
#         temporal_kernel_size = default(temporal_kernel_size, kernel_size)

#         self.spatial_conv = nn.Conv2d(dim, dim_out, kernel_size = kernel_size, padding = padding // 2)
#         self.temporal_conv = nn.Conv1d(dim_out, dim_out, kernel_size = temporal_kernel_size, padding = padding// 2) if kernel_size > 1 else None

#         if exists(self.temporal_conv):
#             nn.init.dirac_(self.temporal_conv.weight.data) # initialized to be identity
#             nn.init.zeros_(self.temporal_conv.bias.data)

#     def forward(
#         self,
#         x,
#         enable_time = True
#     ):
#         b, c, *_, h, w = x.shape

#         is_video = x.ndim == 5
#         enable_time &= is_video

#         if is_video:
#             x = rearrange(x, 'b c f h w -> (b f) c h w')

#         x = self.spatial_conv(x)

#         if is_video:
#             x = rearrange(x, '(b f) c h w -> b c f h w', b = b)

#         if not enable_time or not exists(self.temporal_conv):
#             return x

#         x = rearrange(x, 'b c f h w -> (b h w) c f')

#         x = self.temporal_conv(x)

#         x = rearrange(x, '(b h w) c f -> b c f h w', h = h, w = w)

#         return x

class pseudoConv3d(nn.Conv2d):
    def __init__(self, *args, dim=0,**kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
    def forward(self, x):
        b,c,f,h,w = x.shape
        if self.dim == 0:
            x = rearrange(x, "b c f h w -> (b f) c h w")
            x = super().forward(x)
            x = rearrange(x, "(b f) c h w -> b c f h w", f=f)
        if self.dim == 1:
            x = rearrange(x, "b c f h w -> (b h) c f w")
            x = super().forward(x)
            x = rearrange(x, "(b h) c f w-> b c f h w", h=h)
        if self.dim == 2:
            x = rearrange(x, "b c f h w -> (b w) c f h")
            x = super().forward(x)
            x = rearrange(x, "(b w) c f h-> b c f h w", w=w)
        return x
class tempConv1d(nn.Conv1d):
    def __init__(self, *args, dim=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
    def forward(self, x):
        b, c, f, h, w = x.shape
        if self.dim ==0:
            x = rearrange(x, 'b c f h w -> (b h w) c f')
            x = super().forward(x)
            x = rearrange(x, '(b h w) c f -> b c f h w', h = h, w = w)
        if self.dim ==1:
            x = rearrange(x, 'b c f h w -> (b f w) c h')
            x = super().forward(x)
            x = rearrange(x, '(b f w) c h -> b c f h w', f = f, w = w)
        if self.dim ==2:
            x = rearrange(x, 'b c f h w -> (b f h) c w')
            x = super().forward(x)
            x = rearrange(x, '(b f h) c w -> b c f h w', h = h, f = f)
        return x


class Upsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv", dim=0, use_conv1d = False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.use_conv1d = use_conv1d
        self.dim = dim

        conv = None
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            conv = pseudoConv3d(self.channels, self.out_channels, 3, padding=1, dim=self.dim)

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv
        if self.use_conv1d:
            self.convtemp = tempConv1d(self.out_channels, self.out_channels, 3, stride=1, padding=1, dim=self.dim)
    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            raise NotImplementedError

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            if self.dim == 0:
                hidden_states = F.interpolate(hidden_states, scale_factor=[1.0, 2.0, 2.0], mode="nearest")
            elif self.dim == 1:
                hidden_states = F.interpolate(hidden_states, scale_factor=[2.0, 1.0, 2.0], mode="nearest")
            elif self.dim == 2:
                hidden_states = F.interpolate(hidden_states, scale_factor=[2.0, 2.0, 1.0], mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)
        if self.use_conv1d:
            hidden_states = self.convtemp(hidden_states)
        return hidden_states


class Downsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv1d=True, use_conv2d=True, out_channels=None, padding=1, name="conv", dim=0,):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        self.use_conv1d = use_conv1d
        self.dim = dim
        
        if use_conv:
            if use_conv2d:
                conv = torch.nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
            else:
                conv = pseudoConv3d(self.channels, self.out_channels, 3, stride=stride, padding=padding, dim=self.dim)
        else:
            raise NotImplementedError
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv
        # TODO: initialize stride to 1 
        if self.use_conv1d:
            self.convtemp = tempConv1d(self.out_channels, self.out_channels, 3, stride=1, padding=padding, dim=self.dim)
    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        if self.use_conv1d:
            hidden_states = self.convtemp(hidden_states)
        return hidden_states


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        output_scale_factor=1.0,
        use_in_shortcut=None,
        use_conv1d =True,
        use_conv2d =False,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        if use_conv2d:
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = pseudoConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        #TODO: Add layer
        if use_conv1d:
            self.conv11 = tempConv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)

        if use_conv2d:
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = pseudoConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if use_conv1d:
            self.conv21 = tempConv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            if use_conv2d:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            else:    
                self.conv_shortcut = pseudoConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.use_conv1d = use_conv1d
    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)
        if self.use_conv1d:
            hidden_states = self.conv11(hidden_states)
        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.use_conv1d:
            hidden_states = self.conv21(hidden_states)
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))