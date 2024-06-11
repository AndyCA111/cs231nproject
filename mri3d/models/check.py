import torch
# from resnet import Downsample3D
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
# Check available memory
class pseudoConv3d(nn.Conv2d):
    def forward(self, x):
        b,c,f,h,w = x.shape
        x = rearrange(x, "b c f h w -> (b w) c f h")
        x = super().forward(x)
        x = rearrange(x, "(b w) c f h-> b c f h w", w=w)
        return x
class tempConv1d(nn.Conv1d):
    def forward(self, x):
        b, c, *_, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b h w) c f')
        x = super().forward(x)
        x = rearrange(x, '(b h w) c f -> b c f h w', h = h, w = w)
        return x

class Downsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv1d=True, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        self.use_conv1d = use_conv1d

        if use_conv:
            conv = pseudoConv3d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
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
            self.convtemp = tempConv1d(self.out_channels, self.out_channels, 3, stride=1, padding=padding)
    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            #TODO :
            raise NotImplementedError

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        if self.use_conv1d:
            hidden_states = self.convtemp(hidden_states)
        return hidden_states
class Upsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            conv = pseudoConv3d(self.channels, self.out_channels, 3, padding=1)

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

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
            hidden_states = F.interpolate(hidden_states, scale_factor=[1.0, 2.0, 2.0], mode="nearest")
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

        return hidden_states

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pse3d = pseudoConv3d(in_channels=3,out_channels=3,kernel_size=3,stride=2,padding=1)
downblock = Downsample3D(channels=3, use_conv=True)
upblock = Upsample3D(channels=3, use_conv=True)
a = torch.rand(1,3,30,30,30)
upblock = upblock.to(device)
a = a.to(device)
print(upblock(a).shape)
