from typing import Sequence, Tuple
import torch
import torch.nn as nn

from wholebody.core.registry import NECKS
from wholebody.models.backbones.cspnext import ConvModule, DepthwiseSeparableConvModule, CSPLayer

@NECKS.register("CSPNeXtPAFPN")
class CSPNeXtPAFPN(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = None,
        out_indices: Sequence[int] = (0, 1, 2),
        num_csp_blocks: int = 3,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_indices = out_indices
        
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(in_channels[idx], in_channels[idx - 1], 1)
            )
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    expand_ratio=expand_ratio
                )
            )

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(in_channels[idx], in_channels[idx], 3, stride=2, padding=1)
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    expand_ratio=expand_ratio
                )
            )

        if self.out_channels is not None:
            self.out_convs = nn.ModuleList()
            for i in range(len(in_channels)):
                self.out_convs.append(
                    conv(in_channels[i], out_channels, 3, padding=1)
                )

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_high = self.reduce_layers[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high

            upsample_feat = self.upsample(feat_high)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1)
            )
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_high], 1)
            )
            outs.append(out)

        if self.out_channels is not None:
            # out convs
            for idx, conv_layer in enumerate(self.out_convs):
                outs[idx] = conv_layer(outs[idx])

        return tuple([outs[i] for i in self.out_indices])
