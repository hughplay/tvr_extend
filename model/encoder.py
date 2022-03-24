from collections import OrderedDict
from math import inf
from typing import Mapping

import torch
from torch.functional import norm
import torch.nn.functional as F
from torch import nn

from .block import ConvBlocks, ResNetBlock, VGGBlock, resnet_fpn_backbone
from torchvision.ops import RoIAlign, RoIPool
from x_transformers import Encoder


class BaseEncoder(nn.Module):
    def __init__(self, encoder, mode):
        super().__init__()
        self.encoder = encoder
        self.mode = mode

    def get_output_shape(self, height, width):
        batch, channel = 1, 3
        tensor = torch.zeros(batch, channel, height, width).to(
            next(self.encoder.parameters()).device
        )
        _, c_out, h_out, w_out = tuple(self.forward(tensor, tensor).shape)
        return c_out, h_out, w_out

    def forward(self, init, fin):
        if self.mode == "concat":
            out = torch.cat([init, fin], dim=1)
        elif self.mode == "subtract":
            out = init - fin
        elif self.mode == "subcat":
            out = torch.cat([init, init - fin], dim=1)
        else:
            raise NotImplementedError
        out = self.encoder(out)
        return out


class CNNEncoder(BaseEncoder):
    def __init__(
        self, mode, c_kernels=[16, 32, 64, 64], s_kernels=[5, 3, 3, 3]
    ):
        c_in = 3 if mode == "subtract" else 6
        encoder = ConvBlocks(
            c_in=c_in, c_kernels=c_kernels, s_kernels=s_kernels
        )
        super().__init__(encoder, mode)


class ResNetEncoder(BaseEncoder):
    def __init__(self, mode, arch="resnet18"):
        c_in = 3 if mode == "subtract" else 6
        encoder = ResNetBlock(arch, c_in=c_in)
        super().__init__(encoder, mode)


class DUDAEncoder(nn.Module):
    def __init__(self, arch):
        super().__init__()

        self.encoder = ResNetBlock(arch)
        c_out = self.encoder.c_out

        self.attention = nn.Sequential(
            ConvBlocks(
                c_out * 2,
                c_kernels=[c_out, c_out],
                s_kernels=[3, 3],
                strides=1,
                norm=None,
                act="relu",
            ),
            nn.Sigmoid(),
        )

    def get_output_shape(self, height, width):
        batch, channel = 1, 3
        tensor = torch.zeros(batch, channel, height, width).to(
            next(self.encoder.parameters()).device
        )
        _, c_out, h_out, w_out = tuple(self.forward(tensor, tensor).shape)
        return c_out, h_out, w_out

    def forward(self, init, fin):

        x_init = self.encoder(init)
        x_fin = self.encoder(fin)
        x_diff = x_fin - x_init

        a_init = self.attention(torch.cat([x_init, x_diff], dim=1))
        a_fin = self.attention(torch.cat([x_fin, x_diff], dim=1))

        l_init = x_init * a_init
        l_fin = x_fin * a_fin
        l_diff = l_fin - l_init

        res = torch.cat([l_init, l_fin, l_diff], dim=1)

        return res


class BCNNEncoder(nn.Module):
    def __init__(self, arch="vgg11_bn"):
        super().__init__()
        self.encoder = VGGBlock(arch)

    def get_output_shape(self, height, width):
        batch, channel = 1, 3
        tensor = torch.zeros(batch, channel, height, width).to(
            next(self.encoder.parameters()).device
        )
        _, c_out = tuple(self.forward(tensor, tensor).shape)
        return c_out

    def forward(self, init, fin):
        init = self.encoder(init)
        N, C, H, W = init.shape
        init = init.view(N, C, H * W)
        L = init.shape[-1]

        fin = self.encoder(fin)
        fin = fin.view(N, C, H * W).transpose(1, 2)

        res = torch.bmm(init, fin) / L

        res = res.view(N, -1)
        res = torch.sqrt(res + 1e-5)
        res = F.normalize(res)

        return res


class DetectorEncoder(nn.Module):
    def __init__(
        self,
        mode="subtract",
        arch="resnet18",
        in_features=["0", "1", "2", "3"],
        roi_type="pool",
        roi_output_size=(2, 2),
        roi_sample_ratio=-1,
        hidden_size=256,
        n_attn_layer=2,
        n_head=8,
    ):
        super().__init__()
        self.MAX_BOX = 11
        self.mode = mode
        c_in = 6 if mode == "concat" else 3
        self.backbone = resnet_fpn_backbone(
            arch, c_in=c_in, norm_layer=nn.BatchNorm2d
        )
        self.in_features = in_features
        if roi_type == "align":
            self.roi_pool = {
                level: RoIAlign(
                    output_size=roi_output_size,
                    spatial_scale=4 + 2 * int(level),
                    sampling_ratio=roi_sample_ratio,
                    aligned=True,
                )
                for level in self.in_features
            }
        elif roi_type == "pool":
            self.roi_pool = {
                level: RoIPool(
                    output_size=roi_output_size,
                    spatial_scale=4 + 2 * int(level),
                )
                for level in self.in_features
            }
        else:
            NotImplementedError

        # TODO: use FPN paper method
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(
                self.backbone.out_channels * len(self.in_features),
                hidden_size,
                kernel_size=roi_output_size,
            ),
            nn.Flatten(start_dim=1),
            nn.LayerNorm(hidden_size),
        )

        self.pos_encode = nn.Sequential(
            nn.Linear(7, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.init_type = nn.Parameter(torch.zeros(1, hidden_size))
        self.fin_type = nn.Parameter(torch.zeros(1, hidden_size))
        self.cls = nn.Parameter(torch.zeros(1, hidden_size))
        self.sep = nn.Parameter(torch.zeros(1, hidden_size))
        # self.cls = nn.Parameter(torch.zeros(1, hidden_size * 2))
        nn.init.normal_(self.init_type)
        nn.init.normal_(self.fin_type)
        nn.init.normal_(self.cls)
        nn.init.normal_(self.sep)

        self.transformers = Encoder(
            dim=hidden_size, depth=n_attn_layer, heads=n_head
        )
        # self.transformers = Encoder(dim=hidden_size * 2, depth=2, heads=8)

    def get_output_shape(self, height, width):
        batch, channel = 1, 3
        device = next(self.backbone.parameters()).device
        tensor = torch.zeros(batch, channel, height, width).to(device)
        n_box, box_dim = 1, 5
        box = torch.zeros(n_box, box_dim).to(device)
        n = (n_box,)
        _, c_flat = tuple(
            self.forward(tensor, tensor, box, box, n, n)
            .flatten(start_dim=1)
            .shape
        )
        return c_flat

    def _split(self, feature, batch_size):
        if isinstance(feature, torch.Tensor):
            left, right = feature[:batch_size], feature[batch_size:]
        elif isinstance(feature, Mapping):
            left, right = OrderedDict(), OrderedDict()
            for key, val in feature.items():
                left[key], right[key] = val[:batch_size], val[batch_size:]
        return left, right

    def _concat_objs(self, init_roi, fin_roi, n_init_box, n_fin_box):
        """concat all rois and pad them to the same length"""
        _, C = init_roi.shape
        init_roi_list = torch.split_with_sizes(init_roi, n_init_box, dim=0)
        fin_roi_list = torch.split_with_sizes(fin_roi, n_fin_box, dim=0)
        roi = torch.stack(
            [
                torch.cat(
                    [
                        self.cls,
                        r_init,
                        self.sep,
                        r_fin,
                        torch.zeros(
                            self.MAX_BOX * 2 - r_init.size(0) - r_fin.size(0),
                            C,
                            device=init_roi.device,
                        ),
                    ],
                    dim=0,
                )
                for r_init, r_fin in zip(init_roi_list, fin_roi_list)
            ]
        )

        n_token = torch.tensor(n_init_box) + torch.tensor(n_fin_box) + 2
        mask = (
            torch.tensor(
                [[1] * n + [0] * (self.MAX_BOX * 2 + 2 - n) for n in n_token]
            )
            .bool()
            .to(roi.device)
        )
        return roi, mask

    def _encode_pos(self, boxes, h, w):
        boxes = boxes[:, 1:] / torch.tensor([w, h, w, h]).to(boxes)
        boxes = boxes.clamp(0, 1)
        width = boxes[:, 2:3] - boxes[:, 0:1]
        height = boxes[:, 3:4] - boxes[:, 1:2]
        area = width * height
        pos = torch.cat([boxes, width, height, area], dim=1)
        return self.pos_encode(pos)

    def forward(self, init, fin, init_boxes, fin_boxes, n_init, n_fin):
        _, _, h, w = init.shape
        init_pos = self._encode_pos(init_boxes, h, w)
        fin_pos = self._encode_pos(fin_boxes, h, w)

        if self.mode in ["concat", "subtract"]:
            if self.mode == "concat":
                feat = torch.cat([init, fin], dim=1)
            elif self.mode == "subtract":
                feat = init - fin
            feat = self.backbone(feat)
            out = self._early_fusion_simple(
                feat, init_boxes, fin_boxes, n_init, n_fin, init_pos, fin_pos
            )
        return out

    def _early_fusion_simple(
        self, feat, init_boxes, fin_boxes, n_init, n_fin, init_pos, fin_pos
    ):
        out = []

        init, fin = [], []
        for key in self.in_features:
            init_roi = self.roi_pool[key](feat[key], init_boxes)
            init.append(init_roi)
            fin_roi = self.roi_pool[key](feat[key], fin_boxes)
            fin.append(fin_roi)
        # N (B * nbox?) * C
        init = self.scale_fusion(torch.cat(init, dim=1))
        # N (B * nbox?) * C
        fin = self.scale_fusion(torch.cat(fin, dim=1))

        # init = init + init_pos
        # fin = fin + fin_pos

        init = init + init_pos + self.init_type.expand_as(init)
        fin = fin + fin_pos + self.fin_type.expand_as(fin)

        # init = torch.cat([init, init_pos], dim=1)
        # fin = torch.cat([fin, fin_pos], dim=1)

        out = self._agg_objs(init, fin, n_init, n_fin)
        return out

    def _agg_objs(self, init_roi, fin_roi, n_init_box, n_fin_box):
        """concat all rois and pad them to the same length"""
        _, C = init_roi.shape
        init_roi_list = torch.split_with_sizes(init_roi, n_init_box, dim=0)
        fin_roi_list = torch.split_with_sizes(fin_roi, n_fin_box, dim=0)
        out = torch.stack(
            [
                torch.cat(
                    [
                        r_init.max(dim=0)[0],
                        r_fin.max(dim=0)[0],
                    ],
                    dim=0,
                )
                for r_init, r_fin in zip(init_roi_list, fin_roi_list)
            ]
        )
        return out

    def forward_transformer(self, init, fin, init_boxes, fin_boxes, n_init, n_fin):
        _, _, h, w = init.shape
        init_pos = self._encode_pos(init_boxes, h, w)
        fin_pos = self._encode_pos(fin_boxes, h, w)

        if self.mode in ["concat", "subtract"]:
            if self.mode == "concat":
                feat = torch.cat([init, fin], dim=1)
            elif self.mode == "subtract":
                feat = init - fin
            feat = self.backbone(feat)
            out = self._early_fusion(
                feat, init_boxes, fin_boxes, n_init, n_fin, init_pos, fin_pos
            )
        elif self.mode == "latter_fusion":
            feat_init = self.backbone(init)
            feat_fin = self.backbone(fin)
            out = self._latter_fusion(
                feat_init,
                feat_fin,
                init_boxes,
                fin_boxes,
                n_init,
                n_fin,
                init_pos,
                fin_pos,
            )
        else:
            raise NotImplementedError
        return out

    def _early_fusion(
        self, feat, init_boxes, fin_boxes, n_init, n_fin, init_pos, fin_pos
    ):
        out = []

        init, fin = [], []
        for key in self.in_features:
            init_roi = self.roi_pool[key](feat[key], init_boxes)
            init.append(init_roi)
            fin_roi = self.roi_pool[key](feat[key], fin_boxes)
            fin.append(fin_roi)
        # N (B * nbox?) * C
        init = self.scale_fusion(torch.cat(init, dim=1))
        # N (B * nbox?) * C
        fin = self.scale_fusion(torch.cat(fin, dim=1))

        # init = init + init_pos
        # fin = fin + fin_pos

        init = init + init_pos + self.init_type.expand_as(init)
        fin = fin + fin_pos + self.fin_type.expand_as(fin)

        # init = torch.cat([init, init_pos], dim=1)
        # fin = torch.cat([fin, fin_pos], dim=1)

        objs, mask = self._concat_objs(init, fin, n_init, n_fin)
        objs = self.layer_norm(objs)
        out = self.transformers(objs, mask=mask)

        # cls
        out = out[:, 0]

        # sep
        # out = out[:, torch.tensor(n_init) + 1]

        # max
        # out[~mask] = -inf
        # out = torch.max(out, dim=1)[0]

        # mean
        # out[~mask] = 0
        # out = torch.sum(out, dim=1) / mask.sum(dim=1, keepdim=True)

        return out

    # def _latter_fusion(
    #         self, init, fin, init_boxes, fin_boxes, n_init, n_fin,
    #         init_pos, fin_pos):
    #     out = [
    #         torch.cat([init['pool'], fin['pool']], dim=1).flatten(start_dim=1)]

    #     for key in ['0', '1', '2', '3']:
    #         roi = self.roi_pool[key](
    #             init[key], init_boxes).flatten(start_dim=1)
    #         roi = self._pad_roi(roi, n_init)
    #         out.append(roi.flatten(start_dim=1))
    #         roi = self.roi_pool[key](fin[key], fin_boxes).flatten(start_dim=1)
    #         roi = self._pad_roi(roi, n_fin)
    #         out.append(roi.flatten(start_dim=1))

    #     out = torch.cat(out, dim=1)
    #     return out

    # def _latter_fusion(
    #         self, feat_init, feat_fin, init_boxes, fin_boxes, n_init, n_fin,
    #         init_pos, fin_pos):
    #     # out = [feat['pool'].flatten(start_dim=1)]
    #     out = []

    #     init, fin = [], []
    #     for key in self.in_features:
    #         # N (B * nbox?) * C * H * W
    #         init_roi = self.roi_pool[key](feat_init[key], init_boxes)
    #         init.append(init_roi)
    #         # init_roi = self._fuse_roi(
    #         #     init_roi, init_pos, self.init_type, n_init, self.init_empty)
    #         fin_roi = self.roi_pool[key](feat_fin[key], fin_boxes)
    #         fin.append(fin_roi)
    #         # fin_roi = self._fuse_roi(
    #         #     fin_roi, fin_pos, self.fin_type, n_fin, self.fin_empty)
    #         # roi = torch.cat([init_roi, fin_roi], dim=1)
    #         # a_roi = self.fuse_attention(roi)
    #         # roi = self.fuse_norm(self.fuse_encode(roi * a_roi) + roi)
    #         # out.append(roi)
    #     # N (B * nbox?) * C
    #     init = self.scale_fusion(torch.cat(init, dim=1)).squeeze()
    #     # N (B * nbox?) * C
    #     fin = self.scale_fusion(torch.cat(fin, dim=1)).squeeze()

    #     # init = init + init_pos
    #     # fin = fin + fin_pos

    #     init = init + init_pos + self.init_type.expand_as(init)
    #     fin = fin + fin_pos + self.fin_type.expand_as(fin)

    #     objs, mask = self._pad_double_roi(init, fin, n_init, n_fin)
    #     objs, mask = self._append_cls(objs, mask)
    #     out = self.transformers(objs, mask=mask)
    #     cls_out = out[:, 0]

    #     # out = torch.cat(out, dim=1)
    #     return cls_out
