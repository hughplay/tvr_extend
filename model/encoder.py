from collections import OrderedDict
from typing import Mapping

import torch
from torch.functional import norm
import torch.nn.functional as F
from torch import nn

from .block import ConvBlocks, ResNetBlock, VGGBlock, resnet_fpn_backbone
from torchvision.ops import RoIAlign, RoIPool


class BaseEncoder(nn.Module):

    def __init__(self, encoder, mode):
        super().__init__()
        self.encoder = encoder
        self.mode = mode

    def get_output_shape(self, height, width):
        batch, channel = 1, 3
        tensor = torch.zeros(batch, channel, height, width).to(
            next(self.encoder.parameters()).device)
        _, c_out, h_out, w_out = tuple(self.forward(tensor, tensor).shape)
        return c_out, h_out, w_out

    def forward(self, init, fin):
        if self.mode == 'concat':
            out = torch.cat([init, fin], dim=1)
        elif self.mode == 'subtract':
            out = init - fin
        else:
            raise NotImplementedError
        out = self.encoder(out)
        return out


class CNNEncoder(BaseEncoder):

    def __init__(
            self, mode, c_kernels=[16, 32, 64, 64], s_kernels=[5, 3, 3, 3]):
        c_in = 3 if mode == 'subtract' else 6
        encoder = ConvBlocks(
            c_in=c_in, c_kernels=c_kernels, s_kernels=s_kernels)
        super().__init__(encoder, mode)


class ResNetEncoder(BaseEncoder):
    def __init__(self, mode, arch="resnet18"):
        c_in = 3 if mode == 'subtract' else 6
        encoder = ResNetBlock(arch, c_in=c_in)
        super().__init__(encoder, mode)


class DUDAEncoder(nn.Module):

    def __init__(self, arch):
        super().__init__()

        self.encoder = ResNetBlock(arch)
        c_out = self.encoder.c_out

        self.attention = nn.Sequential(
            ConvBlocks(
                c_out * 2, c_kernels=[c_out, c_out], s_kernels=[3, 3],
                strides=1, norm=None, act='relu'),
            nn.Sigmoid()
        )

    def get_output_shape(self, height, width):
        batch, channel = 1, 3
        tensor = torch.zeros(batch, channel, height, width).to(
            next(self.encoder.parameters()).device)
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

    def __init__(self, arch='vgg11_bn'):
        super().__init__()
        self.encoder = VGGBlock(arch)

    def get_output_shape(self, height, width):
        batch, channel = 1, 3
        tensor = torch.zeros(batch, channel, height, width).to(
            next(self.encoder.parameters()).device)
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
            self, mode, arch="resnet18", roi_type='align',
            roi_output_size=(7,7), roi_sample_ratio=-1, hidden_size=256):
        super().__init__()
        self.MAX_BOX=11
        self.mode = mode
        c_in = 6 if mode == 'concat' else 3
        self.backbone = resnet_fpn_backbone(
            arch, c_in=c_in, norm_layer=nn.BatchNorm2d)
        if roi_type == 'align':
            self.roi_pool = {
                '0': RoIAlign(
                    output_size=roi_output_size, spatial_scale=4,
                    sampling_ratio=roi_sample_ratio, aligned=True),
                '1': RoIAlign(
                    output_size=roi_output_size, spatial_scale=8,
                    sampling_ratio=roi_sample_ratio, aligned=True),
                '2': RoIAlign(
                    output_size=roi_output_size, spatial_scale=16,
                    sampling_ratio=roi_sample_ratio, aligned=True),
                '3': RoIAlign(
                    output_size=roi_output_size, spatial_scale=32,
                    sampling_ratio=roi_sample_ratio, aligned=True),
            }
        elif roi_type == 'pool':
            self.roi_pool = {
                '0': RoIPool(output_size=roi_output_size, spatial_scale=4),
                '1': RoIPool(output_size=roi_output_size, spatial_scale=8),
                '2': RoIPool(output_size=roi_output_size, spatial_scale=16),
                '3': RoIPool(output_size=roi_output_size, spatial_scale=32),
            }
        else:
            NotImplementedError

        roi_dim = roi_output_size[0] * roi_output_size[1] \
            * self.backbone.out_channels
        self.roi_encode = nn.Sequential(
            nn.Linear(roi_dim, hidden_size),
            nn.LayerNorm(hidden_size, eps=1e-12),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

        self.pos_encode = nn.Sequential(
            nn.Linear(7, hidden_size),
            nn.LayerNorm(hidden_size, eps=1e-12),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

        # TODO transformer
        num_max_obj = 11
        self.fuse_attention = nn.Sequential(
            nn.Linear(hidden_size * 11 * 2, hidden_size * 11 * 2),
            nn.Softmax()
        )
        self.fuse_encode = nn.Linear(
            hidden_size * 11 * 2, hidden_size * 11 * 2)
        self.fuse_norm = nn.LayerNorm(hidden_size * 11 * 2)

        self.init_type = nn.Parameter(torch.zeros(1, hidden_size))
        self.fin_type = nn.Parameter(torch.zeros(1, hidden_size))
        self.init_empty = nn.Parameter(torch.zeros(1, hidden_size))
        self.fin_empty = nn.Parameter(torch.zeros(1, hidden_size))
        nn.init.normal_(self.init_type)
        nn.init.normal_(self.fin_type)
        nn.init.normal_(self.init_empty)
        nn.init.normal_(self.fin_empty)

    def get_output_shape(self, height, width):
        batch, channel = 1, 3
        device = next(self.backbone.parameters()).device
        tensor = torch.zeros(batch, channel, height, width).to(device)
        n_box, box_dim = 2, 5
        box = torch.zeros(n_box, box_dim).to(device)
        n = (n_box,)
        _, c_flat = tuple(
            self.forward(tensor, tensor, box, box, n, n).flatten(
                start_dim=1).shape)
        return c_flat

    def _split(self, feature, batch_size):
        if isinstance(feature, torch.Tensor):
            left, right = feature[:batch_size], feature[batch_size:]
        elif isinstance(feature, Mapping):
            left, right = OrderedDict(), OrderedDict()
            for key, val in feature.items():
                left[key], right[key] = val[:batch_size], val[batch_size:]
        return left, right


    def _pad_roi(self, roi, n_box, fill_roi):
        # N (B * ?) * (256 * 2 * 2) => B * 11 * (256 * 2 * 2)
        roi_list = torch.split_with_sizes(roi, n_box, dim=0)
        roi = torch.stack(
            [torch.cat(
                [r, fill_roi.expand(self.MAX_BOX - r.size(0), -1)], dim=0)
            for r in roi_list])
        return roi

    def _encode_pos(self, boxes, h, w):
        boxes = boxes[:, 1:] / torch.tensor([w, h, w, h]).to(boxes)
        width = boxes[:, 2:3] - boxes[:, 0:1]
        height = boxes[:, 3:4] - boxes[:, 1:2]
        area = width * height
        pos = torch.cat([boxes, width, height, area], dim=1)
        return self.pos_encode(pos)

    def forward(self, init, fin, init_boxes, fin_boxes, n_init, n_fin):
        batch_size, _, h, w = init.shape
        if self.mode == 'concat':
            feat = torch.cat([init, fin], dim=1)
        elif self.mode == 'subtract':
            feat = init - fin
        elif self.mode == 'latter_concat':
            feat = torch.cat([init, fin], dim=0)
        else:
            raise NotImplementedError
        feat = self.backbone(feat)
        init_pos = self._encode_pos(init_boxes, h, w)
        fin_pos = self._encode_pos(fin_boxes, h, w)

        if self.mode in ['concat', 'subtract']:
            out = self._early_fusion(
                feat, init_boxes, fin_boxes, n_init, n_fin, init_pos, fin_pos)
        else:
            feat_init, feat_fin = self._split(feat, batch_size)
            out = self._latter_fusion(
                feat_init, feat_fin, init_boxes, fin_boxes, n_init, n_fin,
                init_pos, fin_pos)
        return out

    def _fuse_roi(self, roi, pos, _type, n_box, emtpy):
        roi = self.roi_encode(roi)
        roi = roi + pos + _type.expand_as(pos)
        roi = self._pad_roi(roi, n_box, emtpy)
        return roi.flatten(start_dim=1)

    def _early_fusion(
            self, feat, init_boxes, fin_boxes, n_init, n_fin,
            init_pos, fin_pos):
        # out = [feat['pool'].flatten(start_dim=1)]
        out = []

        for key in ['0', '1', '2', '3']:
            init_roi = self.roi_pool[key](
                feat[key], init_boxes).flatten(start_dim=1)
            init_roi = self._fuse_roi(
                init_roi, init_pos, self.init_type, n_init, self.init_empty)
            fin_roi = self.roi_pool[key](
                feat[key], fin_boxes).flatten(start_dim=1)
            fin_roi = self._fuse_roi(
                fin_roi, fin_pos, self.fin_type, n_fin, self.fin_empty)
            roi = torch.cat([init_roi, fin_roi], dim=1)
            a_roi = self.fuse_attention(roi)
            roi = self.fuse_norm(self.fuse_encode(roi * a_roi) + roi)
            out.append(roi)

        out = torch.cat(out, dim=1)
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