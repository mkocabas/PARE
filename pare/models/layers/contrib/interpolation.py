from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as tf


def _bchw2bhwc(tensor):
    return tensor.transpose(1,2).transpose(2,3)


def _bhwc2bchw(tensor):
    return tensor.transpose(2,3).transpose(1,2)


class Meshgrid(nn.Module):
    def __init__(self):
        super(Meshgrid, self).__init__()
        self.width = 0
        self.height = 0
        self.register_buffer("xx", torch.zeros(1,1))
        self.register_buffer("yy", torch.zeros(1,1))
        self.register_buffer("rangex", torch.zeros(1,1))
        self.register_buffer("rangey", torch.zeros(1,1))

    def _compute_meshgrid(self, width, height):
        torch.arange(0, width, out=self.rangex)
        torch.arange(0, height, out=self.rangey)
        self.xx = self.rangex.repeat(height, 1).contiguous()
        self.yy = self.rangey.repeat(width, 1).t().contiguous()

    def forward(self, width, height):
        if self.width != width or self.height != height:
            self._compute_meshgrid(width=width, height=height)
            self.width = width
            self.height = height
        return self.xx, self.yy


class BatchSub2Ind(nn.Module):
    def __init__(self):
        super(BatchSub2Ind, self).__init__()
        self.register_buffer("_offsets", torch.LongTensor())

    def forward(self, shape, row_sub, col_sub, out=None):
        batch_size = row_sub.size(0)
        height, width = shape
        ind = row_sub*width + col_sub
        torch.arange(batch_size, out=self._offsets)
        self._offsets *= (height*width)

        if out is None:
            return torch.add(ind, self._offsets.view(-1,1,1))
        else:
            torch.add(ind, self._offsets.view(-1,1,1), out=out)


class Interp2(nn.Module):
    def __init__(self, clamp=False):
        super(Interp2, self).__init__()
        self._clamp = clamp
        self._batch_sub2ind = BatchSub2Ind()
        self.register_buffer("_x0", torch.LongTensor())
        self.register_buffer("_x1", torch.LongTensor())
        self.register_buffer("_y0", torch.LongTensor())
        self.register_buffer("_y1", torch.LongTensor())
        self.register_buffer("_i00", torch.LongTensor())
        self.register_buffer("_i01", torch.LongTensor())
        self.register_buffer("_i10", torch.LongTensor())
        self.register_buffer("_i11", torch.LongTensor())
        self.register_buffer("_v00", torch.FloatTensor())
        self.register_buffer("_v01", torch.FloatTensor())
        self.register_buffer("_v10", torch.FloatTensor())
        self.register_buffer("_v11", torch.FloatTensor())
        self.register_buffer("_x", torch.FloatTensor())
        self.register_buffer("_y", torch.FloatTensor())

    def forward(self, v, xq, yq):
        batch_size, channels, height, width = v.size()

        # clamp if wanted
        if self._clamp:
            xq.clamp_(0, width - 1)
            yq.clamp_(0, height - 1)

        # ------------------------------------------------------------------
        # Find neighbors
        #
        # x0 = torch.floor(xq).long(),          x0.clamp_(0, width - 1)
        # x1 = x0 + 1,                          x1.clamp_(0, width - 1)
        # y0 = torch.floor(yq).long(),          y0.clamp_(0, height - 1)
        # y1 = y0 + 1,                          y1.clamp_(0, height - 1)
        #
        # ------------------------------------------------------------------
        torch.clamp(torch.floor(xq).long(), 0, width - 1, out=self._x0)
        torch.clamp(torch.floor(yq).long(), 0, height - 1, out=self._y0)

        torch.clamp(torch.add(self._x0, 1), 0, width - 1,  out=self._x1)
        torch.clamp(torch.add(self._y0, 1), 0, height - 1, out=self._y1)

        # batch_sub2ind
        self._batch_sub2ind([height, width], self._y0, self._x0, out=self._i00)
        self._batch_sub2ind([height, width], self._y0, self._x1, out=self._i01)
        self._batch_sub2ind([height, width], self._y1, self._x0, out=self._i10)
        self._batch_sub2ind([height, width], self._y1, self._x1, out=self._i11)

        # reshape
        v_flat = _bchw2bhwc(v).contiguous().view(-1, channels)
        torch.index_select(v_flat, dim=0, index=self._i00.view(-1), out=self._v00)
        torch.index_select(v_flat, dim=0, index=self._i01.view(-1), out=self._v01)
        torch.index_select(v_flat, dim=0, index=self._i10.view(-1), out=self._v10)
        torch.index_select(v_flat, dim=0, index=self._i11.view(-1), out=self._v11)

        # local_coords
        torch.add(xq, - self._x0.float(), out=self._x)
        torch.add(yq, - self._y0.float(), out=self._y)

        # weights
        w00 = torch.unsqueeze((1.0 - self._y) * (1.0 - self._x), dim=1)
        w01 = torch.unsqueeze((1.0 - self._y) * self._x, dim=1)
        w10 = torch.unsqueeze(self._y * (1.0 - self._x), dim=1)
        w11 = torch.unsqueeze(self._y * self._x, dim=1)

        def _reshape(u):
            return _bhwc2bchw(u.view(batch_size, height, width, channels))

        # values
        values = _reshape(self._v00)*w00 + _reshape(self._v01)*w01 \
            + _reshape(self._v10)*w10 + _reshape(self._v11)*w11

        if self._clamp:
            return values
        else:
            #  find_invalid
            invalid = ((xq < 0) | (xq >= width) | (yq < 0) | (yq >= height)).unsqueeze(dim=1).float()
            # maskout invalid
            transformed = invalid * torch.zeros_like(values) + (1.0 - invalid)*values

        return transformed


def resize2D(inputs, size_targets, mode="bilinear"):
    size_inputs = [inputs.size(2), inputs.size(3)]

    if all([size_inputs == size_targets]):
        return inputs  # nothing to do
    elif any([size_targets < size_inputs]):
        resized = tf.adaptive_avg_pool2d(inputs, size_targets)  # downscaling
    else:
        resized = tf.upsample(inputs, size=size_targets, mode=mode)  # upsampling

    # correct scaling
    return resized


def resize2D_as(inputs, output_as, mode="bilinear"):
    size_targets = [output_as.size(2), output_as.size(3)]
    return resize2D(inputs, size_targets, mode=mode)


class InverseWarpImage(nn.Module):
    def __init__(self, clamp=True):
        super(InverseWarpImage, self).__init__()
        self._meshgrid = Meshgrid()
        self._interp2 = Interp2(clamp=clamp)
        self._clamp = clamp

    def forward(self, image, u, v):
        height, width = image.size()[2:]
        xx, yy = self._meshgrid(width=width, height=height)
        xq = xx + torch.squeeze(u, dim=1)
        yq = yy + torch.squeeze(v, dim=1)

        return self._interp2(image, xq, yq)
