import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        #self.aspect_ratios = cfg['aspect_ratios']
        # 'min_sizes': [[32, 64, 128], [256], [512]]
        self.min_sizes = cfg['min_sizes']
        # 'steps': [32, 64, 128]
        self.steps = cfg['steps']
        # 'clip': False
        self.clip = cfg['clip']
        # (800, 600)
        self.image_size = image_size
        # image size에서 step을 나누고 올림처리
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        # image size h /step, image_size w/step step 갯수만큼 반복
        for k, f in enumerate(self.feature_maps):
            # 'min_sizes': [[32, 64, 128], [256], [512]]
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.25, j+0.5, j+0.75]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.25, i+0.5, i+0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.5]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
