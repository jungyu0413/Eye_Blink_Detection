from detector import Detector
import cv2, os
import numpy as np
import torch
import torch.nn as nn
from utils.config import cfg
from utils.prior_box import PriorBox
from utils.nms_wrapper import nms
from utils.faceboxes import FaceBoxesV2
from utils.box_utils import decode
import time
from scipy.spatial import distance

class FaceBoxesDetector(Detector):
    def __init__(self, model_arch, model_weights, use_gpu, device):
        super().__init__(model_arch, model_weights)
        self.name = 'FaceBoxesDetector'
        self.device = device
        # face detector
        self.net = FaceBoxesV2(phase='test', size=None, num_classes=2)    # https://arxiv.org/abs/1708.05234 FaceBoxes        self.device = device

        state_dict = torch.load(self.model_weights, map_location=self.device)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        self.net.load_state_dict(new_state_dict)
        self.net = self.net.to(self.device)
        self.net.eval()

    # input image, thresh, 1
    # image_scale, 
    def detect(self, image, thresh=0.8, im_scale=None):
        # auto resize for large images
        if im_scale is None:
            height, width, _ = image.shape
            if min(height, width) > 600:
                im_scale = 600. / min(height, width)
            else:
                im_scale = 1
        # fx, fy : x,y 방향의 scale 비율 1이기에 원본과 차이없음
        image_scale = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        # tensor([800., 600., 800., 600.])
        scale = torch.Tensor([image_scale.shape[1], image_scale.shape[0], image_scale.shape[1], image_scale.shape[0]])
        image_scale = torch.from_numpy(image_scale.transpose(2,0,1)).to(self.device).int()
        mean_tmp = torch.IntTensor([104, 117, 123]).to(self.device)
        # torsh.shape : [3] -> [3, 1, 1]
        mean_tmp = mean_tmp.unsqueeze(1).unsqueeze(2)
        # 채널별 평균값 빼기??
        image_scale -= mean_tmp
        image_scale = image_scale.float().unsqueeze(0)
        scale = scale.to(self.device)
        with torch.no_grad():
            out = self.net(image_scale) #/workspace/debug_img/image_scale_2.png
            
            #priorbox = PriorBox(cfg, out[2], (image_scale.size()[2], image_scale.size()[3]), phase='test')
            priorbox = PriorBox(cfg, image_size=(image_scale.size()[2], image_scale.size()[3]))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            loc, conf = out
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale
            boxes = boxes.cpu().numpy()
            scores = conf.data.cpu().numpy()[:, 1]
            
            box_dis = [i[2]-i[0] for i in boxes]
            bbox_thresh = 100
            # box size
            inds_bbox = np.where(np.array(box_dis) >= bbox_thresh)[0]
            boxes = boxes[inds_bbox]
            scores = scores[inds_bbox]

            
            # ignore low scores
            if thresh == 'max':
                thresh = np.max(scores)

            # score
                inds = np.where(scores >= thresh)[0]
                boxes = boxes[inds]
                scores = scores[inds]

                dets = boxes
                score = scores[0]
                detections_scale = []
                xmin = int(dets[0][0])
                ymin = int(dets[0][1])
                xmax = int(dets[0][2])
                ymax = int(dets[0][3])
                width = xmax - xmin
                height = ymax - ymin
                detections_scale.append(['face', score, xmin, ymin, width, height])

            else:
                # keep top-K before NMS
                order = scores.argsort()[::-1][:5000]
                boxes = boxes[order]
                scores = scores[order]

                # do NMS
                # max를 위에서 걸렀기에 NMS 불필요

                dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = nms(dets, 0.3)
                dets = dets[keep, :]

                dets = dets[:750, :]

                detections_scale = []
                for i in range(dets.shape[0]):
                    xmin = int(dets[i][0])
                    ymin = int(dets[i][1])
                    xmax = int(dets[i][2])
                    ymax = int(dets[i][3])
                    score = dets[i][4]
                    width = xmax - xmin
                    height = ymax - ymin
                    detections_scale.append(['face', score, xmin, ymin, width, height])

        # adapt bboxes to the original image size
        if len(detections_scale) > 0:
            detections_scale = [[det[0],det[1],int(det[2]/im_scale),int(det[3]/im_scale),int(det[4]/im_scale),int(det[5]/im_scale)] for det in detections_scale]

        return detections_scale, im_scale

