import cv2
import numpy as np
import clean_ear_utils
import time
from functions import *
import faceboxes_detector

def demo_image(image, net, preprocess, input_size, net_stride, num_nb, use_gpu, device, reverse_index1, reverse_index2, max_len):
    st_time = time.time()
    detector = faceboxes_detector.FaceBoxesDetector('FaceBoxes', '/workspace/Eye_blink_detection/src/weights/FaceBoxesV2.pth', use_gpu, device)


    det_box_scale = 1.2

    net.eval()
    image_height, image_width, _ = image.shape
    detections, _ = detector.detect(image, 'max', 1)
    for i in range(len(detections)):
        det_xmin = detections[i][2]
        det_ymin = detections[i][3]
        det_width = detections[i][4]
        det_height = detections[i][5]
        det_xmax = det_xmin + det_width - 1
        det_ymax = det_ymin + det_height - 1

        det_xmin -= int(det_width * (det_box_scale-1)/2)
        # remove a part of top area for alignment, see paper for details
        det_ymin += int(det_height * (det_box_scale-1)/2)
        det_xmax += int(det_width * (det_box_scale-1)/2)
        det_ymax += int(det_height * (det_box_scale-1)/2)
        det_xmin = max(det_xmin, 0)
        det_ymin = max(det_ymin, 0)
        det_xmax = min(det_xmax, image_width-1)
        det_ymax = min(det_ymax, image_height-1)
        det_width = det_xmax - det_xmin + 1
        det_height = det_ymax - det_ymin + 1
       # cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
        det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
        det_crop = cv2.resize(det_crop, (input_size, input_size))
        inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
        inputs = preprocess(inputs).unsqueeze(0)
        inputs = inputs.to(device)


        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, input_size, net_stride, num_nb)
        if torch.mean(max_cls).detach().cpu().numpy() < 0.7:
            100/0
        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
        tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(98, max_len)
        tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(98, max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        lms_pred = lms_pred.cpu().numpy()
        lms_pred_merge = lms_pred_merge.cpu().numpy()

    
    l_eye, r_eye = clean_ear_utils.ld_eye(lms_pred_merge,det_width,det_height,det_xmin,det_ymin)
    
    l_ear = clean_ear_utils.l_calculate_EAR(l_eye)
    r_ear = clean_ear_utils.l_calculate_EAR(r_eye)
    
    EAR = np.round((l_ear+r_ear)/2, 2)
    eye_region_landmarks = []
    for i in [72,68,60,64,61,62,63,65,66,67,69,70,71,71,73,74,75,96,97]:
        x_pred = lms_pred_merge[i*2] * det_width
        y_pred = lms_pred_merge[i*2+1] * det_height
        xy = [int(x_pred)+det_xmin, int(y_pred)+det_ymin]
        eye_region_landmarks.append(xy)
        
    eye_region_landmarks = np.array(eye_region_landmarks)
    #cv2.circle(image, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 1)
            
    #for i in [69,75,71,73,68,72,63,65,61,67,64,60]:
     #   x_pred = lms_pred_merge[i*2] * det_width
      #  y_pred = lms_pred_merge[i*2+1] * det_height
       # cv2.circle(image, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 1)
    # 코 중심
    check = [int(lms_pred_merge[51*2]*det_width)+det_xmin,int(lms_pred_merge[51*2+1]* det_height)+det_ymin]
    ed_time = time.time()
    total_time = ed_time - st_time
    return image, EAR, total_time, check, eye_region_landmarks





