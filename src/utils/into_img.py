import cv2

def img2img(origin, into_img, side_set, set, init, side):
    or_h, or_w, or_c = origin.shape
    h, w, c = into_img.shape
    if side_set:
        side = (or_w-w)//2
    if set:
        into_img_init = or_h-h-side
    else:
        into_img_init = init
    roi = origin[into_img_init:into_img_init+h, side:side+w]
    mask = cv2.cvtColor(into_img, cv2.COLOR_BGR2GRAY)
    mask[mask[:]==255]=0
    mask[mask[:]>0]=255
    mask_inv = cv2.bitwise_not(mask)
    into_img = cv2.bitwise_and(into_img, into_img, mask=mask)
    back = cv2.bitwise_and(roi, roi, mask=mask_inv)
    dst = cv2.add(into_img, back)
    return into_img_init, h, w, side, dst


