import os
import cv2
from PIL import Image
import numpy as np


def canny_detect(img_path):
    img = cv2.imread(img_path)

    return cv2.Canny(img, 64, 128)


if __name__ == '__main__':
    src = './dataset/TrainDataset/image'
    src_gt = './dataset/TrainDataset/mask'
    dst_a = './dataset/TrainDataset/canny'
    dst_b = './dataset/TrainDataset/gradient-gt'

    os.makedirs(dst_a, exist_ok=True)
    os.makedirs(dst_b, exist_ok=True)

    for img_name in os.listdir(src):
        canny_img = canny_detect(os.path.join(src, img_name))
        cv2.imwrite(os.path.join(dst_a, img_name.replace('.jpg', '.png')), canny_img)
        

    for gt_name in os.listdir(src_gt):
        gt = Image.open(os.path.join(src_gt, gt_name)).convert('1')
        canny = Image.open(os.path.join(dst_a, gt_name)).convert('1')
        fore = Image.fromarray(np.array(canny) * np.array(gt))
        fore.save(os.path.join(dst_b, gt_name))
