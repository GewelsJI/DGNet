import os
import cv2
from PIL import Image
import numpy as np


def canny_detect(img_path):
    img = cv2.imread(img_path)

    return cv2.Canny(img, 64, 128)


"""
camourflage_00051
camourflage_00109
"""


if __name__ == '__main__':
<<<<<<< HEAD
    src = '/home/admin/workspace/daniel_ji/workspace/alibaba-dgnet/dataset/TrainDatasetPolyp/image'
    src_gt = '/home/admin/workspace/daniel_ji/workspace/alibaba-dgnet/dataset/TrainDatasetPolyp/mask'
    dst_a = '/home/admin/workspace/daniel_ji/workspace/alibaba-dgnet/dataset/TrainDatasetPolyp/canny'
    dst_b = '/home/admin/workspace/daniel_ji/workspace/alibaba-dgnet/dataset/TrainDatasetPolyp/gradient-gt'
=======
    src = '/media/nercms/NERCMS/GepengJi/2020ACMMM/Dataset/Polyp_Data/TrainDataset/images'
    src_gt = '/media/nercms/NERCMS/GepengJi/2020ACMMM/Dataset/Polyp_Data/TrainDataset/masks'
    dst_a = '/media/nercms/NERCMS/GepengJi/2020ACMMM/Dataset/Polyp_Data/TrainDataset/Gradient-Canny'
    dst_b = '/media/nercms/NERCMS/GepengJi/2020ACMMM/Dataset/Polyp_Data/TrainDataset/Gradient-Foreground'
    # dst_c = '/Users/icbu-daniel/Documents/CVPR2022-SRCOD/data/TrainDataset/Edge'
>>>>>>> 4c882def2abc14bf96e871bcb8b479b1a27e2571

    os.makedirs(dst_a, exist_ok=True)
    os.makedirs(dst_b, exist_ok=True)
    # os.makedirs(dst_c, exist_ok=True)

<<<<<<< HEAD
    for img_name in os.listdir(src):
        canny_img = canny_detect(os.path.join(src, img_name))
        cv2.imwrite(os.path.join(dst_a, img_name.replace('.jpg', '.png')), canny_img)
        
=======
    # for img_name in os.listdir(src):
    #     canny_img = canny_detect(os.path.join(src, img_name))
    #     cv2.imwrite(os.path.join(dst_a, img_name.replace('.jpg', '.png')), canny_img)
>>>>>>> 4c882def2abc14bf96e871bcb8b479b1a27e2571

    for gt_name in os.listdir(src_gt):
        gt = Image.open(os.path.join(src_gt, gt_name)).convert('1')
        canny = Image.open(os.path.join(dst_a, gt_name)).convert('1')
<<<<<<< HEAD
        fore = Image.fromarray(np.array(canny) * np.array(gt))
        fore.save(os.path.join(dst_b, gt_name))

=======

        fore = Image.fromarray(np.array(canny) * np.array(gt))
        fore.save(os.path.join(dst_b, gt_name))
>>>>>>> 4c882def2abc14bf96e871bcb8b479b1a27e2571
