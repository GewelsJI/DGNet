import os
import numpy as np
from PIL import Image

from jittor import transform
from jittor.dataset import Dataset


class test_dataset(Dataset):

    def __init__(self, image_root, gt_root, testsize):
        super().__init__()

        self.testsize = testsize
        self.images = [(image_root + f) for f in os.listdir(image_root) if (f.endswith('.jpg') or f.endswith('.png'))]
        self.gts = [(gt_root + f) for f in os.listdir(gt_root) if (f.endswith('.tif') or f.endswith('.png'))]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transform.Compose([
            transform.Resize((self.testsize, self.testsize)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transform.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[(- 1)]
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = (name.split('.jpg')[0] + '.png')
        self.index += 1
        self.index = (self.index % self.size)
        return (image, gt, name, np.array(image_for_post))

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
