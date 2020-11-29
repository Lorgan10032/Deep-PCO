import os
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from PIL import Image


class FirstDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transforms=None):
        super(FirstDataset, self).__init__()
        self.transforms = transforms
        imgs = []
        for item in os.listdir(img_dir):
            name = item.split('.')[0]
            img_path = os.path.join(img_dir, item)
            label_path = os.path.join(annotation_dir, name + '.xml')
            imgs.append((img_path, label_path))
        self.imgs = imgs
        return

    def __getitem__(self, item):
        img_path, annotation_path = self.imgs[item]
        img = Image.open(img_path)
        target = self.parse_xml(annotation_path)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

    def parse_xml(self, annotation):
        root = ET.parse(annotation).getroot()
        target = {'labels': torch.tensor([1], dtype=torch.int64)}
        bndbox_node = root.find("./object/bndbox")
        box = []
        for i in range(4):
            box.append(int(bndbox_node[i].text))
        target['boxes'] = torch.tensor([box], dtype=torch.float)
        # target['boxes'] = [box]
        return target
