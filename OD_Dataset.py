import os
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageChops


class OD_Dataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transforms=None):
        super(OD_Dataset, self).__init__()
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


if __name__ == '__main__':
    img_dir = '/home/shiyaohu/Data/pco_data/imgs'
    annotation_dir = '/home/shiyaohu/Data/pco_data/label'

    # delete duplicate images and annotations
    imgs = []
    items = os.listdir(img_dir)
    for item in items:
        this_name = item.split('.')[0]
        img_path = os.path.join(img_dir, item)
        label_path = os.path.join(annotation_dir, this_name + '.xml')
        this_img = Image.open(img_path)
        repeat = False
        for name, img in imgs:
            diff = ImageChops.difference(img, this_img)
            if not diff.getbbox():
                print('remove', this_name, 'same as', name)
                # delete this img
                os.remove(img_path)
                # delete this annotation
                os.remove(label_path)
                repeat = True
                break
        if not repeat:
            imgs.append((this_name, this_img))

    # rename rest images and annotations
    n = 1
    items = os.listdir(img_dir)
    print('count:', len(items))
    for item in items:
        path = os.path.join(img_dir, item)
        name = item.split('.')[0]
        label_path = os.path.join(annotation_dir, name + '.xml')
        if os.path.isfile(path):
            filename = str(n).zfill(4) + '.jpg'
            filename_label = str(n).zfill(4) + '.xml'
            os.rename(path, os.path.join(img_dir, filename))
            os.rename(label_path, os.path.join(annotation_dir, filename_label))
            n += 1
