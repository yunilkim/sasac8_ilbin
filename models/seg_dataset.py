"""
역할: segmentation 라벨(폴리곤)을 torch 모델이 먹을 수 있는 형태로 바꿔주는 Dataset

YOLO 라벨과 torch(Mask R-CNN) 가 원하는 형태가 다르다.
  YOLO 라벨 : <클래스> <x1> <y1> <x2> <y2> ...   (0~1 로 정규화된 폴리곤 꼭짓점)
  Mask R-CNN : boxes  -> [x1, y1, x2, y2] 픽셀 좌표
               labels -> 클래스 번호 (0 은 배경이라 +1 해서 넣는다)
               masks  -> 0/1 로 채워진 이미지 크기의 마스크

그래서 폴리곤을 그려서 마스크를 만들고, 폴리곤을 감싸는 사각형으로 박스를 만든다.
"""

import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.metrics import read_label_polygons

IMAGE_EXT = ('.jpg', '.jpeg', '.png')


def polygon_to_mask(points, image_w, image_h):
    """정규화된 폴리곤 좌표로 0/1 마스크 이미지를 그린다."""
    mask_image = Image.new('L', (image_w, image_h), 0)
    draw = ImageDraw.Draw(mask_image)

    # 0~1 값을 픽셀 좌표로 바꾼다
    xy = [(x * image_w, y * image_h) for x, y in points]
    draw.polygon(xy, outline=1, fill=1)

    return np.array(mask_image, dtype=np.uint8)


def polygon_to_box(points, image_w, image_h):
    """폴리곤을 감싸는 사각형 [x1, y1, x2, y2] 를 픽셀 좌표로 돌려준다."""
    xs = [x * image_w for x, y in points]
    ys = [y * image_h for x, y in points]

    return [min(xs), min(ys), max(xs), max(ys)]


class SegDataset(Dataset):
    """segmentation 라벨 폴더를 읽어 (이미지, 타겟) 을 돌려주는 데이터셋"""

    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.label_dir = image_dir.replace('images', 'labels')
        self.to_tensor = transforms.ToTensor()

        # 라벨이 없거나 폴리곤이 하나도 없는 이미지는 학습에 쓸 수 없으므로 미리 걸러낸다
        self.samples = []

        for file_name in sorted(os.listdir(image_dir)):
            if not file_name.lower().endswith(IMAGE_EXT):
                continue

            txt_path = os.path.join(self.label_dir, os.path.splitext(file_name)[0] + '.txt')

            if not os.path.exists(txt_path):
                continue

            if len(read_label_polygons(txt_path)) == 0:
                continue

            self.samples.append({
                'image_path': os.path.join(image_dir, file_name),
                'label_path': txt_path,
            })

        print(f'{image_dir} : 학습에 쓸 이미지 {len(self.samples)}개')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(sample['image_path']).convert('RGB')
        image_w, image_h = image.size

        boxes = []
        labels = []
        masks = []

        for class_no, points in read_label_polygons(sample['label_path']):
            box = polygon_to_box(points, image_w, image_h)

            # 너비나 높이가 0 인 박스는 학습 중 오류가 나므로 건너뛴다
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            boxes.append(box)
            labels.append(class_no + 1)     # 0 은 배경이라 1부터 시작
            masks.append(polygon_to_mask(points, image_w, image_h))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        masks = torch.tensor(np.array(masks), dtype=torch.uint8)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([index]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros(len(labels), dtype=torch.int64),
        }

        return self.to_tensor(image), target


def collate_fn(batch):
    """
    이미지마다 객체 수가 달라서 하나의 텐서로 못 묶는다.
    그래서 배치를 (이미지 튜플, 타겟 튜플) 로 그냥 묶어준다. (torchvision 검출 모델 필수)
    """
    return tuple(zip(*batch))


def get_dataloader(dataset_dir, split, batch_size=2, shuffle=True):
    """
    dataset_dir/split/images 폴더로 DataLoader 를 만든다.
    예) get_dataloader('./Data/vest-helmet-seg', 'train')
    """
    image_dir = os.path.join(dataset_dir, split, 'images')
    dataset = SegDataset(image_dir)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=0,
                        collate_fn=collate_fn)

    return loader
