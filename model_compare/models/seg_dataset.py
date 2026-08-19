"""
역할: Mask R-CNN 학습에 쓸 데이터를 읽어오는 Dataset

Preprocessing/convert_to_maskrcnn.py 가 미리 만들어 둔 마스크 파일(npz)을 읽는다.
학습할 때마다 폴리곤을 그리지 않으므로 빠르고, Mask R-CNN 만 따로 돌릴 수도 있다.

  <데이터셋>/<split>/images/<이름>.jpg   : 원본 이미지
  <데이터셋>/<split>/masks/<이름>.npz    : 변환된 마스크 (masks, classes, boxes)

Mask R-CNN 이 요구하는 형태
  boxes  -> [x1, y1, x2, y2] 픽셀 좌표
  labels -> 클래스 번호 (0 은 배경이라 +1 해서 넣는다)
  masks  -> 0/1 로 채워진 이미지 크기의 마스크 (객체마다 한 장)

--- 데이터 증강 ---
YOLO 는 학습할 때 좌우반전·색상변화 같은 증강을 알아서 넣는다.
Mask R-CNN 쪽에 증강이 없으면 비교 결과가 '모델 차이'가 아니라 '증강 유무 차이'가 되므로,
train 에서만 YOLO 와 비슷한 수준의 증강을 넣는다.
  - 좌우 반전 : 이미지·마스크·박스를 함께 뒤집는다
  - 색상 변화 : 이미지만 바꾼다 (위치가 안 바뀌므로 라벨은 그대로)
"""

import os
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXT = ('.jpg', '.jpeg', '.png')

# 증강 세기 (YOLO 기본값과 비슷하게 맞췄다)
FLIP_PROB = 0.5         # 좌우 반전 확률 (YOLO fliplr=0.5)
BRIGHTNESS = 0.4        # 밝기 변화 폭 (YOLO hsv_v=0.4)
SATURATION = 0.7        # 채도 변화 폭 (YOLO hsv_s=0.7)
HUE = 0.015             # 색조 변화 폭 (YOLO hsv_h=0.015)


def find_image_path(image_dir, name):
    """확장자를 모르는 상태에서 이미지 파일을 찾는다."""
    for ext in IMAGE_EXT:
        path = os.path.join(image_dir, name + ext)

        if os.path.exists(path):
            return path

    return None


def flip_horizontal(image, masks, boxes):
    """
    이미지를 좌우로 뒤집고 마스크와 박스도 같이 뒤집는다.
    박스는 왼쪽 끝과 오른쪽 끝이 서로 바뀐다 : x1 -> W - x2,  x2 -> W - x1
    """
    image_w = image.size[0]

    image = ImageOps.mirror(image)
    masks = masks[:, :, ::-1].copy()      # 마지막 축(가로)을 뒤집는다

    boxes = boxes.copy()
    old_x1 = boxes[:, 0].copy()
    boxes[:, 0] = image_w - boxes[:, 2]
    boxes[:, 2] = image_w - old_x1

    return image, masks, boxes


class SegDataset(Dataset):
    """변환된 마스크 폴더를 읽어 (이미지, 타겟) 을 돌려주는 데이터셋"""

    def __init__(self, dataset_dir, split, augment=False):
        self.image_dir = os.path.join(dataset_dir, split, 'images')
        self.mask_dir = os.path.join(dataset_dir, split, 'masks')
        self.augment = augment
        self.to_tensor = transforms.ToTensor()

        # 색상 변화는 이미지 내용만 바꾸므로 라벨을 건드리지 않는다
        self.color_jitter = transforms.ColorJitter(brightness=BRIGHTNESS,
                                                   saturation=SATURATION,
                                                   hue=HUE)

        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(
                f'변환된 마스크 폴더가 없습니다 : {self.mask_dir}\n'
                f'  python -m Preprocessing.convert_to_maskrcnn 을 먼저 실행하세요.')

        # 마스크와 이미지가 짝을 이루는 것만 쓴다
        self.samples = []

        for mask_file in sorted(os.listdir(self.mask_dir)):
            if not mask_file.endswith('.npz'):
                continue

            name = os.path.splitext(mask_file)[0]
            image_path = find_image_path(self.image_dir, name)

            if image_path is None:
                continue

            self.samples.append({
                'image_path': image_path,
                'mask_path': os.path.join(self.mask_dir, mask_file),
            })

        state = '증강 사용' if augment else '증강 없음'
        print(f'{self.mask_dir} : 이미지 {len(self.samples)}개 ({state})')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(sample['image_path']).convert('RGB')

        data = np.load(sample['mask_path'])
        masks = data['masks']       # (객체수, H, W) 0/1
        classes = data['classes']   # (객체수,) 0부터
        boxes = data['boxes'].astype(np.float32)   # (객체수, 4) 픽셀 좌표

        # train 에서만 증강한다
        if self.augment:
            if random.random() < FLIP_PROB:
                image, masks, boxes = flip_horizontal(image, masks, boxes)

            image = self.color_jitter(image)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(classes + 1, dtype=torch.int64),   # 0 은 배경
            'masks': torch.tensor(masks, dtype=torch.uint8),
            'image_id': torch.tensor([index]),
            'area': torch.tensor((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                                 dtype=torch.float32),
            'iscrowd': torch.zeros(len(classes), dtype=torch.int64),
        }

        return self.to_tensor(image), target


def collate_fn(batch):
    """
    이미지마다 객체 수가 달라서 하나의 텐서로 못 묶는다.
    그래서 배치를 (이미지 튜플, 타겟 튜플) 로 그냥 묶어준다. (torchvision 검출 모델 필수)
    """
    return tuple(zip(*batch))


def get_dataloader(dataset_dir, split, batch_size=2, shuffle=True, augment=False):
    """
    dataset_dir/split 로 DataLoader 를 만든다.
    증강은 train 에서만 켠다.
    예) get_dataloader('./Data/vest-helmet_crop_dedup_seg', 'train', augment=True)
    """
    dataset = SegDataset(dataset_dir, split, augment=augment)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=0,
                        collate_fn=collate_fn)

    return loader


def has_split(dataset_dir, split):
    """그 split 에 변환된 마스크가 있는지 확인한다."""
    mask_dir = os.path.join(dataset_dir, split, 'masks')

    if not os.path.isdir(mask_dir):
        return False

    return any(f.endswith('.npz') for f in os.listdir(mask_dir))
