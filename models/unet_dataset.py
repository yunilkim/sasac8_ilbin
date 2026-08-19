"""
Preprocessing/convert_to_unet.py 가 만들어 둔 클래스 지도(png)를 사용

    <데이터셋>/<split>/images/<이름>.jpg     : 원본 이미지
    <데이터셋>/<split>/semantic/<이름>.png   : 클래스 지도 (0=배경, 1,2=클래스)

증강은 yolov8n-seg 와 같은 수준으로 맞춘다. (좌우반전 + 색상변화)
"""

import os
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXT = ('.jpg', '.jpeg', '.png')
SEMANTIC_DIR = 'semantic'

# ImageNet 사전학습 인코더를 쓰므로 같은 방식으로 정규화
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# 증강 세기 (models/seg_dataset.py 와 동일하게 맞췄다)
FLIP_PROB = 0.5
BRIGHTNESS = 0.4
SATURATION = 0.7
HUE = 0.015

# 이미지 파일 찾기
def find_image_path(image_dir, name):
    for ext in IMAGE_EXT:
        path = os.path.join(image_dir, name + ext)

        if os.path.exists(path):
            return path

    return None

# split된 클래스 지도 확인
def has_split(dataset_dir, split):
    semantic_dir = os.path.join(dataset_dir, split, SEMANTIC_DIR)

    if not os.path.isdir(semantic_dir):
        return False

    return any(f.endswith('.png') for f in os.listdir(semantic_dir))


# 만들어진 라벨지도로 데이터셋 작성
class UnetDataset(Dataset):
    def __init__(self, dataset_dir, split, augment=False, input_size=640):
        self.image_dir = os.path.join(dataset_dir, split, 'images')
        self.semantic_dir = os.path.join(dataset_dir, split, SEMANTIC_DIR)
        self.augment = augment
        self.input_size = input_size

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])
        self.color_jitter = transforms.ColorJitter(brightness=BRIGHTNESS,
                                                    saturation=SATURATION,
                                                    hue=HUE)

        if not os.path.isdir(self.semantic_dir):
            raise FileNotFoundError(
                f'클래스 지도 폴더가 없습니다 : {self.semantic_dir}\n'
                f'  python -m Preprocessing.convert_to_unet 을 먼저 실행하세요.')

        self.samples = []

        for file_name in sorted(os.listdir(self.semantic_dir)):
            if not file_name.endswith('.png'):
                continue

            name = os.path.splitext(file_name)[0]
            image_path = find_image_path(self.image_dir, name)

            if image_path is None:
                continue

            self.samples.append({
                'image_path': image_path,
                'semantic_path': os.path.join(self.semantic_dir, file_name),
            })

        state = '증강 사용' if augment else '증강 없음'
        print(f'{self.semantic_dir} : 이미지 {len(self.samples)}개 ({state})')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(sample['image_path']).convert('RGB')
        semantic = Image.open(sample['semantic_path'])

        # 배치로 묶으려면 크기가 같아야 한다
        size = (self.input_size, self.input_size)
        image = image.resize(size, Image.BILINEAR)
        # 지도는 클래스 번호라서 섞이면 안 된다. 최근접 이웃으로 키운다.
        semantic = semantic.resize(size, Image.NEAREST)

        if self.augment:
            if random.random() < FLIP_PROB:
                image = ImageOps.mirror(image)
                semantic = ImageOps.mirror(semantic)

            image = self.color_jitter(image)

        target = torch.from_numpy(np.array(semantic, dtype=np.int64))

        return self.normalize(image), target


# 데이터셋 만들기 실행 함수
def get_dataloader(dataset_dir, split, batch_size=8, shuffle=True, augment=False, input_size=640):
    dataset = UnetDataset(dataset_dir, split, augment=augment, input_size=input_size)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=0)

    return loader
