"""
역할: 인스턴스 마스크(npz)를 U-Net 이 쓰는 클래스 지도(png)로 바꾼다.

U-Net 은 '픽셀마다 클래스 하나'를 맞히는 모델이라, 객체를 따로 구분하지 않는다.
그래서 정답도 객체별 마스크가 아니라 한 장의 지도여야 한다.

  <데이터셋>/<split>/masks/<이름>.npz      : 객체별 마스크 (Mask R-CNN 용, 이미 만들어 둔 것)
        ↓
  <데이터셋>/<split>/semantic/<이름>.png   : 클래스 지도 (U-Net 용)
        0 = 배경, 1 = reflective_jacket, 2 = safety_helmet

겹치는 픽셀 처리
  한 픽셀에 두 객체가 겹치면 클래스를 하나만 남겨야 한다.
  면적이 작은 객체를 나중에 그려서 살린다. (헬멧이 조끼에 가려지지 않도록)
  같은 클래스끼리 겹치는 것은 어차피 같은 값이라 문제되지 않는다.
  실측 결과 다른 클래스끼리 겹치는 픽셀은 전체의 0.36% 였다.

실행 : python -m Preprocessing.convert_to_unet
"""

import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from Preprocessing.convert_to_maskrcnn import find_image_path, load_mask_file

SPLITS = ['train', 'valid', 'test']

# 결과가 저장될 폴더 이름
SEMANTIC_DIR = 'semantic'


def to_semantic(masks, classes):
    """
    객체별 마스크를 클래스 지도 한 장으로 합친다.

    masks   : (객체수, H, W) 0/1
    classes : (객체수,) 0부터 시작하는 클래스 번호
    돌려주는 값 : (H, W) uint8, 배경 0 / 클래스는 1부터
    """
    semantic = np.zeros(masks.shape[1:], dtype=np.uint8)

    # 면적이 큰 것부터 그린다. 작은 객체가 나중에 덮어써서 살아남는다.
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    order = np.argsort(-areas)

    for i in order:
        semantic[masks[i] == 1] = int(classes[i]) + 1

    return semantic


def convert_split(dataset_dir, split, overwrite=False):
    """split 하나를 클래스 지도로 바꾼다."""
    mask_dir = os.path.join(dataset_dir, split, 'masks')
    out_dir = os.path.join(dataset_dir, split, SEMANTIC_DIR)

    if not os.path.isdir(mask_dir):
        print(f'  {split:<6} : masks 폴더가 없어 건너뜁니다. '
              f'(먼저 python -m Preprocessing.convert_to_maskrcnn 실행)')
        return None

    mask_files = [f for f in sorted(os.listdir(mask_dir)) if f.endswith('.npz')]

    if not mask_files:
        print(f'  {split:<6} : 마스크가 없어 건너뜁니다.')
        return None

    os.makedirs(out_dir, exist_ok=True)

    done = skipped = 0
    class_pixels = {}

    for mask_file in tqdm(mask_files, desc=f'{split} 변환'):
        name = os.path.splitext(mask_file)[0]
        out_path = os.path.join(out_dir, name + '.png')

        if not overwrite and os.path.exists(out_path):
            skipped += 1
            continue

        masks, classes, _ = load_mask_file(os.path.join(mask_dir, mask_file))
        semantic = to_semantic(masks, classes)

        # 값이 0,1,2 뿐이라 png 로 저장하면 매우 작아진다
        Image.fromarray(semantic).save(out_path, optimize=True)

        for value, count in zip(*np.unique(semantic, return_counts=True)):
            class_pixels[int(value)] = class_pixels.get(int(value), 0) + int(count)

        done += 1

    print(f'  {split:<6} : 변환 {done}개 / 건너뜀 {skipped}개')

    if class_pixels:
        total = sum(class_pixels.values())
        share = ', '.join(f'{k}:{v / total * 100:.1f}%' for k, v in sorted(class_pixels.items()))
        print(f'           픽셀 비율 (0=배경) {share}')

    return {'split': split, 'done': done, 'skipped': skipped, 'class_pixels': class_pixels}


def convert_dataset(dataset_dir, overwrite=False):
    """데이터셋 전체를 클래스 지도로 바꾼다."""
    print(f'===== U-Net 용 클래스 지도 변환 : {dataset_dir} =====')

    if not os.path.isdir(dataset_dir):
        print('데이터셋 폴더가 없습니다.')
        return []

    results = []

    for split in SPLITS:
        result = convert_split(dataset_dir, split, overwrite=overwrite)

        if result:
            results.append(result)

    return results


def verify_split(dataset_dir, split, num_classes=2, sample_count=3):
    """
    변환이 제대로 됐는지 확인한다.

      1) 클래스 지도의 크기가 원본 이미지와 같은가
      2) 값이 0 ~ num_classes 범위 안인가
      3) 원본 마스크에 있던 클래스가 지도에도 남아 있는가
         (겹쳐서 완전히 가려진 객체는 사라질 수 있으므로 그 개수를 센다)
    """
    mask_dir = os.path.join(dataset_dir, split, 'masks')
    out_dir = os.path.join(dataset_dir, split, SEMANTIC_DIR)
    image_dir = os.path.join(dataset_dir, split, 'images')

    if not os.path.isdir(out_dir):
        print(f'  {split:<6} : 변환된 지도가 없습니다.')
        return []

    files = [f for f in sorted(os.listdir(out_dir)) if f.endswith('.png')]

    if not files:
        print(f'  {split:<6} : 변환된 지도가 없습니다.')
        return []

    problems = []
    hidden_objects = 0
    total_objects = 0
    total_size = 0

    for file_name in tqdm(files, desc=f'{split} 검증'):
        name = os.path.splitext(file_name)[0]
        path = os.path.join(out_dir, file_name)

        total_size += os.path.getsize(path)
        semantic = np.array(Image.open(path))

        # 1) 크기
        image_path = find_image_path(image_dir, name)
        image_w, image_h = Image.open(image_path).size

        if semantic.shape != (image_h, image_w):
            problems.append(f'{name} : 크기 {semantic.shape} != 이미지 ({image_h}, {image_w})')

        # 2) 값 범위
        values = set(np.unique(semantic).tolist())

        if not values <= set(range(num_classes + 1)):
            problems.append(f'{name} : 이상한 값 {sorted(values - set(range(num_classes + 1)))}')

        # 3) 원본 클래스가 남아 있는가
        masks, classes, _ = load_mask_file(os.path.join(mask_dir, name + '.npz'))
        total_objects += len(classes)

        for want in set(int(c) + 1 for c in classes):
            if want not in values:
                hidden_objects += 1
                problems.append(f'{name} : 클래스 {want} 가 겹쳐서 완전히 사라짐')

    mb = total_size / (1024 * 1024)
    print(f'  {split:<6} : 지도 {len(files)}장 / 객체 {total_objects}개 / 용량 {mb:.1f} MB')

    if problems:
        print(f'           문제 {len(problems)}건 (완전히 가려진 클래스 {hidden_objects}건)')
        for text in problems[:5]:
            print(f'             {text}')
    else:
        print('           문제 없음')

    if sample_count:
        save_sample(dataset_dir, split, files, sample_count)

    return problems


def save_sample(dataset_dir, split, files, sample_count=3):
    """클래스 지도를 원본 이미지에 겹쳐 그려 저장한다. (눈으로 확인용)"""
    import random

    import cv2
    import matplotlib.pyplot as plt

    from utils.visualize import get_color

    image_dir = os.path.join(dataset_dir, split, 'images')
    out_dir = os.path.join(dataset_dir, split, SEMANTIC_DIR)

    picked = random.sample(files, min(sample_count, len(files)))

    _, axes = plt.subplots(1, len(picked), figsize=(5 * len(picked), 5))

    if len(picked) == 1:
        axes = [axes]

    for ax, file_name in zip(axes, picked):
        name = os.path.splitext(file_name)[0]

        image = cv2.imread(find_image_path(image_dir, name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        semantic = np.array(Image.open(os.path.join(out_dir, file_name)))

        colored = image.copy()

        # 클래스마다 다른 색으로 반투명하게 덮는다 (0 은 배경이라 건너뛴다)
        for value in sorted(set(np.unique(semantic).tolist()) - {0}):
            color = np.array(get_color(value - 1), dtype=float)
            area = semantic == value
            colored[area] = (colored[area] * 0.5 + color * 0.5).astype(np.uint8)

        ax.imshow(colored)
        ax.set_title(f'{name[:20]}...', fontsize=8)
        ax.axis('off')

    save_path = os.path.join('./result', f'unet_convert_{split}.jpg')
    os.makedirs('./result', exist_ok=True)

    plt.suptitle(f'U-Net semantic map check ({split})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f'           확인용 그림 저장 : {save_path}')


def verify_dataset(dataset_dir, num_classes=2, sample_count=3):
    """데이터셋 전체의 변환 결과를 확인한다."""
    print(f'===== 변환 결과 확인 : {dataset_dir} =====')

    problems = []

    for split in SPLITS:
        problems += verify_split(dataset_dir, split, num_classes, sample_count)

    return problems


if __name__ == '__main__':
    from Preprocessing.check_dataset import SEGMENT_DATASET

    convert_dataset(SEGMENT_DATASET)
    verify_dataset(SEGMENT_DATASET)
