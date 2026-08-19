"""
역할: YOLO seg 라벨(폴리곤 txt)을 Mask R-CNN 이 쓰는 마스크 파일로 미리 바꿔 둔다.

왜 미리 바꾸나
  YOLO(yolov8n-seg)는 폴리곤 txt 를 그대로 읽지만, torchvision Mask R-CNN 은
  '픽셀이 칠해진 마스크'가 필요하다. 학습할 때마다 폴리곤을 그리면 느리고,
  Mask R-CNN 만 따로 돌릴 때도 매번 변환 코드를 거쳐야 한다.
  그래서 학습 전에 한 번만 변환해서 파일로 저장해 둔다.

왜 png 가 아니라 npz 인가
  처음에는 '객체마다 번호를 칠한 png 한 장'으로 만들었는데, 이 데이터셋은
  사람들이 서로 겹쳐 있어서 뒤에 그린 객체가 앞 객체를 덮어버렸다.
  (심한 경우 한 객체의 99.5% 가 사라졌다)
  Mask R-CNN 은 객체마다 따로 마스크가 필요하므로,
  객체별 마스크를 그대로 쌓아서 npz 한 파일에 저장한다.

만들어지는 것
  <데이터셋>/<split>/masks/<이름>.npz
      masks   : (객체수, 높이, 너비) uint8   0 또는 1
      classes : (객체수,) int64             클래스 번호 (0부터)
      boxes   : (객체수, 4) int64           마스크에서 구한 [x1, y1, x2, y2]

실행 : python -m Preprocessing.convert_to_maskrcnn
"""

import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.metrics import read_label_polygons

SPLITS = ['train', 'valid', 'test']
IMAGE_EXT = ('.jpg', '.jpeg', '.png')

# 마스크 면적이 이 픽셀 수보다 작으면 학습에 도움이 안 되므로 버린다
MIN_AREA = 4


def find_image_path(image_dir, name):
    """확장자를 모르는 상태에서 이미지 파일을 찾는다."""
    for ext in IMAGE_EXT:
        path = os.path.join(image_dir, name + ext)

        if os.path.exists(path):
            return path

    return None


def polygon_to_pixels(points, image_w, image_h):
    """0~1 정규화 폴리곤을 픽셀 좌표 배열로 바꾼다."""
    return np.array([[x * image_w, y * image_h] for x, y in points], dtype=np.int32)


def convert_one(label_path, image_w, image_h):
    """
    라벨 파일 하나를 객체별 마스크로 바꾼다.
    객체마다 빈 도화지에 따로 그리기 때문에 서로 겹쳐도 잘리지 않는다.

    돌려주는 값 : (masks, classes, boxes, 버린 객체 수)
      masks   : (객체수, H, W) uint8
      classes : (객체수,) int64
      boxes   : (객체수, 4) int64
    """
    masks = []
    classes = []
    boxes = []
    dropped = 0

    for class_no, points in read_label_polygons(label_path):
        pixels = polygon_to_pixels(points, image_w, image_h)

        # 객체 하나만 있는 빈 도화지에 그린다
        mask = np.zeros((image_h, image_w), dtype=np.uint8)
        cv2.fillPoly(mask, [pixels], color=1)

        ys, xs = np.where(mask == 1)

        # 너무 작거나 (선처럼 얇아서) 칠해지지 않은 객체는 버린다
        if len(xs) < MIN_AREA:
            dropped += 1
            continue

        masks.append(mask)
        classes.append(class_no)
        # 박스는 폴리곤 좌표가 아니라 실제로 칠해진 픽셀에서 구한다
        # (이미지 밖으로 나간 부분이 잘려도 마스크와 박스가 항상 일치한다)
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])

    if not masks:
        return None, None, None, dropped

    return (np.array(masks, dtype=np.uint8),
            np.array(classes, dtype=np.int64),
            np.array(boxes, dtype=np.int64),
            dropped)


def convert_split(dataset_dir, split, overwrite=False):
    """
    split 하나를 통째로 변환한다.
    overwrite=False 면 이미 만들어진 파일은 건너뛴다.
    """
    image_dir = os.path.join(dataset_dir, split, 'images')
    label_dir = os.path.join(dataset_dir, split, 'labels')
    mask_dir = os.path.join(dataset_dir, split, 'masks')

    if not os.path.isdir(label_dir):
        print(f'  {split:<6} : labels 폴더가 없어 건너뜁니다.')
        return None

    label_files = [f for f in sorted(os.listdir(label_dir)) if f.endswith('.txt')]

    if not label_files:
        print(f'  {split:<6} : 라벨이 없어 건너뜁니다.')
        return None

    os.makedirs(mask_dir, exist_ok=True)

    done = skipped = no_image = empty = 0
    total_objects = total_dropped = 0

    for label_file in tqdm(label_files, desc=f'{split} 변환'):
        name = os.path.splitext(label_file)[0]
        mask_path = os.path.join(mask_dir, name + '.npz')

        if not overwrite and os.path.exists(mask_path):
            skipped += 1
            continue

        image_path = find_image_path(image_dir, name)

        if image_path is None:
            no_image += 1
            continue

        # 이미지 크기만 필요하므로 헤더만 읽는다 (전체를 읽지 않아 빠르다)
        image_w, image_h = Image.open(image_path).size

        masks, classes, boxes, dropped = convert_one(
            os.path.join(label_dir, label_file), image_w, image_h)

        total_dropped += dropped

        if masks is None:
            empty += 1
            continue

        # 0과 1만 있는 배열이라 압축하면 크기가 확 줄어든다
        np.savez_compressed(mask_path, masks=masks, classes=classes, boxes=boxes)

        done += 1
        total_objects += len(classes)

    print(f'  {split:<6} : 변환 {done}개 / 건너뜀 {skipped}개 / 이미지없음 {no_image}개 / 빈라벨 {empty}개')
    print(f'           객체 {total_objects}개 (너무 작아 버린 객체 {total_dropped}개)')

    return {'split': split, 'done': done, 'skipped': skipped, 'no_image': no_image,
            'empty': empty, 'objects': total_objects, 'dropped': total_dropped}


def convert_dataset(dataset_dir, overwrite=False):
    """데이터셋 전체(train/valid/test)를 변환한다."""
    print(f'===== Mask R-CNN 용 마스크 변환 : {dataset_dir} =====')

    if not os.path.isdir(dataset_dir):
        print('데이터셋 폴더가 없습니다.')
        return []

    results = []

    for split in SPLITS:
        result = convert_split(dataset_dir, split, overwrite=overwrite)

        if result:
            results.append(result)

    return results


def load_mask_file(mask_path):
    """변환된 npz 를 읽어 (masks, classes, boxes) 로 돌려준다."""
    data = np.load(mask_path)

    return data['masks'], data['classes'], data['boxes']


def verify_split(dataset_dir, split, sample_count=3):
    """
    변환이 제대로 됐는지 확인한다.

      1) 라벨의 객체 수와 저장된 마스크 수가 맞는가
      2) 저장된 박스가 마스크와 일치하는가
      3) 클래스 번호가 라벨과 같은가
      4) 겹친 객체가 서로를 갉아먹지 않았는가 (객체별로 따로 저장했으므로 원래 면적이 남아야 한다)

    돌려주는 값 : 문제 목록
    """
    label_dir = os.path.join(dataset_dir, split, 'labels')
    mask_dir = os.path.join(dataset_dir, split, 'masks')

    if not os.path.isdir(mask_dir):
        print(f'  {split:<6} : 변환된 마스크가 없습니다.')
        return []

    mask_files = [f for f in sorted(os.listdir(mask_dir)) if f.endswith('.npz')]

    if not mask_files:
        print(f'  {split:<6} : 변환된 마스크가 없습니다.')
        return []

    problems = []
    total_objects = 0
    total_size = 0

    for mask_file in tqdm(mask_files, desc=f'{split} 검증'):
        name = os.path.splitext(mask_file)[0]
        mask_path = os.path.join(mask_dir, mask_file)

        total_size += os.path.getsize(mask_path)
        masks, classes, boxes = load_mask_file(mask_path)
        total_objects += len(classes)

        polygons = read_label_polygons(os.path.join(label_dir, name + '.txt'))

        # 1) 객체 수 (너무 작아서 버린 것이 있으면 라벨보다 적을 수 있다)
        if len(classes) > len(polygons):
            problems.append(f'{name} : 마스크 {len(classes)}개 > 라벨 {len(polygons)}개')

        for i in range(len(classes)):
            ys, xs = np.where(masks[i] == 1)

            # 2) 박스가 마스크와 일치하는가
            real_box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

            if real_box != list(boxes[i]):
                problems.append(f'{name} : {i}번 박스가 마스크와 다름 {list(boxes[i])} vs {real_box}')

            # 4) 마스크가 비어 있지 않은가
            if len(xs) < MIN_AREA:
                problems.append(f'{name} : {i}번 마스크가 비어 있음')

        # 3) 클래스가 라벨 순서와 같은가 (버린 객체가 없을 때만 순서가 그대로다)
        if len(classes) == len(polygons):
            label_classes = [cls for cls, _ in polygons]

            if list(classes) != label_classes:
                problems.append(f'{name} : 클래스가 라벨과 다름 {list(classes)} vs {label_classes}')

    mb = total_size / (1024 * 1024)
    print(f'  {split:<6} : 마스크 파일 {len(mask_files)}개 / 객체 {total_objects}개 / 용량 {mb:.1f} MB')

    if problems:
        print(f'           문제 {len(problems)}건 (상위 5건)')
        for text in problems[:5]:
            print(f'             {text}')
    else:
        print('           문제 없음')

    if sample_count:
        save_sample(dataset_dir, split, mask_files, sample_count)

    return problems


def save_sample(dataset_dir, split, mask_files, sample_count=3):
    """변환된 마스크를 원본 이미지에 겹쳐 그려 저장한다. (눈으로 확인용)"""
    import random

    import matplotlib.pyplot as plt

    image_dir = os.path.join(dataset_dir, split, 'images')
    mask_dir = os.path.join(dataset_dir, split, 'masks')

    # 객체가 여러 개인(겹침이 있을 만한) 파일을 우선 고른다
    picked = random.sample(mask_files, min(sample_count, len(mask_files)))

    _, axes = plt.subplots(1, len(picked), figsize=(5 * len(picked), 5))

    if len(picked) == 1:
        axes = [axes]

    for ax, mask_file in zip(axes, picked):
        name = os.path.splitext(mask_file)[0]

        image = cv2.imread(find_image_path(image_dir, name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks, classes, boxes = load_mask_file(os.path.join(mask_dir, mask_file))

        colored = image.copy()

        for i in range(len(classes)):
            # 객체마다 다른 색으로 반투명하게 덮는다
            color = np.array([(int(i) * 70 + 60) % 256,
                              (int(i) * 130 + 20) % 256,
                              (int(i) * 190 + 140) % 256], dtype=float)
            area = masks[i] == 1
            colored[area] = (colored[area] * 0.5 + color * 0.5).astype(np.uint8)

            x1, y1, x2, y2 = boxes[i]
            cv2.rectangle(colored, (x1, y1), (x2, y2), (255, 255, 0), 2)

        ax.imshow(colored)
        ax.set_title(f'{name[:20]}... ({len(classes)} objects)', fontsize=8)
        ax.axis('off')

    save_path = os.path.join('./result', f'maskrcnn_convert_{split}.jpg')
    os.makedirs('./result', exist_ok=True)

    plt.suptitle(f'Mask R-CNN convert check ({split})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f'           확인용 그림 저장 : {save_path}')


def verify_dataset(dataset_dir, sample_count=3):
    """데이터셋 전체의 변환 결과를 확인한다."""
    print(f'===== 변환 결과 확인 : {dataset_dir} =====')

    problems = []

    for split in SPLITS:
        problems += verify_split(dataset_dir, split, sample_count=sample_count)

    return problems


if __name__ == '__main__':
    from Preprocessing.check_dataset import SEGMENT_DATASET

    convert_dataset(SEGMENT_DATASET)
    verify_dataset(SEGMENT_DATASET)
