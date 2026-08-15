"""
역할: 그림으로 확인하는 기능 모음

  1) show_label()         : 이미지 한 장에 정답 라벨을 그려서 눈으로 확인
  2) draw_label_samples() : 데이터셋에서 무작위로 몇 장을 뽑아 라벨을 그려 한 장으로 저장
                            (detection 은 박스, segmentation 은 폴리곤 선)
  3) draw_compare_plot()  : 모델들의 지표를 막대그래프로 비교
"""

import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.metrics import read_label_boxes, read_label_polygons

IMAGE_EXT = ('.jpg', '.jpeg', '.png')

# 클래스 번호별 색 (RGB). 클래스가 더 많으면 앞에서부터 다시 돌려 쓴다.
CLASS_COLORS = [(255, 0, 0), (0, 100, 255), (0, 180, 0), (255, 160, 0)]


def get_color(class_no):
    """클래스 번호에 맞는 색을 돌려준다."""
    return CLASS_COLORS[class_no % len(CLASS_COLORS)]


def put_class_name(image, name, x, y, color):
    """
    클래스 이름을 이미지에 쓴다.
    작은 이미지에서 글자가 밖으로 잘리지 않도록 글자 크기와 위치를 이미지 크기에 맞춰 조절한다.
    """
    image_h, image_w = image.shape[:2]

    font_scale = max(0.4, image_w / 1000)
    thickness = max(1, int(font_scale * 2))

    # 글자가 차지할 크기를 미리 구해서 이미지 안으로 밀어넣는다
    (text_w, text_h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = min(max(int(x), 0), max(image_w - text_w, 0))
    y = min(max(int(y) - 5, text_h), image_h - 2)

    cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    return image


def show_label(image_file, txt_file, task='detect', class_names=None):
    """이미지 한 장에 라벨을 그려 창으로 띄운다. (파일 하나만 따로 확인하고 싶을 때)"""
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if task == 'detect':
        image = draw_detect_label(image, txt_file, class_names)
    else:
        image = draw_segment_label(image, txt_file, class_names)

    plt.imshow(image)
    plt.axis('off')
    plt.show()


def draw_detect_label(image, txt_file, class_names=None):
    """이미지 위에 detection 라벨(박스)을 그린다."""
    image_h, image_w = image.shape[:2]

    for class_no, x1, y1, x2, y2 in read_label_boxes(txt_file, image_w, image_h):
        color = get_color(class_no)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        name = class_names[class_no] if class_names else str(class_no)
        put_class_name(image, name, x1, y1, color)

    return image


def draw_segment_label(image, txt_file, class_names=None):
    """이미지 위에 segmentation 라벨(폴리곤 외곽선)을 그린다."""
    image_h, image_w = image.shape[:2]

    for class_no, points in read_label_polygons(txt_file):
        color = get_color(class_no)

        # 0~1 정규화 좌표를 픽셀 좌표로 바꾸고 정수로 만든다
        pixel_points = np.array([[x * image_w, y * image_h] for x, y in points], dtype=np.int32)

        # isClosed=True -> 마지막 점과 첫 점을 이어 닫힌 선으로 그린다
        cv2.polylines(image, [pixel_points], isClosed=True, color=color, thickness=2)

        name = class_names[class_no] if class_names else str(class_no)
        x, y = pixel_points[0]
        put_class_name(image, name, x, y, color)

    return image


def draw_label_samples(dataset_dir, task, save_path, class_names=None,
                       sample_count=3, split='train', show=False):
    """
    데이터셋에서 이미지를 무작위로 몇 장 뽑아 라벨을 그린 뒤 한 장으로 붙여 저장한다.
    라벨링이 이미지에 제대로 얹혀 있는지 눈으로 확인하는 용도다.

    task        : 'detect' -> 박스로 그림 / 'segment' -> 폴리곤 선으로 그림
    save_path   : 저장할 파일 경로
    show        : True 면 창으로도 띄운다 (창을 닫아야 다음 코드가 실행된다)
    """
    image_dir = os.path.join(dataset_dir, split, 'images')
    label_dir = os.path.join(dataset_dir, split, 'labels')

    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
        print(f'  [주의] 샘플을 뽑을 폴더가 없습니다 : {image_dir}')
        return None

    # 라벨이 있는 이미지 중에서만 고른다
    label_names = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')}
    image_list = [f for f in os.listdir(image_dir)
                  if f.lower().endswith(IMAGE_EXT) and os.path.splitext(f)[0] in label_names]

    if not image_list:
        print('  [주의] 라벨이 있는 이미지가 없어 샘플을 만들지 못했습니다.')
        return None

    # 이미지가 3장보다 적으면 있는 만큼만 뽑는다
    picked = random.sample(image_list, min(sample_count, len(image_list)))

    _, axes = plt.subplots(1, len(picked), figsize=(5 * len(picked), 5))

    # 1장만 뽑히면 axes 가 리스트가 아니라서 리스트로 맞춰준다
    if len(picked) == 1:
        axes = [axes]

    for ax, file_name in zip(axes, picked):
        image = cv2.imread(os.path.join(image_dir, file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        txt_file = os.path.join(label_dir, os.path.splitext(file_name)[0] + '.txt')

        if task == 'detect':
            image = draw_detect_label(image, txt_file, class_names)
        else:
            image = draw_segment_label(image, txt_file, class_names)

        ax.imshow(image)
        ax.set_title(file_name, fontsize=9)
        ax.axis('off')

    plt.suptitle(f'{task} label sample ({split})')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f'  샘플 이미지 저장 : {save_path}')

    if show:
        plt.show()

    plt.close()

    return picked


# 그래프로 그릴 지표 (0~1 사이 값만. 속도나 용량은 단위가 달라서 표로만 본다)
PLOT_KEYS = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1', 'mIoU']


def draw_compare_plot(scores, save_path='./result/compare.jpg'):
    """
    여러 모델의 지표를 나란히 막대그래프로 그린다.

    scores 예시
      {'yolov8n(detect)': {'mAP50': 0.80, 'mIoU': 0.70, ...},
       'yolov8n-seg':     {'mAP50': 0.85, 'mIoU': 0.75, ...},
       'maskrcnn':        {'mAP50': 0.88, 'mIoU': 0.78, ...}}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model_names = list(scores.keys())

    # 모든 모델이 공통으로 가지고 있는 지표만 그린다
    keys = [k for k in PLOT_KEYS if all(k in scores[name] for name in model_names)]

    if not keys:
        print('그릴 수 있는 공통 지표가 없습니다.')
        return

    x = range(len(keys))

    # 모델 수에 맞춰 막대 너비와 위치를 정한다
    width = 0.8 / len(model_names)
    start = -0.4 + width / 2

    plt.figure(figsize=(10, 5))

    for i, name in enumerate(model_names):
        positions = [pos + start + (i * width) for pos in x]
        plt.bar(positions, [scores[name][k] for k in keys], width=width, label=name)

    plt.xticks(list(x), keys)
    plt.ylim(0, 1.0)
    plt.title('model comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f'그래프 저장 : {save_path}')
