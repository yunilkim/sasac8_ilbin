"""
역할: [2단계] 모델 학습

이 프로젝트는 세 가지 모델을 만들어 비교한다.
  - train_detect()   : detection 라벨    + yolov8n.pt      -> runs/detect/vest_helmet_detect
  - train_segment()  : segmentation 라벨 + yolov8n-seg.pt  -> runs/segment/vest_helmet_seg
  - train_maskrcnn() : segmentation 라벨 + Mask R-CNN(torchvision) -> result/mask_rcnn.pth

  개별 실행 : 위 함수를 직접 부른다
  일괄 실행 : train_all() 을 부르면 config/run_config.py 의 RUN_MODELS 에 있는 것만 학습한다

나중에 비교표에 쓰려고 학습에 걸린 시간을 result/train_time.txt 에 적어둔다.

실행 : python step2_train.py
"""

import os
import time

from ultralytics import YOLO

from config.run_config import MODEL_NAMES, RUN_MODELS, WEIGHTS
from models.mask_rcnn import train_model

# ---- 공통 설정 ----
DETECT_YAML = './config/detect.yaml'
SEGMENT_YAML = './config/segment.yaml'

# Mask R-CNN 은 yaml 이 아니라 폴더 경로를 직접 쓴다 (segmentation 데이터가 들어오면 수정)
SEGMENT_DATASET = './Data/vest-helmet-seg'
NUM_CLASSES = 2          # 배경을 뺀 클래스 수 (helmet, vest)

# ---- YOLO 학습 설정 ----
EPOCHS = 50              # 학습 횟수
IMGSZ = 640              # 입력 이미지 크기
BATCH = 16               # 한 번에 학습할 이미지 수 (GPU 메모리 부족하면 8로 줄이기)
DEVICE = 0               # 0 -> GPU, 'cpu' -> CPU

DETECT_NAME = 'vest_helmet_detect'
SEGMENT_NAME = 'vest_helmet_seg'

# ---- Mask R-CNN 학습 설정 ----
# ResNet50 기반이라 yolov8n 보다 무겁다. epoch 과 batch 를 작게 잡는다.
MASKRCNN_EPOCHS = 10
MASKRCNN_BATCH = 2
MASKRCNN_LR = 0.005
MASKRCNN_NAME = 'mask_rcnn'

# 가중치 저장 위치는 config/run_config.py 한 곳에서 관리한다
MASKRCNN_PATH = WEIGHTS['maskrcnn']

TIME_LOG = './result/train_time.txt'


def save_train_time(name, seconds):
    """학습에 걸린 시간을 초 단위로 파일에 기록한다. (step5_compare.py 에서 읽어 쓴다)"""
    os.makedirs(os.path.dirname(TIME_LOG), exist_ok=True)

    with open(TIME_LOG, 'a', encoding='utf-8') as f:
        f.write(f'{name},{seconds:.1f}\n')

    print(f'학습 시간 : {seconds / 60:.1f}분')


def read_train_time(name):
    """기록해둔 학습 시간(초)을 읽는다. 없으면 0 을 돌려준다."""
    if not os.path.exists(TIME_LOG):
        return 0.0

    seconds = 0.0
    with open(TIME_LOG, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')

            # 같은 이름으로 여러 번 학습했으면 마지막 값을 쓴다
            if len(parts) == 2 and parts[0] == name:
                seconds = float(parts[1])

    return seconds


def train_detect():
    """detection 라벨로 yolov8n 을 학습한다."""
    start = time.time()

    model = YOLO('yolov8n.pt')
    model.train(data=DETECT_YAML,
                epochs=EPOCHS,
                imgsz=IMGSZ,
                batch=BATCH,
                device=DEVICE,
                name=DETECT_NAME,
                exist_ok=True,
                plots=True)

    save_train_time(DETECT_NAME, time.time() - start)
    print(f'detection 학습 완료 -> runs/detect/{DETECT_NAME}/weights/best.pt')


def train_segment():
    """segmentation 라벨로 yolov8n-seg 를 학습한다. (라벨 데이터셋이 준비된 뒤 실행)"""
    start = time.time()

    model = YOLO('yolov8n-seg.pt')
    model.train(data=SEGMENT_YAML,
                epochs=EPOCHS,
                imgsz=IMGSZ,
                batch=BATCH,
                device=DEVICE,
                name=SEGMENT_NAME,
                exist_ok=True,
                plots=True)

    save_train_time(SEGMENT_NAME, time.time() - start)
    print(f'segmentation 학습 완료 -> runs/segment/{SEGMENT_NAME}/weights/best.pt')


def train_maskrcnn():
    """segmentation 라벨로 torchvision Mask R-CNN 을 학습한다."""
    start = time.time()

    os.makedirs(os.path.dirname(MASKRCNN_PATH), exist_ok=True)

    train_model(dataset_dir=SEGMENT_DATASET,
                num_classes=NUM_CLASSES,
                save_path=MASKRCNN_PATH,
                epochs=MASKRCNN_EPOCHS,
                batch_size=MASKRCNN_BATCH,
                lr=MASKRCNN_LR)

    save_train_time(MASKRCNN_NAME, time.time() - start)
    print(f'Mask R-CNN 학습 완료 -> {MASKRCNN_PATH}')


# 모델 구분값과 학습 함수를 짝지어 둔다 (train_all 이 이걸 보고 골라 실행한다)
TRAIN_FUNCS = {
    'detect': train_detect,
    'segment': train_segment,
    'maskrcnn': train_maskrcnn,
}


def train_all(run_models=RUN_MODELS):
    """
    RUN_MODELS 에 적힌 모델을 순서대로 학습한다.
    실행할 모델은 config/run_config.py 에서 정한다.
    """
    print(f'학습할 모델 : {run_models}')

    for key in run_models:
        if key not in TRAIN_FUNCS:
            print(f'[주의] 모르는 모델 이름입니다 : {key}')
            continue

        print(f'\n===== {MODEL_NAMES[key]} 학습 시작 =====')
        TRAIN_FUNCS[key]()


if __name__ == '__main__':
    train_all()
