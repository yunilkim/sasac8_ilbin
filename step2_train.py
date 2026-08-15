"""
역할: YOLOv8n 학습

이 프로젝트는 같은 이미지에 대해 두 가지 라벨로 모델을 만들고 비교한다.
  - train_detect()  : detection 라벨  + yolov8n.pt      -> runs/detect/vest_helmet_detect
  - train_segment() : segmentation 라벨 + yolov8n-seg.pt -> runs/segment/vest_helmet_seg

나중에 비교표에 쓰려고 학습에 걸린 시간을 result/train_time.txt 에 적어둔다.

실행 : python train.py
"""

import os
import time

from ultralytics import YOLO

# ---- 학습 설정 (필요하면 이 값만 바꾸면 된다) ----
DETECT_YAML = './config/detect.yaml'
SEGMENT_YAML = './config/segment.yaml'

EPOCHS = 50          # 학습 횟수
IMGSZ = 640          # 입력 이미지 크기
BATCH = 16           # 한 번에 학습할 이미지 수 (GPU 메모리 부족하면 8로 줄이기)
DEVICE = 0           # 0 -> GPU, 'cpu' -> CPU

DETECT_NAME = 'vest_helmet_detect'
SEGMENT_NAME = 'vest_helmet_seg'

TIME_LOG = './result/train_time.txt'


def save_train_time(name, seconds):
    """학습에 걸린 시간을 초 단위로 파일에 기록한다. (compare.py 에서 읽어 쓴다)"""
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


if __name__ == '__main__':
    train_detect()

    # segmentation 데이터셋이 준비되면 아래 주석을 푼다
    # train_segment()
