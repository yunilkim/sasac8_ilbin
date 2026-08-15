"""
역할: [2단계] 모델 학습

이 프로젝트는 세 가지 모델을 만들어 비교한다.
  - train_detect()   : detection 라벨    + yolov8n.pt      -> runs/detect/vest_helmet_detect...
  - train_segment()  : segmentation 라벨 + yolov8n-seg.pt  -> runs/segment/vest_helmet_seg...
  - train_maskrcnn() : segmentation 라벨 + Mask R-CNN(torchvision)

  개별 실행 : 위 함수를 직접 부른다
  일괄 실행 : train_all() 을 부르면 config/run_config.py 의 RUN_MODELS 에 있는 것만 학습한다

--- 학습 결과가 저장되는 방식 ---
1) YOLO 는 실행할 때마다 runs/ 아래에 새 폴더를 만든다 (vest_helmet_detect, ...detect2, ...)
   각 폴더에 args.yaml(그때 쓴 설정)과 results.csv(epoch 별 지표)가 남으므로 실험 이력이 된다.
2) 학습이 끝나면 그 폴더의 best.pt 를 result/weights/ 로 복사한다.
   평가·추론·비교는 항상 이 고정 경로를 보므로 폴더가 늘어나도 설정을 고칠 필요가 없다.
3) 실행 조건과 결과를 result/train_log.csv 한 줄로 남긴다. (실험 노트)

실행 : python step2_train.py
"""

import csv
import os
import shutil
import time
from datetime import datetime

from ultralytics import YOLO

from config.run_config import MODEL_NAMES, RUN_MODELS, WEIGHTS
from models.mask_rcnn import train_model

# ---- 공통 설정 ----
DETECT_YAML = './config/detect.yaml'
SEGMENT_YAML = './config/segment.yaml'

# Mask R-CNN 은 yaml 이 아니라 폴더 경로를 직접 쓴다 (segmentation 데이터가 들어오면 수정)
SEGMENT_DATASET = './Data/vest-helmet-seg'
NUM_CLASSES = 2          # 배경을 뺀 클래스 수

# ---- YOLO 학습 설정 ----
EPOCHS = 100             # 최대 학습 횟수
IMGSZ = 640              # 입력 이미지 크기
BATCH = 16               # 한 번에 학습할 이미지 수 (GPU 메모리 부족하면 8로 줄이기)
DEVICE = 0               # 0 -> GPU, 'cpu' -> CPU

# Early Stopping : 성능이 PATIENCE 번 연속 나아지지 않으면 남은 epoch 을 건너뛰고 멈춘다.
# best.pt 는 항상 가장 좋았던 epoch 의 가중치이므로 중간에 멈춰도 손해가 없다.
PATIENCE = 15

DETECT_NAME = 'vest_helmet_detect'
SEGMENT_NAME = 'vest_helmet_seg'

# ---- Mask R-CNN 학습 설정 ----
# ResNet50 기반이라 yolov8n 보다 무겁다. batch 를 작게 잡는다.
# 세 모델을 비교할 때는 학습 조건을 맞춰야 하므로, segmentation 라벨이 들어오면
# 실제 걸리는 시간을 보고 epoch 을 정한다. (지금 값은 임시)
MASKRCNN_EPOCHS = 30
MASKRCNN_BATCH = 2
MASKRCNN_LR = 0.005

# 가중치 저장 위치는 config/run_config.py 한 곳에서 관리한다
MASKRCNN_PATH = WEIGHTS['maskrcnn']

# ---- 실험 이력 ----
TRAIN_LOG = './result/train_log.csv'
LOG_COLUMNS = ['날짜', '모델', '실행폴더', 'epochs', 'patience', 'imgsz', 'batch',
               '학습시간(분)', 'mAP50']


def read_best_map(save_dir):
    """
    학습 폴더의 results.csv 에서 가장 좋았던 mAP50 을 읽는다.
    (best.pt 가 최고 성능 epoch 의 가중치이므로 최댓값을 쓴다)
    파일이 없으면 빈 문자열을 돌려준다. (Mask R-CNN 은 results.csv 가 없다)
    """
    csv_path = os.path.join(save_dir, 'results.csv')

    if not os.path.exists(csv_path):
        return ''

    best = 0.0

    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            # 열 이름 앞뒤에 공백이 섞여 있을 수 있어 정리한다
            row = {key.strip(): value for key, value in row.items() if key}
            value = row.get('metrics/mAP50(B)')

            if value:
                best = max(best, float(value))

    return round(best, 4)


def save_train_log(key, save_dir, epochs, patience, minutes, best_map=''):
    """실행 조건과 결과를 result/train_log.csv 에 한 줄 추가한다."""
    os.makedirs(os.path.dirname(TRAIN_LOG), exist_ok=True)

    # 파일이 처음 만들어질 때만 열 이름을 쓴다
    is_new = not os.path.exists(TRAIN_LOG)

    with open(TRAIN_LOG, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)

        if is_new:
            writer.writerow(LOG_COLUMNS)

        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M'),
            key,
            save_dir,
            epochs,
            patience,
            IMGSZ,
            BATCH,
            round(minutes, 1),
            best_map,
        ])

    print(f'학습 시간 : {minutes:.1f}분   (이력 기록 : {TRAIN_LOG})')


def read_train_time(key):
    """
    기록해둔 학습 시간(분)을 읽는다. 같은 모델을 여러 번 학습했으면 마지막 값을 쓴다.
    기록이 없으면 0 을 돌려준다.
    """
    if not os.path.exists(TRAIN_LOG):
        return 0.0

    minutes = 0.0

    with open(TRAIN_LOG, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row.get('모델') == key and row.get('학습시간(분)'):
                minutes = float(row['학습시간(분)'])

    return minutes


def copy_best_weights(save_dir, key):
    """
    학습 폴더의 best.pt 를 '지금 쓰는 모델' 위치로 복사한다.
    학습 폴더는 이력으로 그대로 남고, 평가·추론은 복사본을 쓴다.
    """
    src = os.path.join(save_dir, 'weights', 'best.pt')
    dst = WEIGHTS[key]

    if not os.path.exists(src):
        print(f'[주의] best.pt 를 찾지 못했습니다 : {src}')
        return None

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f'가중치 복사 : {src} -> {dst}')

    return dst


def train_yolo(key, base_model, data_yaml, run_name):
    """
    YOLO 계열(detect / segment) 학습을 공통으로 처리한다.
    exist_ok 를 주지 않으므로 같은 이름이 있으면 뒤에 번호가 붙은 새 폴더가 만들어진다.
    """
    start = time.time()

    model = YOLO(base_model)
    model.train(data=data_yaml,
                epochs=EPOCHS,
                patience=PATIENCE,
                imgsz=IMGSZ,
                batch=BATCH,
                device=DEVICE,
                name=run_name,
                plots=True)

    minutes = (time.time() - start) / 60

    # 이번 학습 결과가 실제로 저장된 폴더 (번호가 붙었을 수 있다)
    save_dir = str(model.trainer.save_dir)
    print(f'학습 결과 폴더 : {save_dir}')

    copy_best_weights(save_dir, key)
    save_train_log(key, save_dir, EPOCHS, PATIENCE, minutes, read_best_map(save_dir))


def train_detect():
    """detection 라벨로 yolov8n 을 학습한다."""
    train_yolo('detect', 'yolov8n.pt', DETECT_YAML, DETECT_NAME)


def train_segment():
    """segmentation 라벨로 yolov8n-seg 를 학습한다. (라벨 데이터셋이 준비된 뒤 실행)"""
    train_yolo('segment', 'yolov8n-seg.pt', SEGMENT_YAML, SEGMENT_NAME)


def train_maskrcnn():
    """
    segmentation 라벨로 torchvision Mask R-CNN 을 학습한다.
    가중치 파일이 170MB 정도로 커서 이력을 쌓지 않고 한 개만 덮어쓴다.
    (조건과 결과는 train_log.csv 에 남는다)
    """
    start = time.time()

    os.makedirs(os.path.dirname(MASKRCNN_PATH), exist_ok=True)

    train_model(dataset_dir=SEGMENT_DATASET,
                num_classes=NUM_CLASSES,
                save_path=MASKRCNN_PATH,
                epochs=MASKRCNN_EPOCHS,
                batch_size=MASKRCNN_BATCH,
                lr=MASKRCNN_LR)

    minutes = (time.time() - start) / 60

    # Mask R-CNN 은 results.csv 가 없어 mAP 칸은 비워둔다 (step3_eval 결과로 확인)
    save_train_log('maskrcnn', MASKRCNN_PATH, MASKRCNN_EPOCHS, '-', minutes)
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
