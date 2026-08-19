"""
[2단계] 모델 학습

이 프로젝트는 세 가지 모델을 만들어 비교
    - train_detect()   : detection 라벨    + yolov8n.pt      -> runs/detect/vest_helmet_detect...
    - train_segment()  : segmentation 라벨 + yolov8n-seg.pt  -> runs/segment/vest_helmet_seg...
    - train_maskrcnn() : segmentation 라벨 + Mask R-CNN(torchvision)

개별 실행 : 위 함수를 직접 부름
일괄 실행 : train_all() 을 부르면 config/run_config.py 의 RUN_MODELS 에 있는 것만 학습함

--- 학습 결과가 저장되는 방식 ---
1) YOLO 는 실행할 때마다 runs/ 아래에 새 폴더를 만든다 (vest_helmet_detect, ...detect2, ...)
    각 폴더에 args.yaml(그때 쓴 설정)과 results.csv(epoch 별 지표)가 남으므로 실험 이력 저장
2) 학습이 끝나면 그 폴더의 best.pt 를 result/weights/ 로 복사
    평가·추론·비교는 항상 이 고정 경로를 보므로 폴더가 늘어나도 설정을 고칠 필요가 없음
3) 실행 조건과 결과를 result/train_log.csv 한 줄로 저장. (실험 노트)

실행 : python step2_train.py
"""

import csv
import os
import shutil
import time
from datetime import datetime

from ultralytics import YOLO

from config.run_config import CLASS_NAMES, CONFIRM_RUN, MODEL_NAMES, RUN_MODELS, WEIGHTS
from models.mask_rcnn import INPUT_SIZE, make_run_dir, train_model
from models.unet import INPUT_SIZE as UNET_INPUT_SIZE
from models.unet import train_model as unet_train
from utils.run_info import confirm_run

# ---- 공통 설정 ----
DETECT_YAML = './config/detect.yaml'
SEGMENT_YAML = './config/segment.yaml'

# Mask R-CNN 은 yaml 이 아니라 폴더 경로를 직접 쓴다 (segmentation 데이터가 들어오면 수정)
SEGMENT_DATASET = './Data/vest-helmet_crop_dedup_seg'
NUM_CLASSES = 2          # 배경을 뺀 클래스 수

# ---- YOLO 학습 설정 ----
EPOCHS = 100             # 최대 학습 횟수
# EPOCHS = 1               # test epochs
IMGSZ = 640              # 입력 이미지 크기
BATCH = 16               # 한 번에 학습할 이미지 수 (GPU 메모리 부족하면 8로 줄이기)
DEVICE = 0               # 0 -> GPU, 'cpu' -> CPU

PATIENCE = 15

DETECT_NAME = 'vest_helmet_detect'
SEGMENT_NAME = 'vest_helmet_seg'

# ---- Mask R-CNN 학습 설정 ----
# 사용안함
# MASKRCNN_EPOCHS = 30
MASKRCNN_EPOCHS = 1 # test epochs

# batch 4 + lr 0.005 = 이미지당 0.00125 로, torchvision 표준 레시피(batch 16 / lr 0.02)와 같다.
# 4GB GPU 에서 2.0GB 를 쓰므로 여유가 있다. (batch 를 더 키워도 속도는 거의 그대로다)
MASKRCNN_BATCH = 4
MASKRCNN_LR = 0.005
MASKRCNN_NAME = 'vest_helmet_maskrcnn'

MASKRCNN_PATIENCE = 7

# 가중치 저장 위치는 config/run_config.py 한 곳에서 관리한다
MASKRCNN_PATH = WEIGHTS['maskrcnn']

# ---- U-Net 학습 설정 ----
# ResNet34 인코더(ImageNet 사전학습)를 쓰는 시맨틱 분할 모델.
UNET_EPOCHS = 100
# UNET_EPOCHS = 1     # test epochs

UNET_BATCH = 6            # GPU Ram 8gb 기준
UNET_LR = 0.0015          # Adam 을 쓰므로 SGD 보다 작은 값을 쓴다
UNET_PATIENCE = 15
UNET_NAME = 'vest_helmet_unet'
UNET_PATH = WEIGHTS['unet']

# ---- 실험 이력 ----
TRAIN_LOG = './result/train_log.csv'
LOG_COLUMNS = ['날짜', '모델', '실행폴더', 'epochs', 'patience', 'imgsz', 'batch', '학습시간(분)', 'mAP50']


# results.csv 에서 best 지표 가져오기
def read_best_map(save_dir):
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

# 실행 조건과 결과 이력 저장(result/train_log.csv)
def save_train_log(key, save_dir, epochs, patience, minutes, best_map=''):
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


# 이력 마지막 값 총 학습 시간 읽기
def read_train_time(key):
    if not os.path.exists(TRAIN_LOG):
        return 0.0

    minutes = 0.0

    with open(TRAIN_LOG, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row.get('모델') == key and row.get('학습시간(분)'):
                minutes = float(row['학습시간(분)'])

    return minutes


# 최종 best.pt를 결과 폴더로 복사
def copy_best_weights(save_dir, key):
    src = os.path.join(save_dir, 'weights', 'best.pt')
    dst = WEIGHTS[key]

    if not os.path.exists(src):
        print(f'[주의] best.pt 를 찾지 못했습니다 : {src}')
        return None

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f'가중치 복사 : {src} -> {dst}')

    return dst


# yolo8n 모델 학습 
def train_yolo(key, base_model, data_yaml, run_name):
    # 실행 전에 어떤 설정으로 도는지 보여준다 (CONFIRM_RUN 이 True 면 폴더명 확인까지)
    settings = {
        '모델': f'{MODEL_NAMES[key]} ({base_model})',
        '데이터': data_yaml,
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'imgsz': IMGSZ,
        'batch': BATCH,
        'device': DEVICE,
        '가중치 저장': WEIGHTS[key],
    }

    ok, run_name = confirm_run(f'{MODEL_NAMES[key]} 학습', settings, run_name=run_name, confirm=CONFIRM_RUN)

    if not ok:
        print('학습을 취소했습니다.')
        return

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


# detect 모델 학습 진행
def train_detect():
    train_yolo('detect', 'yolov8n.pt', DETECT_YAML, DETECT_NAME)


# segment 모델 학습진행
def train_segment():
    train_yolo('segment', 'yolov8n-seg.pt', SEGMENT_YAML, SEGMENT_NAME)


# 사용안함
def train_maskrcnn():
    settings = {
        '모델': 'Mask R-CNN (torchvision)',
        '데이터': SEGMENT_DATASET,
        'epochs': MASKRCNN_EPOCHS,
        'patience': MASKRCNN_PATIENCE,
        'imgsz': f'{INPUT_SIZE} (YOLO 와 동일)',
        'batch': MASKRCNN_BATCH,
        'lr': f'{MASKRCNN_LR} (이미지당 {MASKRCNN_LR / MASKRCNN_BATCH:.5f})',
        '학습률 스케줄': 'CosineAnnealing (0 까지 매끄럽게 감소)',
        '증강': '좌우반전 + 색상변화 (train 만)',
        '가중치 저장': MASKRCNN_PATH,
    }

    ok, run_name = confirm_run('Mask R-CNN 학습', settings, run_name=MASKRCNN_NAME, confirm=CONFIRM_RUN)

    if not ok:
        print('학습을 취소했습니다.')
        return

    run_dir = make_run_dir('./runs/maskrcnn', run_name)
    print(f'학습 결과 폴더 : {run_dir}')

    start = time.time()

    result = train_model(dataset_dir=SEGMENT_DATASET,
                            num_classes=NUM_CLASSES,
                            save_path=MASKRCNN_PATH,
                            epochs=MASKRCNN_EPOCHS,
                            batch_size=MASKRCNN_BATCH,
                            lr=MASKRCNN_LR,
                            patience=MASKRCNN_PATIENCE,
                            run_dir=run_dir)

    minutes = (time.time() - start) / 60

    # mAP 칸에는 값이 없다 (Mask R-CNN 은 loss 로 판단하므로 mAP 는 step3_eval 에서 확인)
    save_train_log('maskrcnn', run_dir, result['epochs_ran'], MASKRCNN_PATIENCE, minutes)
    print(f'Mask R-CNN 학습 완료 -> {MASKRCNN_PATH}')


# Unet 학습 진행
def train_unet():
    settings = {
        '모델': 'U-Net (ResNet34 인코더, ImageNet 사전학습)',
        '데이터': f'{SEGMENT_DATASET} (semantic 폴더)',
        'epochs': UNET_EPOCHS,
        'patience': UNET_PATIENCE,
        'imgsz': f'{UNET_INPUT_SIZE} (YOLO 와 동일)',
        'batch': UNET_BATCH,
        'lr': f'{UNET_LR} (Adam)',
        '학습률 스케줄': 'CosineAnnealing (0 까지 매끄럽게 감소)',
        '증강': '좌우반전 + 색상변화 (train 만)',
        '클래스 가중치': '사용 (배경 83% / 헬멧 3% 불균형 보정)',
        '판단 기준': 'mIoU (높을수록 좋음)',
        '가중치 저장': UNET_PATH,
    }

    ok, run_name = confirm_run('U-Net 학습', settings, run_name=UNET_NAME, confirm=CONFIRM_RUN)

    if not ok:
        print('학습을 취소했습니다.')
        return

    run_dir = make_run_dir('./runs/unet', run_name)
    print(f'학습 결과 폴더 : {run_dir}')

    start = time.time()

    result = unet_train(dataset_dir=SEGMENT_DATASET,
                        num_classes=NUM_CLASSES,
                        save_path=UNET_PATH,
                        epochs=UNET_EPOCHS,
                        batch_size=UNET_BATCH,
                        lr=UNET_LR,
                        patience=UNET_PATIENCE,
                        input_size=UNET_INPUT_SIZE,
                        run_dir=run_dir,
                        class_names=CLASS_NAMES)

    minutes = (time.time() - start) / 60

    # mAP 대신 mIoU 를 기록한다 (박스가 없어 mAP 를 낼 수 없다)
    save_train_log('unet', run_dir, result['epochs_ran'], UNET_PATIENCE, minutes, round(result['best_score'], 4))
    print(f'U-Net 학습 완료 -> {UNET_PATH}')


# 모델 구분값과 학습 함수를 짝지어 둔다 (train_all 이 이걸 보고 골라 실행한다)
TRAIN_FUNCS = {
    'detect': train_detect,
    'segment': train_segment,
    'unet': train_unet,
    'maskrcnn': train_maskrcnn,
}

# config 설정에 따른 모델 학습 진행 
def train_all(run_models=RUN_MODELS):
    print(f'학습할 모델 : {run_models}')

    for key in run_models:
        if key not in TRAIN_FUNCS:
            print(f'[주의] 모르는 모델 이름입니다 : {key}')
            continue

        print(f'\n===== {MODEL_NAMES[key]} 학습 시작 =====')
        TRAIN_FUNCS[key]()


if __name__ == '__main__':
    train_all()
