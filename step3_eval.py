"""
[3단계] 학습한 모델의 성능 지표

test 셋으로 mAP50, mAP50-95, Precision, Recall, 추론 속도를 구함
    - YOLO 모델      : ultralytics 가 지표를 계산 (eval_model)
    - Mask R-CNN     : ultralytics 를 못 쓰므로 torchmetrics 와 직접 만든 IoU 로 계산 (eval_maskrcnn)
(PR Curve, Confusion Matrix 그림은 ultralytics 가 runs/.../val 폴더에 자동 저장)

개별 실행 : eval_one('detect') 처럼 모델 하나만 평가
일괄 실행 : eval_all() 을 부르면 config/run_config.py 의 RUN_MODELS 에 있는 것만 평가

평가에는 시간이 걸리므로 결과를 result/eval_scores.csv 에 저장
(나중에 수치를 다시 볼 때 평가를 또 돌리지 않음)

실행 : python step3_eval.py
"""

import os

from ultralytics import YOLO

from config.run_config import MODEL_NAMES, RUN_MODELS, WEIGHTS
from models.mask_rcnn import evaluate_model, load_trained_model
from models.unet import evaluate_model as unet_evaluate
from models.unet import load_trained_model as unet_load
from step2_train import (MASKRCNN_PATH, NUM_CLASSES, SEGMENT_DATASET, UNET_INPUT_SIZE, UNET_PATH)
from utils.run_info import print_settings
from utils.table import save_table_csv

# ---- 평가 설정 (가중치 경로는 config/run_config.py 에서 가져온다) ----
DETECT_WEIGHTS = WEIGHTS['detect']
SEGMENT_WEIGHTS = WEIGHTS['segment']

DETECT_YAML = './config/detect.yaml'
SEGMENT_YAML = './config/segment.yaml'

IMGSZ = 640
SPLIT = 'test'      # train / val / test

# 평가 결과를 저장할 파일
SAVE_CSV = './result/eval_scores.csv'

# yolo8n 모델 평가(detect, segment)
def eval_model(weights, data_yaml, split=SPLIT):
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, split=split, imgsz=IMGSZ)

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    scores = {
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'Precision': precision,
        'Recall': recall,
    }

    # F1 = 2 * (정밀도 * 재현율) / (정밀도 + 재현율)
    if precision + recall > 0:
        scores['F1'] = 2 * precision * recall / (precision + recall)
    else:
        scores['F1'] = 0.0

    # 이미지 한 장 추론에 걸린 시간(ms) -> FPS 로도 바꿔본다
    infer_ms = float(metrics.speed['inference'])
    scores['추론시간(ms)'] = infer_ms
    scores['FPS'] = 1000 / infer_ms if infer_ms > 0 else 0.0

    # segmentation 모델일 때만 마스크 지표가 있다
    if hasattr(metrics, 'seg'):
        scores['mask_mAP50'] = float(metrics.seg.map50)
        scores['mask_mAP50-95'] = float(metrics.seg.map)

    return scores

# 사용 안함
def eval_maskrcnn(weights=MASKRCNN_PATH, dataset_dir=SEGMENT_DATASET, split=SPLIT):
    model = load_trained_model(weights, NUM_CLASSES)
    scores = evaluate_model(model, dataset_dir, split=split)

    return scores


# Unet 모델 평가(픽셀 기준 지표(mIoU, Dice, 정확도) 사용)
def eval_unet(weights=UNET_PATH, dataset_dir=SEGMENT_DATASET, split=SPLIT):
    model = unet_load(weights, NUM_CLASSES)
    scores = unet_evaluate(model, dataset_dir, NUM_CLASSES, split=split, input_size=UNET_INPUT_SIZE)

    return scores


# 지표 출력
def print_scores(title, scores):
    """지표 딕셔너리를 보기 좋게 출력한다."""
    print(f'--- {title} ---')

    for key, value in scores.items():
        print(f'{key:15s} : {value:.4f}')


# 모델 하나의 지표 반환
def eval_one(key):
    # 어떤 가중치를 어떤 데이터로 평가하는지 먼저 보여준다
    print_settings(f'{MODEL_NAMES[key]} 평가', {
        '가중치': WEIGHTS[key],
        '데이터': DETECT_YAML if key == 'detect' else (
            SEGMENT_YAML if key == 'segment' else SEGMENT_DATASET),
        'split': SPLIT,
        'imgsz': IMGSZ,
        '지표': '픽셀 기준 (mIoU/Dice)' if key == 'unet' else '박스 기준 (mAP)',
    })

    if key == 'detect':
        return eval_model(DETECT_WEIGHTS, DETECT_YAML)

    if key == 'segment':
        return eval_model(SEGMENT_WEIGHTS, SEGMENT_YAML)

    if key == 'unet':
        return eval_unet()

    return eval_maskrcnn()


# RUN_MODELS 설정에 맞는 모델 검증 전체 실행
def eval_all(run_models=RUN_MODELS, save_csv=SAVE_CSV):
    """
    반환값 : {모델키: 지표 딕셔너리}
    """
    print(f'평가할 모델 : {run_models}')

    results = {}

    for key in run_models:
        if key not in WEIGHTS:
            print(f'[주의] 모르는 모델 이름입니다 : {key}')
            continue

        if not os.path.exists(WEIGHTS[key]):
            print(f'[안내] {MODEL_NAMES[key]} 가중치가 없습니다. 먼저 학습하세요 : {WEIGHTS[key]}')
            continue

        print(f'\n===== {MODEL_NAMES[key]} 평가 시작 =====')
        scores = eval_one(key)
        print_scores(MODEL_NAMES[key], scores)

        results[key] = scores

    # 표로 저장할 때는 보기 좋은 모델 이름을 쓴다
    if results and save_csv:
        save_table_csv({MODEL_NAMES[key]: value for key, value in results.items()}, save_csv)

    return results


if __name__ == '__main__':
    eval_all()
