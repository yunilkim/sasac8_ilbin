"""
역할: 이 프로젝트의 목표 - detection 모델과 segmentation 모델 비교

같은 이미지, 라벨만 다른 두 데이터셋으로 만든 두 모델을 같은 기준으로 비교한다.

  비교 항목
    - 박스 지표 : mAP50, mAP50-95, Precision, Recall, F1
    - 평균 IoU  : 정답 박스와 예측 박스가 얼마나 겹치는지 (utils/metrics.py)
    - 마스크 지표 : mask mAP (segmentation 모델만 나옴, 참고용)
    - 추론 속도 : 이미지 한 장당 ms, FPS
    - 학습 시간, 모델 파일 크기

결과는 result/compare.csv(표)와 result/compare.jpg(그래프)로 저장된다.

실행 : python compare.py
"""

import os

import pandas as pd
from ultralytics import YOLO

from eval import DETECT_WEIGHTS, DETECT_YAML, SEGMENT_WEIGHTS, SEGMENT_YAML, eval_model
from train import DETECT_NAME, SEGMENT_NAME, read_train_time
from utils.metrics import mean_iou
from utils.visualize import draw_compare_plot

# 각 모델의 테스트 이미지 폴더 (segmentation 폴더는 데이터가 들어오면 경로 수정)
DETECT_TEST_DIR = './Data/vest-helmet.v1i_roboflow/test/images'
SEGMENT_TEST_DIR = './Data/vest-helmet-seg/test/images'

SAVE_CSV = './result/compare.csv'


def get_model_size(weights):
    """모델 파일 크기를 MB 단위로 돌려준다."""
    if not os.path.exists(weights):
        return 0.0

    return os.path.getsize(weights) / (1024 * 1024)


def collect_scores(weights, data_yaml, test_dir, train_name):
    """모델 하나에 대한 비교 항목을 전부 모아 딕셔너리로 돌려준다."""
    # 1) mAP, Precision, Recall, 속도
    scores = eval_model(weights, data_yaml)

    # 2) 평균 IoU (예측 박스와 정답 박스의 겹침 정도)
    model = YOLO(weights)
    scores['mIoU'] = mean_iou(model, test_dir)

    # 3) 학습 시간(분)과 모델 크기(MB)
    scores['학습시간(분)'] = read_train_time(train_name) / 60
    scores['모델크기(MB)'] = get_model_size(weights)

    return scores


def compare_models():
    """두 모델을 평가해서 비교표와 그래프를 만든다."""
    # segmentation 모델을 아직 학습하지 않았으면 안내만 하고 끝낸다
    if not os.path.exists(SEGMENT_WEIGHTS):
        print('[안내] segmentation 모델이 아직 없습니다.')
        print('       segmentation 라벨 데이터셋을 Data/ 에 넣고')
        print('       config/segment.yaml 의 path 를 수정한 뒤')
        print('       train.py 의 train_segment() 를 실행하세요.')
        return None

    print('=== detection 모델 평가 ===')
    detect_scores = collect_scores(DETECT_WEIGHTS, DETECT_YAML, DETECT_TEST_DIR, DETECT_NAME)

    print('=== segmentation 모델 평가 ===')
    segment_scores = collect_scores(SEGMENT_WEIGHTS, SEGMENT_YAML, SEGMENT_TEST_DIR, SEGMENT_NAME)

    # 표로 정리 (detection 에는 마스크 지표가 없어서 빈칸이 생긴다)
    table = pd.DataFrame({
        'detection(yolov8n)': detect_scores,
        'segmentation(yolov8n-seg)': segment_scores,
    })
    table = table.round(4)

    print(table)

    os.makedirs(os.path.dirname(SAVE_CSV), exist_ok=True)
    table.to_csv(SAVE_CSV, encoding='utf-8-sig')
    print(f'비교표 저장 : {SAVE_CSV}')

    # 0~1 지표만 막대그래프로 그린다
    draw_compare_plot({'detect': detect_scores, 'segment': segment_scores})

    return table


if __name__ == '__main__':
    compare_models()
