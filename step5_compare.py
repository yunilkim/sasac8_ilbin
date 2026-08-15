"""
역할: [5단계] 이 프로젝트의 목표 - 세 모델을 같은 기준으로 비교

  1) yolov8n      : detection 라벨(박스)로 학습
  2) yolov8n-seg  : segmentation 라벨(폴리곤)로 학습
  3) Mask R-CNN   : 같은 segmentation 라벨을 쓰는 torch 계열 모델

  비교 항목
    - 박스 지표 : mAP50, mAP50-95, Precision, Recall, F1
    - 평균 IoU  : 정답 박스와 예측 박스가 얼마나 겹치는지
    - 마스크 지표 : mask mAP (yolov8n-seg 만 나옴, 참고용)
    - 추론 속도 : 이미지 한 장당 ms, FPS
    - 학습 시간, 모델 파일 크기

비교는 세 모델이 모두 학습된 뒤에만 실행된다.
하나라도 가중치가 없으면 무엇이 빠졌는지 알려주고 표를 만들지 않는다.
결과는 result/compare.csv(표)와 result/compare.jpg(그래프)로 저장된다.

실행 : python step5_compare.py
"""

import csv
import os

from ultralytics import YOLO

from config.run_config import MODEL_KEYS, MODEL_NAMES, WEIGHTS
from step2_train import (DETECT_NAME, MASKRCNN_NAME, MASKRCNN_PATH,
                         SEGMENT_DATASET, SEGMENT_NAME, read_train_time)
from step3_eval import (DETECT_WEIGHTS, DETECT_YAML, SEGMENT_WEIGHTS,
                        SEGMENT_YAML, eval_maskrcnn, eval_model)
from utils.metrics import mean_iou
from utils.visualize import draw_compare_plot

# 각 모델의 테스트 이미지 폴더
DETECT_TEST_DIR = './Data/vest-helmet.v1i_roboflow/test/images'

# segmentation 데이터셋 경로는 step2_train.py 한 곳에서만 관리한다
SEGMENT_TEST_DIR = os.path.join(SEGMENT_DATASET, 'test', 'images')

SAVE_CSV = './result/compare.csv'


def get_model_size(weights):
    """모델 파일 크기를 MB 단위로 돌려준다."""
    if not os.path.exists(weights):
        return 0.0

    return os.path.getsize(weights) / (1024 * 1024)


def collect_yolo_scores(weights, data_yaml, test_dir, train_name):
    """YOLO 모델 하나의 비교 항목을 모아 딕셔너리로 돌려준다."""
    # 1) mAP, Precision, Recall, 속도
    scores = eval_model(weights, data_yaml)

    # 2) 평균 IoU (예측 박스와 정답 박스의 겹침 정도)
    model = YOLO(weights)
    scores['mIoU'] = mean_iou(model, test_dir)

    # 3) 학습 시간(분)과 모델 크기(MB)
    scores['학습시간(분)'] = read_train_time(train_name) / 60
    scores['모델크기(MB)'] = get_model_size(weights)

    return scores


def collect_maskrcnn_scores():
    """Mask R-CNN 의 비교 항목을 모아 딕셔너리로 돌려준다."""
    # mAP, Precision, Recall, mIoU, 속도가 한 번에 나온다
    scores = eval_maskrcnn()

    scores['학습시간(분)'] = read_train_time(MASKRCNN_NAME) / 60
    scores['모델크기(MB)'] = get_model_size(MASKRCNN_PATH)

    return scores


def get_all_keys(scores):
    """
    모델마다 나오는 지표가 조금씩 달라서(예: mask mAP) 표에 쓸 항목을 모아 정리한다.
    먼저 나온 순서를 그대로 유지한다.
    """
    keys = []

    for model_scores in scores.values():
        for key in model_scores:
            if key not in keys:
                keys.append(key)

    return keys


def print_table(scores):
    """비교표를 화면에 출력한다. 없는 항목은 - 로 표시한다."""
    names = list(scores.keys())
    keys = get_all_keys(scores)

    print()
    print(f'{"항목":<16}' + ''.join(f'{name:>18}' for name in names))

    for key in keys:
        row = f'{key:<16}'

        for name in names:
            value = scores[name].get(key)
            row += f'{value:>18.4f}' if value is not None else f'{"-":>18}'

        print(row)

    print()


def save_table_csv(scores, save_path=SAVE_CSV):
    """비교표를 csv 로 저장한다. (엑셀에서 열 수 있도록 utf-8-sig)"""
    names = list(scores.keys())
    keys = get_all_keys(scores)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['항목'] + names)

        for key in keys:
            row = [key]

            for name in names:
                value = scores[name].get(key)
                row.append(round(value, 4) if value is not None else '')

            writer.writerow(row)

    print(f'비교표 저장 : {save_path}')


def find_missing_models():
    """
    세 모델 중 아직 학습되지 않은(가중치 파일이 없는) 모델 목록을 돌려준다.
    비어 있으면 전부 준비된 것이다.
    """
    return [key for key in MODEL_KEYS if not os.path.exists(WEIGHTS[key])]


def collect_scores(key):
    """모델 하나(key)의 비교 항목을 모아 돌려준다."""
    if key == 'detect':
        return collect_yolo_scores(DETECT_WEIGHTS, DETECT_YAML, DETECT_TEST_DIR, DETECT_NAME)

    if key == 'segment':
        return collect_yolo_scores(SEGMENT_WEIGHTS, SEGMENT_YAML, SEGMENT_TEST_DIR, SEGMENT_NAME)

    return collect_maskrcnn_scores()


def compare_models():
    """
    세 모델을 모두 평가해서 비교표와 그래프를 만든다.
    하나라도 학습이 안 되어 있으면 비교하지 않고 무엇이 빠졌는지 알려준다.
    """
    missing = find_missing_models()

    if missing:
        print('[안내] 아직 준비되지 않은 모델이 있어 비교를 실행하지 않습니다.')

        for key in missing:
            print(f'  - {MODEL_NAMES[key]} : 가중치 없음 ({WEIGHTS[key]})')

        print('  config/run_config.py 의 RUN_MODELS 에 세 모델을 모두 넣고')
        print('  step2_train.py 를 실행해 학습을 끝낸 뒤 다시 시도하세요.')
        return None

    scores = {}

    for key in MODEL_KEYS:
        print(f'=== {MODEL_NAMES[key]} 평가 ===')
        scores[MODEL_NAMES[key]] = collect_scores(key)

    print_table(scores)
    save_table_csv(scores)

    # 0~1 지표만 막대그래프로 그린다
    draw_compare_plot(scores)

    return scores


if __name__ == '__main__':
    compare_models()
