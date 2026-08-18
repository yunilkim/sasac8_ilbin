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

import os

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from config.run_config import MODEL_KEYS, MODEL_NAMES, WEIGHTS
from step2_train import (MASKRCNN_PATH, NUM_CLASSES, SEGMENT_DATASET, UNET_PATH,
                         read_train_time)
from step3_eval import (DETECT_WEIGHTS, DETECT_YAML, SEGMENT_WEIGHTS, SEGMENT_YAML,
                        SPLIT, eval_maskrcnn, eval_model, eval_unet)
from utils.metrics import mean_iou
from utils.table import print_table, save_table_csv
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


def collect_yolo_scores(key, weights, data_yaml, test_dir):
    """YOLO 모델 하나의 비교 항목을 모아 딕셔너리로 돌려준다."""
    # 1) mAP, Precision, Recall, 속도
    scores = eval_model(weights, data_yaml)

    # 2) 평균 IoU (예측 박스와 정답 박스의 겹침 정도)
    model = YOLO(weights)
    scores['mIoU'] = mean_iou(model, test_dir)

    # 3) 학습 시간(분)과 모델 크기(MB)  ※ 학습 시간은 result/train_log.csv 에서 읽는다
    scores['학습시간(분)'] = read_train_time(key)
    scores['모델크기(MB)'] = get_model_size(weights)

    return scores


def collect_maskrcnn_scores():
    """Mask R-CNN 의 비교 항목을 모아 딕셔너리로 돌려준다."""
    # mAP, Precision, Recall, mIoU, 속도가 한 번에 나온다
    scores = eval_maskrcnn()

    scores['학습시간(분)'] = read_train_time('maskrcnn')
    scores['모델크기(MB)'] = get_model_size(MASKRCNN_PATH)

    return scores


def collect_unet_scores():
    """U-Net 의 비교 항목을 모아 딕셔너리로 돌려준다. (픽셀 기준 지표)"""
    scores = eval_unet()

    scores['학습시간(분)'] = read_train_time('unet')
    scores['모델크기(MB)'] = get_model_size(UNET_PATH)

    return scores


def yolo_pixel_scores(weights, dataset_dir, num_classes, split='test', conf=0.45):
    """
    yolov8n-seg 의 예측을 '픽셀 지도' 로 눌러서 U-Net 과 같은 기준으로 지표를 낸다.

    yolov8n-seg 는 객체마다 마스크를 주고 U-Net 은 지도 한 장을 준다.
    그대로는 비교할 수 없으므로, 객체 마스크들을 한 장으로 합쳐서 맞춰 준다.
    (면적이 큰 것부터 그려 작은 객체가 살아남게 한다 - 정답을 만들 때와 같은 규칙)
    """
    import cv2
    from PIL import Image

    model = YOLO(weights)

    image_dir = os.path.join(dataset_dir, split, 'images')
    semantic_dir = os.path.join(dataset_dir, split, 'semantic')

    if not os.path.isdir(semantic_dir):
        print('[안내] 정답 지도(semantic)가 없어 픽셀 지표를 건너뜁니다.')
        return {}

    inter = np.zeros(num_classes + 1)
    union = np.zeros(num_classes + 1)

    files = [f for f in sorted(os.listdir(semantic_dir)) if f.endswith('.png')]

    for file_name in tqdm(files, desc='픽셀 지표'):
        name = os.path.splitext(file_name)[0]

        image_path = os.path.join(image_dir, name + '.jpg')
        if not os.path.exists(image_path):
            continue

        truth = np.array(Image.open(os.path.join(semantic_dir, file_name)))
        predict = np.zeros_like(truth)

        result = model.predict(image_path, conf=conf, verbose=False)[0]

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            # 면적이 큰 것부터 그린다
            areas = masks.reshape(len(masks), -1).sum(axis=1)

            for i in np.argsort(-areas):
                m = cv2.resize(masks[i], (truth.shape[1], truth.shape[0]),
                               interpolation=cv2.INTER_NEAREST) > 0.5
                predict[m] = classes[i] + 1

        for c in range(num_classes + 1):
            p = predict == c
            t = truth == c
            inter[c] += float((p & t).sum())
            union[c] += float((p | t).sum())

    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)

    return {'pixel_mIoU': float(np.nanmean(iou[1:]))}


def find_missing_models():
    """
    세 모델 중 아직 학습되지 않은(가중치 파일이 없는) 모델 목록을 돌려준다.
    비어 있으면 전부 준비된 것이다.
    """
    return [key for key in MODEL_KEYS if not os.path.exists(WEIGHTS[key])]


def collect_scores(key):
    """모델 하나(key)의 비교 항목을 모아 돌려준다."""
    if key == 'detect':
        return collect_yolo_scores(key, DETECT_WEIGHTS, DETECT_YAML, DETECT_TEST_DIR)

    if key == 'segment':
        scores = collect_yolo_scores(key, SEGMENT_WEIGHTS, SEGMENT_YAML, SEGMENT_TEST_DIR)

        # U-Net 과 나란히 놓으려면 픽셀 기준 지표도 필요하다
        scores.update(yolo_pixel_scores(SEGMENT_WEIGHTS, SEGMENT_DATASET,
                                        NUM_CLASSES, split=SPLIT))
        return scores

    if key == 'unet':
        return collect_unet_scores()

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
    save_table_csv(scores, SAVE_CSV)

    # 지표 성격이 달라 어떤 칸이 왜 비는지 알려준다
    print('  * mAP 계열   : 박스를 내는 모델만 (yolov8n, yolov8n-seg)')
    print('  * pixel 계열 : 픽셀 지도로 비교 가능한 모델만 (yolov8n-seg, U-Net)')
    print('  -> yolov8n-seg 가 양쪽에 모두 있어 두 축을 이어주는 기준이 된다')

    # 0~1 지표만 막대그래프로 그린다
    draw_compare_plot(scores)

    return scores


if __name__ == '__main__':
    compare_models()
