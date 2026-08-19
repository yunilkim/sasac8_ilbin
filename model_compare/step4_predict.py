"""
[4단계] 테스트 이미지로 결과를 확인

폴더 안의 이미지를 한꺼번에 추론해서
    - 결과 이미지(박스 또는 마스크가 그려진 그림)를 result/ 아래에 저장하고
    - 이미지마다 어떤 안전장구가 몇 개 검출됐는지 출력

YOLO 는 predict_images() 사용

개별 실행 : predict_one('detect') 또는 위 함수를 직접 호출
일괄 실행 : predict_all() 을 부르면 config/run_config.py 의 RUN_MODELS 에 있는 것만 추론

실행 : python step4_predict.py
"""

import os

from ultralytics import YOLO

from config.run_config import CLASS_NAMES, MODEL_NAMES, RUN_MODELS, WEIGHTS
from models.mask_rcnn import load_trained_model
from models.mask_rcnn import predict_images as maskrcnn_predict
from models.unet import load_trained_model as unet_load
from models.unet import predict_images as unet_predict
from step2_train import (MASKRCNN_PATH, NUM_CLASSES, SEGMENT_DATASET, UNET_INPUT_SIZE, UNET_PATH)
from utils.run_info import print_settings

# ---- 추론 설정 (가중치 경로는 config/run_config.py 에서 가져온다) ----
DETECT_WEIGHTS = WEIGHTS['detect']
SEGMENT_WEIGHTS = WEIGHTS['segment']

DETECT_TEST_DIR = './Data/vest-helmet_crop_dedup/test/images'

# segmentation 데이터셋 경로는 step2_train.py 한 곳에서만 관리한다
SEGMENT_TEST_DIR = os.path.join(SEGMENT_DATASET, 'test', 'images')

CONF = 0.45         # 이 값보다 확신이 낮은 검출은 버린다
IMGSZ = 640
SAVE_DIR = './result'


# yolo8n 모델들 검출 결과 클래스 갯수 카운트
def count_classes(result):
    counts = {}

    for box in result.boxes:
        name = result.names[int(box.cls[0])]
        counts[name] = counts.get(name, 0) + 1

    return counts


# 이미지 폴더 예측하고 결과 저장
def predict_images(weights, source=DETECT_TEST_DIR, save_name='predict'):
    """
    save_name : 결과가 저장될 폴더 이름 (result/<save_name>)
    """
    model = YOLO(weights)

    results = model.predict(source=source,
                            conf=CONF,
                            imgsz=IMGSZ,
                            save=True,
                            project=SAVE_DIR,
                            name=save_name,
                            exist_ok=True,
                            verbose=False)

    for result in results:
        counts = count_classes(result)

        # 파일 경로가 길어서 파일 이름만 잘라서 보여준다
        file_name = os.path.basename(result.path)

        if counts:
            print(f'{file_name} : {counts}')
        else:
            print(f'{file_name} : 검출된 안전장구 없음')

    print(f'결과 이미지 저장 : {SAVE_DIR}/{save_name}')

    return results


# 사용안함
def predict_maskrcnn(source=SEGMENT_TEST_DIR, save_name='predict_maskrcnn'):
    model = load_trained_model(MASKRCNN_PATH, NUM_CLASSES)
    maskrcnn_predict(model, source, os.path.join(SAVE_DIR, save_name), CLASS_NAMES)


# Unet 에측
def predict_unet(source=SEGMENT_TEST_DIR, save_name='predict_unet'):
    """
    U-Net 으로 이미지 폴더를 추론하고 결과를 저장한다.
    박스가 없으므로 클래스별로 색을 칠한 그림만 나온다.
    """
    model = unet_load(UNET_PATH, NUM_CLASSES)
    unet_predict(model, source, os.path.join(SAVE_DIR, save_name), CLASS_NAMES, input_size=UNET_INPUT_SIZE)


# 모델 하나 테스트 이미지로 예측
def predict_one(key):
    source = DETECT_TEST_DIR if key == 'detect' else SEGMENT_TEST_DIR

    print_settings(f'{MODEL_NAMES[key]} 추론', {
        '가중치': WEIGHTS[key],
        '입력 이미지': source,
        'conf': CONF,
        'imgsz': IMGSZ,
        '결과 저장': f'{SAVE_DIR}/predict_{key}',
    })

    if key == 'detect':
        return predict_images(DETECT_WEIGHTS, DETECT_TEST_DIR, save_name='predict_detect')

    if key == 'segment':
        return predict_images(SEGMENT_WEIGHTS, SEGMENT_TEST_DIR, save_name='predict_seg')

    if key == 'unet':
        return predict_unet()

    return predict_maskrcnn()


# RUN_MODELS 에 설정된 모델 모두 예측 진행
def predict_all(run_models=RUN_MODELS):
    print(f'추론할 모델 : {run_models}')

    for key in run_models:
        if key not in WEIGHTS:
            print(f'[주의] 모르는 모델 이름입니다 : {key}')
            continue

        if not os.path.exists(WEIGHTS[key]):
            print(f'[안내] {MODEL_NAMES[key]} 가중치가 없습니다. 먼저 학습하세요 : {WEIGHTS[key]}')
            continue

        print(f'\n===== {MODEL_NAMES[key]} 추론 시작 =====')
        predict_one(key)


if __name__ == '__main__':
    predict_all()
