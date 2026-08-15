"""
역할: 전체 실행 순서를 모아둔 파일

필요한 단계의 주석을 풀고 python main.py 로 실행한다.
(각 파일을 따로 실행해도 된다 : python train.py, python eval.py ...)

--- 프로젝트 목표 ---
같은 이미지에 라벨만 다르게 붙인 두 데이터셋으로 모델을 만들고 비교한다.
  1) detection 라벨(박스)      -> yolov8n
  2) segmentation 라벨(폴리곤) -> yolov8n-seg   (라벨 데이터 준비 중)
  3) 두 모델의 성능/속도를 비교

--- 진행 순서 ---
  1. 데이터 확인   : utils/dataset.py   (이미 Data/ 에 데이터셋이 있으므로 다운로드는 없음)
  2. 학습         : train.py
  3. 평가         : eval.py
  4. 테스트 이미지 추론 : predict.py
  5. 두 모델 비교  : compare.py
"""

from compare import compare_models
from eval import DETECT_WEIGHTS, DETECT_YAML, eval_model, print_scores
from predict import predict_images
from train import train_detect, train_segment
from utils.dataset import check_dataset

if __name__ == '__main__':
    # 1. 데이터셋 확인 (이미지/라벨 개수, 클래스 이름)
    check_dataset()

    # 2. detection 모델 학습
    # train_detect()

    # 3. 평가
    # detect_scores = eval_model(DETECT_WEIGHTS, DETECT_YAML)
    # print_scores('detection (yolov8n)', detect_scores)

    # 4. 테스트 이미지 추론 결과 확인
    # predict_images(DETECT_WEIGHTS, save_name='predict_detect')

    # --- 여기서부터는 segmentation 라벨 데이터셋이 준비된 뒤에 실행 ---
    # 5. segmentation 모델 학습 (config/segment.yaml 의 path 먼저 수정)
    # train_segment()

    # 6. 두 모델 비교 -> result/compare.csv, result/compare.jpg
    # compare_models()
