"""
역할: 전체 실행 순서를 모아둔 파일

필요한 단계의 주석을 풀고 python main.py 로 실행한다.
(각 단계 파일을 따로 실행해도 된다 : python step2_train.py, python step3_eval.py ...)

--- 프로젝트 목표 ---
같은 이미지에 라벨만 다르게 붙인 데이터셋으로 세 모델을 만들어 비교한다.
  1) detection 라벨(박스)      -> yolov8n
  2) segmentation 라벨(폴리곤) -> yolov8n-seg
  3) segmentation 라벨(폴리곤) -> Mask R-CNN (torchvision)
  4) 세 모델의 성능/속도를 한 표에서 비교

--- 진행 순서 (파일 이름 앞의 step 번호가 실행 순서다) ---
  step1_check.py   : 데이터 검사 (라벨 형식, 이미지-라벨 짝, 개수)
  step2_train.py   : 학습
  step3_eval.py    : 평가
  step4_predict.py : 테스트 이미지 추론
  step5_compare.py : 세 모델 비교
"""

from Preprocessing.check_dataset import check_detect_dataset, check_segment_dataset
from step2_train import train_detect, train_maskrcnn, train_segment
from step3_eval import DETECT_WEIGHTS, DETECT_YAML, eval_model, print_scores
from step4_predict import predict_images, predict_maskrcnn
from step5_compare import compare_models

if __name__ == '__main__':
    # step1. 데이터셋 검사 (문제가 있으면 result/check_detect.csv 로 저장된다)
    check_detect_dataset()

    # step2. detection 모델 학습
    # train_detect()

    # step3. 평가
    # detect_scores = eval_model(DETECT_WEIGHTS, DETECT_YAML)
    # print_scores('detection (yolov8n)', detect_scores)

    # step4. 테스트 이미지 추론 결과 확인
    # predict_images(DETECT_WEIGHTS, save_name='predict_detect')

    # --- 여기서부터는 segmentation 라벨 데이터셋이 준비된 뒤에 실행 ---
    # step1. segmentation 라벨 검사 (Preprocessing/check_dataset.py 의 경로 먼저 수정)
    # check_segment_dataset()

    # step2. segmentation 모델 두 개 학습 (config/segment.yaml, step2_train.py 의 경로도 수정)
    # train_segment()
    # train_maskrcnn()

    # step4. Mask R-CNN 추론 결과 확인
    # predict_maskrcnn()

    # step5. 세 모델 비교 -> result/compare.csv, result/compare.jpg
    # compare_models()
