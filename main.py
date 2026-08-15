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
  step1_check.py   : 데이터 검사 (라벨 형식, 이미지-라벨 짝, 개수, 샘플 그림)
  step2_train.py   : 학습
  step3_eval.py    : 평가
  step4_predict.py : 테스트 이미지 추론
  step5_compare.py : 세 모델 비교 (세 모델이 모두 학습돼 있어야 실행된다)

--- 어떤 모델을 돌릴지 ---
  config/run_config.py 의 RUN_MODELS 에서 정한다.
  step2~4 의 *_all() 함수가 모두 이 값을 본다.
    지금(segmentation 라벨 없음) : RUN_MODELS = ['detect']
    라벨이 들어온 뒤             : RUN_MODELS = ['detect', 'segment', 'maskrcnn']
"""

from config.run_config import RUN_MODELS
from Preprocessing.check_dataset import check_detect_dataset, check_segment_dataset
from step2_train import train_all, train_detect, train_maskrcnn, train_segment
from step3_eval import eval_all, eval_one
from step4_predict import predict_all, predict_one
from step5_compare import compare_models

if __name__ == '__main__':
    print(f'실행할 모델 : {RUN_MODELS}')

    # step1. 데이터셋 검사 (문제가 있으면 result/check_detect.csv 로 저장된다)
    check_detect_dataset()

    # segmentation 라벨이 들어오면 이것도 검사한다
    # check_segment_dataset()

    # step2. 학습 (RUN_MODELS 에 있는 모델만)
    # train_all()

    # step3. 평가
    # eval_all()

    # step4. 테스트 이미지 추론
    # predict_all()

    # step5. 세 모델 비교 -> result/compare.csv, result/compare.jpg
    # (세 모델이 모두 학습돼 있어야 실행된다)
    # compare_models()

    # --- 모델 하나만 따로 돌리고 싶을 때는 이렇게 부른다 ---
    # train_detect()
    # eval_one('detect')
    # predict_one('detect')
