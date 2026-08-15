"""
역할: 어떤 모델을 실행할지 한 곳에서 정하는 설정 파일

step2_train, step3_eval, step4_predict 가 모두 이 파일의 RUN_MODELS 를 본다.
아래 한 줄만 고치면 학습·평가·추론이 모두 같은 모델만 돌아간다.

  - 지금처럼 detection 만 있을 때        -> RUN_MODELS = ['detect']
  - segmentation 라벨이 들어온 뒤 전부   -> RUN_MODELS = ['detect', 'segment', 'maskrcnn']

특정 모델 하나만 따로 돌리고 싶으면 함수를 직접 부르면 된다.
  예) step2_train.train_detect()  /  step4_predict.predict_maskrcnn()
"""

# 이 프로젝트에서 쓰는 모델 구분값 (순서 = 비교표에 나오는 순서)
MODEL_KEYS = ['detect', 'segment', 'maskrcnn']

# 화면과 비교표에 보여줄 이름
MODEL_NAMES = {
    'detect': 'yolov8n(detect)',
    'segment': 'yolov8n-seg',
    'maskrcnn': 'maskrcnn',
}

# '지금 쓰는 모델' 가중치 위치. 평가·추론·비교는 항상 여기를 본다.
#
# YOLO 학습 결과는 runs/detect/vest_helmet_detect, ...detect2, ...detect3 처럼
# 실행할 때마다 새 폴더에 쌓인다(이력 보존). 학습이 끝나면 그 폴더의 best.pt 를
# 아래 경로로 복사하기 때문에, 폴더 이름이 늘어나도 이 파일을 고칠 필요가 없다.
WEIGHTS = {
    'detect': './result/weights/detect_best.pt',
    'segment': './result/weights/segment_best.pt',
    'maskrcnn': './result/weights/mask_rcnn.pth',
}

# ---- 여기를 고쳐서 실행할 모델을 정한다 ----
# segmentation 라벨이 아직 없으므로 지금은 detection 만 실행한다.
RUN_MODELS = ['detect']

# 세 모델을 한 번에 돌릴 때는 위 줄을 지우고 아래 줄의 주석을 푼다
# RUN_MODELS = ['detect', 'segment', 'maskrcnn']
