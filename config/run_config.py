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

# 학습이 끝나면 가중치가 저장되는 위치 (학습·평가·추론이 모두 이 경로를 쓴다)
WEIGHTS = {
    'detect': './runs/detect/vest_helmet_detect/weights/best.pt',
    'segment': './runs/segment/vest_helmet_seg/weights/best.pt',
    'maskrcnn': './result/mask_rcnn.pth',
}

# ---- 여기를 고쳐서 실행할 모델을 정한다 ----
# segmentation 라벨이 아직 없으므로 지금은 detection 만 실행한다.
RUN_MODELS = ['detect']

# 세 모델을 한 번에 돌릴 때는 위 줄을 지우고 아래 줄의 주석을 푼다
# RUN_MODELS = ['detect', 'segment', 'maskrcnn']
