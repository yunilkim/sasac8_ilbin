# 역할: models 패키지 표시 파일
# YOLO 는 ultralytics 라이브러리가 다 해주지만, torch 계열 모델은 직접 만들어야 한다.
# - seg_dataset.py : YOLO 폴리곤 라벨을 torch 학습용 데이터(마스크, 박스)로 바꿔주는 Dataset
# - mask_rcnn.py   : torchvision Mask R-CNN 모델 생성 / 학습 / 평가
