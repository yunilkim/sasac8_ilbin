# 역할: Preprocessing 패키지 표시 파일
# 학습 전에 라벨이 제대로 만들어졌는지 검사하는 코드를 모아둔다.
# - check_format.py  : 라벨 파일 내용이 모델(detection/segmentation) 형식에 맞는지 검사
# - check_sync.py    : 이미지 파일과 라벨 파일의 짝, 총 개수 확인
# - check_dataset.py : 위 두 검사를 실행하고 문제를 csv 로 저장 (여기를 실행하면 된다)
