"""
[1단계] 학습 전 데이터셋 검사

실제 검사 코드는 Preprocessing 폴더에 들어있고, 이 파일은 그것을 불러다 실행
  - 이미지와 라벨 txt 의 짝이 맞는지, split 별/전체 개수가 몇 개인지
  - 라벨 내용이 모델 형식(detection / segmentation)에 맞게 만들어졌는지
  - 무작위 3장에 라벨을 그려서 제대로 얹혔는지 눈으로 확인

문제가 있으면 result/check_detect.csv (또는 check_segment.csv) 로 저장된다.
샘플 그림은 result/label_sample_detect.jpg (또는 label_sample_segment.jpg) 로 저장된다.
문제가 있어도 프로그램이 멈추지는 않으니, csv 와 그림을 보고 고칠지 판단하면 된다.

실행 : python step1_check.py
"""

from Preprocessing.check_dataset import check_detect_dataset, check_segment_dataset

# 샘플 이미지 출력 설정(True 시 출력된 이미지를 닫아야 이후 코드 실행됨)
SHOW_SAMPLE = False

if __name__ == '__main__':
    check_detect_dataset(show=SHOW_SAMPLE)

    # segmentation 라벨 데이터셋이 준비되면 아래 주석을 푼다
    # (Preprocessing/check_dataset.py 의 SEGMENT_DATASET 경로를 먼저 수정)
    # check_segment_dataset(show=SHOW_SAMPLE)
