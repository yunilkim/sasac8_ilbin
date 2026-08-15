"""
역할: 학습 전에 데이터셋이 제대로 준비됐는지 확인한다.

데이터셋은 이미 Data/ 폴더에 들어있으므로 다운로드는 하지 않는다.
  - split(train/valid/test) 별 이미지, 라벨 개수 세기
  - 데이터셋 원본 data.yaml 의 클래스 이름 출력 (config/detect.yaml 과 맞추기 위함)
  - 라벨이 detection 인지 segmentation 인지 구분해서 알려주기
"""

import os

import yaml

# 데이터셋 폴더 (segmentation 데이터셋이 들어오면 이 경로만 바꿔서 다시 확인하면 된다)
DATASET_DIR = './Data/vest-helmet.v1i_roboflow'
SPLITS = ['train', 'valid', 'test']
IMAGE_EXT = ('.jpg', '.jpeg', '.png')


def print_class_names(dataset_dir=DATASET_DIR):
    """데이터셋 원본 data.yaml 에 적힌 클래스 이름을 출력한다."""
    yaml_path = os.path.join(dataset_dir, 'data.yaml')

    if not os.path.exists(yaml_path):
        print(f'[-] data.yaml 이 없습니다 : {yaml_path}')
        return None

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    names = data['names']
    print(f'클래스 : {names}')
    print('-> config/detect.yaml 의 names 와 순서까지 같은지 꼭 확인하세요.')

    return names


def check_label_type(label_dir):
    """
    라벨 폴더 안의 첫 번째 txt 를 열어 detection 인지 segmentation 인지 구분한다.
    한 줄의 숫자가 5개면 박스(detection), 그보다 많으면 폴리곤(segmentation).
    """
    txt_list = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    if not txt_list:
        return '라벨 없음'

    with open(os.path.join(label_dir, txt_list[0]), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) == 5:
                return 'detection'
            elif len(parts) > 5:
                return 'segmentation'

    return '알 수 없음'


def check_dataset(dataset_dir=DATASET_DIR):
    """split 별 이미지/라벨 개수와 라벨 종류를 출력한다."""
    print(f'=== 데이터셋 확인 : {dataset_dir} ===')
    print_class_names(dataset_dir)

    for split in SPLITS:
        image_dir = os.path.join(dataset_dir, split, 'images')
        label_dir = os.path.join(dataset_dir, split, 'labels')

        if not os.path.isdir(image_dir):
            print(f'{split} 폴더가 없습니다.')
            continue

        images = [f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXT)]
        labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

        label_type = check_label_type(label_dir)
        print(f'{split:5s} : 이미지 {len(images)}개 / 라벨 {len(labels)}개 ({label_type})')

        # 이미지와 라벨 개수가 다르면 학습 중 경고가 나므로 미리 알려준다
        if len(images) != len(labels):
            print(f'    [주의] 이미지와 라벨 개수가 다릅니다.')


if __name__ == '__main__':
    check_dataset()
