"""
역할: 라벨 검사 실행 파일 (Preprocessing 폴더의 진입점)

학습을 시작하기 전에 이 파일을 실행해서 데이터셋에 문제가 없는지 확인한다.
  1. 이미지-라벨 동기화 검사 (check_sync.py)
  2. 라벨 형식 검사 (check_format.py)
  3. 문제가 있으면 result/ 아래에 csv 로 저장
  4. 무작위로 3장을 뽑아 라벨을 그려보고 result/ 에 저장 (눈으로 확인)

숫자만 봐서는 라벨이 엉뚱한 위치에 찍혔는지 알 수 없기 때문에
마지막에 샘플 이미지를 그려본다. detection 은 박스, segmentation 은 폴리곤 선으로 그린다.

문제가 있어도 프로그램을 멈추지는 않는다. csv 를 열어서 확인한 뒤
라벨을 고칠지 그냥 진행할지는 직접 판단하면 된다.

실행 : python -m Preprocessing.check_dataset
"""

import csv
import os

import yaml

from Preprocessing.check_duplicate import check_duplicate
from Preprocessing.check_format import check_format
from Preprocessing.check_sync import check_sync
from utils.visualize import draw_label_samples

# ---- 검사할 데이터셋 경로 ----
# detection : split 간 중복을 제거한 데이터셋
DETECT_DATASET = './Data/vest-helmet_crop_dedup'

# segmentation : 같은 이미지에 SAM 으로 폴리곤 라벨을 붙인 데이터셋
SEGMENT_DATASET = './Data/vest-helmet_crop_dedup_seg'

# 원본(중복 제거 전) : 중복 검사 비교용으로만 쓴다
ORIGIN_DATASET = './Data/vest-helmet.v1i_roboflow_origin'

SAVE_DIR = './result'

# csv 에 저장할 열 이름
CSV_COLUMNS = ['split', 'image_file', 'label_file', 'problem_type', 'detail']

# ---- 샘플 라벨 확인 설정 ----
SAMPLE_COUNT = 3        # 무작위로 뽑아 그려볼 이미지 수
SAMPLE_SPLIT = 'train'  # 어느 폴더에서 뽑을지
SHOW_SAMPLE = False     # True 로 바꾸면 저장과 함께 창으로도 띄운다 (창을 닫아야 다음이 진행됨)


# 데이터셋 data.yaml 의 클래스 이름을 출력
def print_class_names(dataset_dir):
    yaml_path = os.path.join(dataset_dir, 'data.yaml')

    if not os.path.exists(yaml_path):
        return None

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    print(f'클래스 : {data["names"]}')
    print('-> config 폴더의 yaml 파일 names 와 순서까지 같은지 확인하세요.')

    return data['names']


# fail list 저장
def save_problem_csv(problems, save_name):
    if not problems:
        return None

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, save_name)

    with open(save_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(problems)

    print(f'문제 목록 저장 : {save_path}')

    return save_path


# fail list 유형별로 출력
def count_problem_type(problems):
    counts = {}

    for problem in problems:
        problem_type = problem['problem_type']
        counts[problem_type] = counts.get(problem_type, 0) + 1

    for problem_type, count in counts.items():
        print(f'  - {problem_type} : {count}건')

    return counts


# 무작위 샘플 이미지 작성 함수
def check_label_samples(dataset_dir, task, sample_name, class_names, show=SHOW_SAMPLE):
    print('[샘플 라벨 확인]')

    save_path = os.path.join(SAVE_DIR, sample_name)

    picked = draw_label_samples(dataset_dir=dataset_dir,
                                task=task,
                                save_path=save_path,
                                class_names=class_names,
                                sample_count=SAMPLE_COUNT,
                                split=SAMPLE_SPLIT,
                                show=show)

    if picked:
        print(f'  뽑힌 이미지 : {picked}')

    return picked


# 데이터셋 하나 검사
def run_check(dataset_dir, task, save_name, sample_name, show=SHOW_SAMPLE):
    """
    dataset_dir : 데이터셋 폴더
    task        : 'detect' 또는 'segment' (라벨이 어떤 형식이어야 하는지)
    save_name   : 문제 목록 csv 파일 이름
    sample_name : 샘플 라벨 그림 파일 이름
    show        : True 면 샘플 그림을 창으로도 띄운다
    """
    print(f'===== 데이터셋 검사 : {dataset_dir} ({task}) =====')

    # 폴더가 없으면 (예: segmentation 라벨이 아직 안 들어온 경우) 안내만 하고 끝낸다
    if not os.path.isdir(dataset_dir):
        print('데이터셋 폴더가 없습니다. 폴더가 준비되면 경로를 확인하고 다시 실행하세요.')
        return []

    class_names = print_class_names(dataset_dir)

    sync_problems, counts = check_sync(dataset_dir)
    format_problems = check_format(dataset_dir, task)

    problems = sync_problems + format_problems

    print(f'[검사 결과] 문제 {len(problems)}건')

    if problems:
        count_problem_type(problems)
        save_problem_csv(problems, save_name)
    else:
        print('  문제 없음. 학습을 진행해도 됩니다.')

    # 마지막으로 라벨이 이미지에 제대로 얹혔는지 눈으로 확인
    check_label_samples(dataset_dir, task, sample_name, class_names, show=show)

    return problems

# detection 데이터셋 검사 (라벨을 박스로 그려 확인)
def check_detect_dataset(show=SHOW_SAMPLE):
    return run_check(DETECT_DATASET, 'detect',
                        save_name='check_detect.csv',
                        sample_name='label_sample_detect.jpg',
                        show=show)

# segmentation 데이터셋 검사 (라벨을 폴리곤 선으로 그려 확인)
def check_segment_dataset(show=SHOW_SAMPLE):
    return run_check(SEGMENT_DATASET, 'segment',
                        save_name='check_segment.csv',
                        sample_name='label_sample_segment.jpg',
                        show=show)


# 원본 데이터셋의 중복 확인 및 목록 저장
def check_origin_duplicate():
    return check_duplicate(ORIGIN_DATASET, save_path=os.path.join(SAVE_DIR, 'duplicate_origin.csv'))


# 사용하는 데이터셋의 split 간 중복 확인 및 목록 저장
def check_current_duplicate():
    return check_duplicate(DETECT_DATASET, save_path=os.path.join(SAVE_DIR, 'duplicate_current.csv'))


if __name__ == '__main__':
    check_detect_dataset()

    # segmentation 라벨 데이터셋이 준비되면 아래 주석을 푼다
    # check_segment_dataset()

    # split 간 중복 검사 (이미지를 전부 읽어서 몇 분 걸린다)
    # check_current_duplicate()   # 지금 쓰는 데이터셋에 중복이 남았는지
    # check_origin_duplicate()    # 원본에 중복이 얼마나 있었는지 (비교용)
