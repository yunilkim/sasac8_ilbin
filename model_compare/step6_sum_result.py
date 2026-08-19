"""
[6단계] 원본과 세 모델의 추론 결과를 한 장으로 합침

step4_predict.py 가 모델마다 따로 저장한 결과 이미지를 나란히 놓아
같은 사진에서 세 모델이 어떻게 다르게 보는지 눈으로 비교
수치(step5)로는 안 보이는 차이 - 어떤 객체를 놓쳤는지, 경계가 어떻게 다른지가 보임

원본        : Data/.../test/images
detect      : runs/detect/result/predict_detect      (박스)
segment     : runs/segment/result/predict_seg        (박스 + 마스크)
unet        : result/predict_unet                    (마스크만)

결과는 result/sum_result/ 에 원본과 같은 이름으로 저장

실행 : python step6_sum_result.py
"""

import os

import cv2
import numpy as np
from tqdm import tqdm

from config.run_config import MODEL_NAMES
from utils.run_info import print_settings

# ---- 입력 폴더 ----
ORIGIN_DIR = './Data/vest-helmet_crop_dedup_seg/test/images'

# 모델 구분값 -> 추론 결과 폴더
# YOLO 결과가 runs/ 아래에 있는 이유 : ultralytics 가 project 인자를 runs/<task>/ 밑으로 넣는다
PREDICT_DIRS = {
    'detect': './runs/detect/result/predict_detect',
    'segment': './runs/segment/result/predict_seg',
    'unet': './result/predict_unet',
}

SAVE_DIR = './result/sum_result'

# ---- 만들 이미지 설정 ----
CELL_SIZE = 640          # 칸 하나의 크기 (원본이 640 이라 그대로 쓴다)
COLUMNS = 2              # 한 줄에 넣을 칸 수. 2 -> 2x2 격자 / 4 -> 가로 한 줄
TITLE_HEIGHT = 44        # 칸 위에 붙는 제목 띠의 높이
BORDER = 4               # 칸 사이 여백

# 몇 장을 만들지. 0 이면 전부 만든다.
MAX_COUNT = 0

IMAGE_EXT = ('.jpg', '.jpeg', '.png')

# 제목 띠 / 글자 색 (BGR)
TITLE_BG = (245, 245, 245)
TITLE_COLOR = (30, 30, 30)


# 이미지 크기를 읽고 제목 붙임
def read_cell(image_path, title):
    image = cv2.imread(image_path)

    if image is None:
        return None

    # 칸 크기를 통일해야 나란히 붙일 수 있다
    if image.shape[0] != CELL_SIZE or image.shape[1] != CELL_SIZE:
        image = cv2.resize(image, (CELL_SIZE, CELL_SIZE))

    # 제목 띠를 만들어 이미지 위에 얹는다
    band = np.full((TITLE_HEIGHT, CELL_SIZE, 3), TITLE_BG, dtype=np.uint8)

    # 글자는 영문만 쓴다
    cv2.putText(band, title, (12, TITLE_HEIGHT - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TITLE_COLOR, 2)

    return np.vstack([band, image])


# 여백 추가(경계를 위해)
def add_border(cell):
    return cv2.copyMakeBorder(cell, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT, value=TITLE_BG)


# 정리된 갯수로 격자 만듬
def make_grid(cells):
    cells = [add_border(cell) for cell in cells]

    # 마지막 줄이 모자라면 빈 칸으로 채운다 (안 채우면 붙일 때 크기가 안 맞는다)
    blank = np.full_like(cells[0], TITLE_BG, dtype=np.uint8)

    while len(cells) % COLUMNS != 0:
        cells.append(blank)

    rows = [np.hstack(cells[i:i + COLUMNS]) for i in range(0, len(cells), COLUMNS)]

    return np.vstack(rows)


# 이미지 합침
def make_one(file_name):
    cells = []

    # 원본을 맨 앞에 놓는다
    cell = read_cell(os.path.join(ORIGIN_DIR, file_name), 'origin')

    if cell is None:
        return None

    cells.append(cell)

    # 모델 결과를 PREDICT_DIRS 에 적힌 순서대로 붙인다
    for key, predict_dir in PREDICT_DIRS.items():
        cell = read_cell(os.path.join(predict_dir, file_name), MODEL_NAMES[key])

        if cell is None:
            return None

        cells.append(cell)

    return make_grid(cells)


# 파일 수집
def find_common_files():
    if not os.path.isdir(ORIGIN_DIR):
        print(f'[주의] 원본 폴더가 없습니다 : {ORIGIN_DIR}')
        return []

    names = {f for f in os.listdir(ORIGIN_DIR) if f.lower().endswith(IMAGE_EXT)}

    for key, predict_dir in PREDICT_DIRS.items():
        if not os.path.isdir(predict_dir):
            print(f'[주의] {MODEL_NAMES[key]} 결과 폴더가 없습니다 : {predict_dir}')
            print('       step4_predict.py 를 먼저 실행하세요.')
            return []

        # 교집합만 남긴다
        names &= set(os.listdir(predict_dir))

    return sorted(names)


# 전체 파일 대상 이미지 합침
def make_all(max_count=MAX_COUNT):
    files = find_common_files()

    if not files:
        print('합칠 이미지가 없습니다.')
        return 0

    total = len(files)

    if max_count > 0:
        files = files[:max_count]

    print_settings('결과 이미지 합치기', {
        '원본': ORIGIN_DIR,
        '모델 결과': ' / '.join(PREDICT_DIRS.values()),
        '배치': f'{COLUMNS} 칸씩 ({len(PREDICT_DIRS) + 1}장)',
        '칸 크기': CELL_SIZE,
        '만들 장수': f'{len(files)}장 / 짝이 맞는 전체 {total}장',
        '저장 위치': SAVE_DIR,
    })

    os.makedirs(SAVE_DIR, exist_ok=True)

    saved = 0

    for file_name in tqdm(files, desc='합치는 중'):
        merged = make_one(file_name)

        if merged is None:
            print(f'  [건너뜀] 읽지 못한 이미지가 있습니다 : {file_name}')
            continue

        cv2.imwrite(os.path.join(SAVE_DIR, file_name), merged)
        saved += 1

    print(f'합친 이미지 {saved}장 저장 : {SAVE_DIR}')

    return saved


if __name__ == '__main__':
    make_all()
