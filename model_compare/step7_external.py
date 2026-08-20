"""
[7단계] 외부 테스트 데이터로 세 모델 추론 + 결과 합치기

step4·step6 은 이 프로젝트 데이터셋의 test 폴더를 대상으로 돈다.
이 파일은 그 흐름과 무관하게, 밖에서 가져온 이미지에 학습된 모델을 그대로 적용한다.
학습에 쓰지 않은 사진에서도 통하는지 눈으로 보는 용도다.

라벨이 없어도 된다. 추론은 이미지만 읽는다. (라벨이 필요한 건 평가인 step3 뿐이다)

앞 단계에서 함수만 빌려 오고 폴더는 전부 이 파일에서 정한다.
    step6_sum_result : 격자 그리기 (read_cell / add_border / make_grid)
    models.unet      : U-Net 불러오기와 추론
    step4_predict    : 검출 개수 세기

실행 : python step7_external.py
"""

import os

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from config.run_config import CLASS_NAMES, MODEL_NAMES, WEIGHTS
from models.unet import load_trained_model as unet_load
from models.unet import predict_images as unet_predict
from step2_train import NUM_CLASSES, UNET_INPUT_SIZE
from step4_predict import count_classes
from step6_sum_result import IMAGE_EXT, add_border, make_grid, read_cell
from utils.run_info import print_settings

# ---- 여기만 고치면 된다 ----
SOURCE_DIR = './Data/external'        # 외부 이미지를 넣어둘 폴더
SAVE_ROOT = './result/external'       # 이 실행의 결과가 전부 이 아래에 쌓인다

CONF = 0.45         # 이 값보다 확신이 낮은 검출은 버린다
IMGSZ = 640

# 몇 장을 합칠지. 0 이면 전부.
MAX_COUNT = 0

# 결과가 만들어지는 곳 (모델 구분값 -> 폴더 이름)
#   result/external/predict_detect  predict_seg  predict_unet  sum_result
PREDICT_NAMES = {
    'detect': 'predict_detect',
    'segment': 'predict_seg',
    'unet': 'predict_unet',
}
SUM_NAME = 'sum_result'


# 결과 폴더로 반환
def predict_dir(key):
    return os.path.join(SAVE_ROOT, PREDICT_NAMES[key])


#
def find_images(image_dir):
    """폴더에서 이미지 파일 이름만 골라 정렬해서 돌려준다."""
    if not os.path.isdir(image_dir):
        return []

    return sorted(f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXT))


def predict_yolo(key, show_counts=True):
    """
    YOLO 모델(detect / segment)로 외부 이미지를 추론하고 결과 이미지를 저장한다.

    step4 의 predict_images 를 쓰지 않는 이유 : 그쪽은 저장 위치가 './result' 로 고정이라
    이 파일이 폴더를 정할 수 없다. 대신 여기서 직접 부르고 project 를 넘긴다.
    """
    model = YOLO(WEIGHTS[key])

    # project 는 반드시 절대경로로 준다.
    # 상대경로면 ultralytics 가 runs/<task>/ 아래로 옮겨 버린다.
    results = model.predict(source=SOURCE_DIR,
                            conf=CONF,
                            imgsz=IMGSZ,
                            save=True,
                            project=os.path.abspath(SAVE_ROOT),
                            name=PREDICT_NAMES[key],
                            exist_ok=True,
                            verbose=False)

    if show_counts:
        for result in results:
            counts = count_classes(result)
            file_name = os.path.basename(result.path)

            if counts:
                print(f'  {file_name} : {counts}')
            else:
                print(f'  {file_name} : 검출된 안전장구 없음')

    print(f'{MODEL_NAMES[key]} 결과 저장 : {predict_dir(key)}')

    return results


def predict_unet_images():
    """U-Net 으로 외부 이미지를 추론한다. 박스가 없어 색칠한 그림만 나온다."""
    model = unet_load(WEIGHTS['unet'], NUM_CLASSES)
    unet_predict(model, SOURCE_DIR, predict_dir('unet'), CLASS_NAMES,
                 input_size=UNET_INPUT_SIZE)


def make_one(file_name):
    """원본 + 세 모델 결과를 격자 한 장으로 만든다. 하나라도 없으면 None."""
    cells = []

    cell = read_cell(os.path.join(SOURCE_DIR, file_name), 'origin')

    if cell is None:
        return None

    cells.append(cell)

    for key in PREDICT_NAMES:
        cell = read_cell(os.path.join(predict_dir(key), file_name), MODEL_NAMES[key])

        if cell is None:
            return None

        cells.append(cell)

    return make_grid(cells)


def find_common_files():
    """원본과 세 결과 폴더에 모두 있는 파일 이름을 돌려준다."""
    names = set(find_images(SOURCE_DIR))

    if not names:
        print(f'[주의] 외부 이미지가 없습니다 : {SOURCE_DIR}')
        return []

    for key in PREDICT_NAMES:
        folder = predict_dir(key)

        if not os.path.isdir(folder):
            print(f'[주의] {MODEL_NAMES[key]} 결과 폴더가 없습니다 : {folder}')
            return []

        names &= set(os.listdir(folder))

    return sorted(names)


def sum_results(max_count=MAX_COUNT):
    """원본과 세 결과를 한 장으로 합쳐 result/external/sum_result/ 에 저장한다."""
    files = find_common_files()

    if not files:
        print('합칠 이미지가 없습니다.')
        return 0

    total = len(files)

    if max_count > 0:
        files = files[:max_count]

    save_dir = os.path.join(SAVE_ROOT, SUM_NAME)
    os.makedirs(save_dir, exist_ok=True)

    saved = 0

    for file_name in tqdm(files, desc='합치는 중'):
        merged = make_one(file_name)

        if merged is None:
            print(f'  [건너뜀] 읽지 못한 이미지가 있습니다 : {file_name}')
            continue

        cv2.imwrite(os.path.join(save_dir, file_name), merged)
        saved += 1

    print(f'합친 이미지 {saved}장 저장 : {save_dir}  (짝이 맞는 전체 {total}장)')

    return saved


def run_all():
    """외부 이미지에 세 모델을 돌리고 결과를 한 장으로 합치는 것까지 한 번에 한다."""
    images = find_images(SOURCE_DIR)

    print_settings('외부 데이터 추론', {
        '입력 폴더': SOURCE_DIR,
        '이미지 수': f'{len(images)}장',
        '저장 위치': SAVE_ROOT,
        'conf': CONF,
        'imgsz': IMGSZ,
        '모델': ' / '.join(MODEL_NAMES[key] for key in PREDICT_NAMES),
    })

    if not images:
        print(f'{SOURCE_DIR} 에 이미지를 넣고 다시 실행하세요.')
        return

    # 가중치가 없으면 그 모델만 건너뛴다
    missing = [key for key in PREDICT_NAMES if not os.path.exists(WEIGHTS[key])]

    if missing:
        for key in missing:
            print(f'[안내] {MODEL_NAMES[key]} 가중치가 없습니다 : {WEIGHTS[key]}')
        print('  step2_train.py 로 학습을 끝낸 뒤 다시 실행하세요.')
        return

    print('\n===== yolov8n(detect) 추론 =====')
    predict_yolo('detect')

    print('\n===== yolov8n-seg 추론 =====')
    predict_yolo('segment')

    print('\n===== U-Net 추론 =====')
    predict_unet_images()

    print('\n===== 결과 합치기 =====')
    sum_results()


if __name__ == '__main__':
    run_all()
