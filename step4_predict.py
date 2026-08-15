"""
역할: 테스트 이미지로 결과를 확인한다.

폴더 안의 이미지를 한꺼번에 추론해서
  - 결과 이미지(박스 또는 마스크가 그려진 그림)를 result/ 아래에 저장하고
  - 이미지마다 어떤 안전장구가 몇 개 검출됐는지 출력한다.

실행 : python predict.py
"""

import os

from ultralytics import YOLO

# ---- 추론 설정 ----
DETECT_WEIGHTS = './runs/detect/vest_helmet_detect/weights/best.pt'
SEGMENT_WEIGHTS = './runs/segment/vest_helmet_seg/weights/best.pt'

TEST_IMAGE_DIR = './Data/vest-helmet.v1i_roboflow/test/images'

CONF = 0.45         # 이 값보다 확신이 낮은 검출은 버린다
IMGSZ = 640
SAVE_DIR = './result'


def count_classes(result):
    """검출 결과에서 클래스별 개수를 세어 딕셔너리로 돌려준다."""
    counts = {}

    for box in result.boxes:
        name = result.names[int(box.cls[0])]
        counts[name] = counts.get(name, 0) + 1

    return counts


def predict_images(weights, source=TEST_IMAGE_DIR, save_name='predict'):
    """
    이미지 폴더(또는 이미지 한 장)를 추론하고 결과를 저장한다.
    save_name : 결과가 저장될 폴더 이름 (result/<save_name>)
    """
    model = YOLO(weights)

    results = model.predict(source=source,
                            conf=CONF,
                            imgsz=IMGSZ,
                            save=True,
                            project=SAVE_DIR,
                            name=save_name,
                            exist_ok=True,
                            verbose=False)

    for result in results:
        counts = count_classes(result)

        # 파일 경로가 길어서 파일 이름만 잘라서 보여준다
        file_name = os.path.basename(result.path)

        if counts:
            print(f'{file_name} : {counts}')
        else:
            print(f'{file_name} : 검출된 안전장구 없음')

    print(f'결과 이미지 저장 : {SAVE_DIR}/{save_name}')

    return results


if __name__ == '__main__':
    predict_images(DETECT_WEIGHTS, save_name='predict_detect')

    # segmentation 모델 학습이 끝나면 아래 주석을 푼다
    # predict_images(SEGMENT_WEIGHTS, save_name='predict_seg')
