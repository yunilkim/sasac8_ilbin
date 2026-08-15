"""
역할: IoU(Intersection over Union) 계산

mAP 는 ultralytics 가 알아서 계산해주지만, 두 모델을 비교할 때
"정답 박스와 예측 박스가 얼마나 겹치는가"를 직접 보기 위해 평균 IoU 를 구한다.

detection 라벨과 segmentation 라벨을 모두 '박스'로 바꿔서 비교하기 때문에
두 모델을 같은 기준으로 볼 수 있다.
  - detection 라벨  : <클래스> <cx> <cy> <w> <h>          -> 그대로 박스
  - segment 라벨    : <클래스> <x1> <y1> <x2> <y2> ...    -> 폴리곤을 감싸는 박스로 변환
"""

import os

IMAGE_EXT = ('.jpg', '.jpeg', '.png')


def box_iou(box1, box2):
    """
    두 박스의 IoU 를 구한다. 박스 형식은 (x1, y1, x2, y2) 픽셀 좌표.
    IoU = 겹치는 넓이 / 합친 넓이  (0 ~ 1, 1에 가까울수록 잘 맞춘 것)
    """
    # 겹치는 사각형의 좌표
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 겹치는 넓이 (안 겹치면 0)
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


def read_label_boxes(txt_path, image_w, image_h):
    """
    라벨 txt 를 읽어 [(클래스, x1, y1, x2, y2), ...] 형태로 돌려준다.
    정규화된 값(0~1)을 픽셀 좌표로 바꾼다.
    """
    boxes = []

    if not os.path.exists(txt_path):
        return boxes

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]

            if len(values) == 4:
                # detection 라벨 : 중심점 + 크기 -> 좌상단/우하단
                cx, cy, w, h = values
                x1 = (cx - w / 2) * image_w
                y1 = (cy - h / 2) * image_h
                x2 = (cx + w / 2) * image_w
                y2 = (cy + h / 2) * image_h
            else:
                # segmentation 라벨 : 폴리곤 꼭짓점 -> 감싸는 박스
                xs = values[0::2]   # 짝수 번째가 x
                ys = values[1::2]   # 홀수 번째가 y
                x1 = min(xs) * image_w
                y1 = min(ys) * image_h
                x2 = max(xs) * image_w
                y2 = max(ys) * image_h

            boxes.append((cls, x1, y1, x2, y2))

    return boxes


def mean_iou(model, image_dir, conf=0.45):
    """
    테스트 이미지 전체에 대해 '정답 박스마다 가장 잘 맞은 예측 박스'의 IoU 를 구하고 평균을 낸다.
    정답을 아예 못 찾은 경우는 IoU 0 으로 계산한다.

    model     : YOLO 모델 객체
    image_dir : 테스트 이미지 폴더
    """
    label_dir = image_dir.replace('images', 'labels')
    image_list = [f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXT)]

    iou_list = []

    for image_name in image_list:
        image_path = os.path.join(image_dir, image_name)
        txt_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')

        result = model.predict(image_path, conf=conf, verbose=False)[0]

        image_h, image_w = result.orig_shape
        true_boxes = read_label_boxes(txt_path, image_w, image_h)

        # 예측 박스 : (클래스, x1, y1, x2, y2)
        pred_boxes = []
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            pred_boxes.append((cls, x1, y1, x2, y2))

        # 정답 박스 하나마다 같은 클래스 예측 중 가장 IoU 가 높은 것을 찾는다
        for true_box in true_boxes:
            best = 0.0

            for pred_box in pred_boxes:
                if pred_box[0] != true_box[0]:
                    continue

                iou = box_iou(true_box[1:], pred_box[1:])
                if iou > best:
                    best = iou

            iou_list.append(best)

    if not iou_list:
        return 0.0

    return sum(iou_list) / len(iou_list)
