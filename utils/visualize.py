"""
역할: 그림으로 확인하는 기능 모음

  1) show_label()        : 이미지 위에 정답 라벨을 그려서 라벨이 제대로 됐는지 눈으로 확인
  2) draw_compare_plot() : detection 모델과 segmentation 모델의 지표를 막대그래프로 비교
"""

import os

import cv2
import matplotlib.pyplot as plt

from utils.metrics import read_label_boxes


def show_label(image_file, txt_file):
    """이미지와 라벨 txt 를 겹쳐 그린다. detection, segmentation 라벨 둘 다 된다."""
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_h, image_w = image.shape[:2]

    boxes = read_label_boxes(txt_file, image_w, image_h)

    for cls, x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image, str(cls), (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    plt.imshow(image)
    plt.axis('off')
    plt.show()


def draw_compare_plot(scores, save_path='./result/compare.jpg'):
    """
    두 모델의 지표를 나란히 막대그래프로 그린다.

    scores 예시
      {'detect':  {'mAP50': 0.8, 'mAP50-95': 0.5, ...},
       'segment': {'mAP50': 0.85, 'mAP50-95': 0.55, ...}}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 0~1 사이 지표만 그래프로 그린다 (속도나 용량은 단위가 달라서 표로만 본다)
    keys = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'mIoU']
    keys = [k for k in keys if k in scores['detect']]

    x = range(len(keys))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], [scores['detect'][k] for k in keys],
            width=width, label='detection (yolov8n)')
    plt.bar([i + width / 2 for i in x], [scores['segment'][k] for k in keys],
            width=width, label='segmentation (yolov8n-seg)')

    plt.xticks(list(x), keys)
    plt.ylim(0, 1.0)
    plt.title('detection vs segmentation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f'그래프 저장 : {save_path}')
