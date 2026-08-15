"""
역할: [3단계] 학습한 모델의 성능 지표를 뽑는다.

test 셋으로 mAP50, mAP50-95, Precision, Recall, 추론 속도를 구한다.
  - YOLO 모델      : ultralytics 가 지표를 계산해준다 (eval_model)
  - Mask R-CNN     : ultralytics 를 못 쓰므로 torchmetrics 와 직접 만든 IoU 로 계산 (eval_maskrcnn)
(PR Curve, Confusion Matrix 그림은 ultralytics 가 runs/.../val 폴더에 자동 저장한다)

실행 : python step3_eval.py
"""

from ultralytics import YOLO

from models.mask_rcnn import evaluate_model, load_trained_model
from step2_train import MASKRCNN_PATH, NUM_CLASSES, SEGMENT_DATASET

# ---- 평가 설정 ----
DETECT_WEIGHTS = './runs/detect/vest_helmet_detect/weights/best.pt'
SEGMENT_WEIGHTS = './runs/segment/vest_helmet_seg/weights/best.pt'

DETECT_YAML = './config/detect.yaml'
SEGMENT_YAML = './config/segment.yaml'

IMGSZ = 640
SPLIT = 'test'      # train / val / test


def eval_model(weights, data_yaml, split=SPLIT):
    """
    YOLO 모델 하나를 평가하고 지표를 딕셔너리로 돌려준다.
    detection 모델과 segmentation 모델 둘 다 이 함수를 쓴다.
    """
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, split=split, imgsz=IMGSZ)

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    scores = {
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'Precision': precision,
        'Recall': recall,
    }

    # F1 = 2 * (정밀도 * 재현율) / (정밀도 + 재현율)
    if precision + recall > 0:
        scores['F1'] = 2 * precision * recall / (precision + recall)
    else:
        scores['F1'] = 0.0

    # 이미지 한 장 추론에 걸린 시간(ms) -> FPS 로도 바꿔본다
    infer_ms = float(metrics.speed['inference'])
    scores['추론시간(ms)'] = infer_ms
    scores['FPS'] = 1000 / infer_ms if infer_ms > 0 else 0.0

    # segmentation 모델일 때만 마스크 지표가 있다
    if hasattr(metrics, 'seg'):
        scores['mask_mAP50'] = float(metrics.seg.map50)
        scores['mask_mAP50-95'] = float(metrics.seg.map)

    return scores


def eval_maskrcnn(weights=MASKRCNN_PATH, dataset_dir=SEGMENT_DATASET, split=SPLIT):
    """Mask R-CNN 을 평가하고 YOLO 와 같은 항목의 지표를 돌려준다."""
    model = load_trained_model(weights, NUM_CLASSES)
    scores = evaluate_model(model, dataset_dir, split=split)

    return scores


def print_scores(title, scores):
    """지표 딕셔너리를 보기 좋게 출력한다."""
    print(f'--- {title} ---')

    for key, value in scores.items():
        print(f'{key:15s} : {value:.4f}')


if __name__ == '__main__':
    detect_scores = eval_model(DETECT_WEIGHTS, DETECT_YAML)
    print_scores('detection (yolov8n)', detect_scores)

    # segmentation 모델 학습이 끝나면 아래 주석을 푼다
    # segment_scores = eval_model(SEGMENT_WEIGHTS, SEGMENT_YAML)
    # print_scores('segmentation (yolov8n-seg)', segment_scores)

    # maskrcnn_scores = eval_maskrcnn()
    # print_scores('segmentation (Mask R-CNN)', maskrcnn_scores)
