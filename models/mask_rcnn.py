"""
역할: torchvision Mask R-CNN 모델 만들기 / 학습 / 평가

yolov8n-seg 와 비교할 torch 계열 segmentation 모델이다.
sassc8_CNN 에서 다룬 Faster R-CNN 에 '마스크를 예측하는 머리'가 하나 더 붙은 구조다.

  Backbone(ResNet50) : 이미지에서 특징 추출
  RPN                : 물체가 있을 만한 후보 영역 제안
  ROI Head           : 후보 영역을 분류 + 박스 보정 + 마스크 예측  <- 여기를 우리 클래스 수에 맞게 교체

학습 모드에서 model(images, targets) 은 loss 딕셔너리를,
평가 모드에서 model(images) 는 예측 딕셔너리(boxes, labels, scores, masks)를 돌려준다.
"""

import os
import time

import cv2
import numpy as np
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms
from tqdm import tqdm

from models.seg_dataset import IMAGE_EXT, get_dataloader
from utils.metrics import box_iou

# 예측을 '검출했다'고 인정할 최소 점수, 정답과 같다고 볼 최소 IoU
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5


def get_device(device=None):
    """GPU 가 있으면 cuda, 없으면 cpu 를 쓴다."""
    if device is not None:
        return device

    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(num_classes):
    """
    사전학습된 Mask R-CNN 을 불러와 우리 클래스 수에 맞게 머리 부분을 바꾼다.
    num_classes : 배경을 뺀 실제 클래스 수 (helmet, vest -> 2)
    """
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # 1) 박스 예측 머리 교체 (배경 포함이라 +1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    # 2) 마스크 예측 머리 교체
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes + 1)

    return model


def load_trained_model(weights_path, num_classes, device=None):
    """저장해둔 가중치(.pth)를 불러온다."""
    device = get_device(device)

    model = load_model(num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def train_one_epoch(model, loader, optimizer, device):
    """1 epoch 학습하고 평균 loss 를 돌려준다."""
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(loader, desc='Train')
    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # 학습 모드에서는 loss 딕셔너리가 나온다 (분류/박스/마스크/RPN loss)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def train_model(dataset_dir, num_classes, save_path,
                epochs=10, batch_size=2, lr=0.005, device=None):
    """Mask R-CNN 을 학습하고 가중치를 저장한다."""
    device = get_device(device)
    print(f'학습 장치 : {device}')

    train_loader = get_dataloader(dataset_dir, 'train', batch_size=batch_size, shuffle=True)

    model = load_model(num_classes)
    model.to(device)

    # 학습이 필요한 파라미터만 골라 옵티마이저에 넣는다
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    # 학습이 진행될수록 학습률을 낮춰 안정적으로 수렴시킨다
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for e in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        print(f'[EPOCH {e + 1}/{epochs}] avg_loss {avg_loss:.4f}')

    torch.save(model.state_dict(), save_path)
    print(f'모델 저장 완료 : {save_path}')

    return model


def match_boxes(pred_boxes, pred_labels, true_boxes, true_labels):
    """
    예측 박스와 정답 박스를 짝지어 맞춘 개수를 센다.

    정답 하나마다 클래스가 같고 IoU 가 가장 높은 예측을 찾는다.
    이미 다른 정답에 쓰인 예측은 다시 쓰지 않는다.

    돌려주는 값 : (맞춘 개수, 정답별 최고 IoU 목록)
    """
    used = set()
    matched = 0
    iou_list = []

    for true_box, true_label in zip(true_boxes, true_labels):
        best_iou = 0.0
        best_index = -1

        for i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
            if i in used or pred_label != true_label:
                continue

            iou = box_iou(true_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_index = i

        iou_list.append(best_iou)

        # IoU 가 기준을 넘으면 제대로 찾은 것으로 본다
        if best_iou >= IOU_THRESHOLD and best_index >= 0:
            used.add(best_index)
            matched += 1

    return matched, iou_list


def evaluate_model(model, dataset_dir, split='test', device=None):
    """
    테스트 데이터로 지표를 계산한다. (YOLO 와 같은 항목으로 맞춘다)
      mAP50, mAP50-95 : torchmetrics 로 계산 (YOLO 와 같은 COCO 방식)
      Precision, Recall, F1 : 점수 0.5 이상 예측을 IoU 0.5 로 맞춰서 직접 계산
      mIoU : 정답 박스마다 가장 잘 맞은 예측과의 IoU 평균
      추론시간, FPS : 이미지 한 장 처리 시간
    """
    # torchmetrics 는 따로 설치해야 하므로 필요할 때만 불러온다
    from torchmetrics.detection import MeanAveragePrecision

    device = get_device(device)
    model.to(device)
    model.eval()

    # 속도를 재야 하므로 한 장씩 넣는다
    loader = get_dataloader(dataset_dir, split, batch_size=1, shuffle=False)
    metric = MeanAveragePrecision(iou_type='bbox')

    total_matched = 0    # 제대로 찾은 개수 (TP)
    total_pred = 0       # 예측한 개수
    total_true = 0       # 정답 개수
    all_iou = []
    total_time = 0.0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Eval'):
            images = [img.to(device) for img in images]

            start = time.time()
            outputs = model(images)
            total_time += time.time() - start

            for output, target in zip(outputs, targets):
                # mAP 계산기에는 모든 예측을 그대로 넣는다
                metric.update(
                    [{'boxes': output['boxes'].cpu(),
                      'scores': output['scores'].cpu(),
                      'labels': output['labels'].cpu()}],
                    [{'boxes': target['boxes'],
                      'labels': target['labels']}],
                )

                # Precision/Recall/IoU 는 점수가 높은 예측만 가지고 계산한다
                keep = output['scores'].cpu() >= SCORE_THRESHOLD
                pred_boxes = output['boxes'].cpu()[keep].tolist()
                pred_labels = output['labels'].cpu()[keep].tolist()

                true_boxes = target['boxes'].tolist()
                true_labels = target['labels'].tolist()

                matched, iou_list = match_boxes(pred_boxes, pred_labels, true_boxes, true_labels)

                total_matched += matched
                total_pred += len(pred_boxes)
                total_true += len(true_boxes)
                all_iou += iou_list

    map_result = metric.compute()

    precision = total_matched / total_pred if total_pred > 0 else 0.0
    recall = total_matched / total_true if total_true > 0 else 0.0
    infer_ms = (total_time / len(loader)) * 1000 if len(loader) > 0 else 0.0

    scores = {
        'mAP50': float(map_result['map_50']),
        'mAP50-95': float(map_result['map']),
        'Precision': precision,
        'Recall': recall,
        'F1': 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0,
        'mIoU': sum(all_iou) / len(all_iou) if all_iou else 0.0,
        '추론시간(ms)': infer_ms,
        'FPS': 1000 / infer_ms if infer_ms > 0 else 0.0,
    }

    return scores


def predict_images(model, image_dir, save_dir, class_names, device=None):
    """
    테스트 이미지를 추론해서 박스와 마스크를 그린 결과 이미지를 저장한다.
    (YOLO 는 save=True 한 줄이면 되지만 torch 모델은 직접 그려야 한다)
    """
    device = get_device(device)
    model.to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    to_tensor = transforms.ToTensor()
    image_list = [f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXT)]

    for file_name in image_list:
        image = cv2.imread(os.path.join(image_dir, file_name))
        image_tensor = to_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to(device)

        with torch.no_grad():
            output = model([image_tensor])[0]

        counts = {}

        for i in range(len(output['scores'])):
            if float(output['scores'][i]) < SCORE_THRESHOLD:
                continue

            label = int(output['labels'][i])
            name = class_names[label - 1] if label - 1 < len(class_names) else str(label)
            counts[name] = counts.get(name, 0) + 1

            # 마스크는 0~1 확률값이라 0.5 를 넘는 부분만 색을 입힌다
            mask = output['masks'][i, 0].cpu().numpy() > 0.5
            image[mask] = (image[mask] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)

            x1, y1, x2, y2 = [int(v) for v in output['boxes'][i].cpu()]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, name, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(save_dir, file_name), image)
        print(f'{file_name} : {counts if counts else "검출된 안전장구 없음"}')

    print(f'결과 이미지 저장 : {save_dir}')
