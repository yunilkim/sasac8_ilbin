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

import csv
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

from models.seg_dataset import IMAGE_EXT, get_dataloader, has_split
from utils.metrics import box_iou

# 예측을 '검출했다'고 인정할 최소 점수, 정답과 같다고 볼 최소 IoU
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# 입력 이미지 크기
# torchvision 기본값은 min_size=800 이라 640 짜리 우리 이미지를 800 으로 '늘려서' 넣는다.
# YOLO 는 640 으로 학습하므로, 같은 조건으로 비교하려면 여기도 640 으로 맞춰야 한다.
# (덤으로 메모리와 학습 시간도 줄어든다 : 1 epoch 29.7분 -> 21.3분)
INPUT_SIZE = 640


def get_device(device=None):
    """GPU 가 있으면 cuda, 없으면 cpu 를 쓴다."""
    if device is not None:
        return device

    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(num_classes, input_size=INPUT_SIZE):
    """
    사전학습된 Mask R-CNN 을 불러와 우리 클래스 수에 맞게 머리 부분을 바꾼다.
    num_classes : 배경을 뺀 실제 클래스 수 (helmet, vest -> 2)
    input_size  : 모델에 들어갈 이미지 크기 (YOLO 와 맞추려고 640 을 쓴다)
    """
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # 1) 박스 예측 머리 교체 (배경 포함이라 +1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    # 2) 마스크 예측 머리 교체
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes + 1)

    # 3) 입력 크기 고정 (기본값 800 으로 확대되는 것을 막는다)
    model.transform.min_size = (input_size,)
    model.transform.max_size = input_size

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


def freeze_batchnorm(model):
    """
    BatchNorm 을 고정한다.

    검증할 때 loss 를 얻으려면 모델을 train 모드로 둬야 하는데(eval 모드는 예측만 돌려준다),
    그러면 BatchNorm 이 검증 데이터로 통계를 갱신해 모델이 오염된다.
    그래서 BatchNorm 만 따로 eval 모드로 돌려 통계가 바뀌지 않게 한다.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


def validate_one_epoch(model, loader, device):
    """
    검증 데이터로 loss 를 구한다. (학습은 하지 않는다)
    이 값이 더 이상 줄지 않으면 학습을 멈춘다.
    """
    model.train()          # loss 를 받으려면 train 모드여야 한다
    freeze_batchnorm(model)  # 대신 BatchNorm 통계는 고정한다

    total_loss = 0.0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Valid', leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            total_loss += sum(loss_dict.values()).item()

    return total_loss / len(loader)


def save_epoch_log(log_path, rows):
    """epoch 별 loss 를 csv 로 남긴다. (YOLO 의 results.csv 와 같은 역할)"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'lr', 'best'])
        writer.writeheader()
        writer.writerows(rows)


def make_run_dir(base_dir, name):
    """
    학습 결과를 담을 폴더를 만든다.
    이미 있으면 뒤에 번호를 붙여 새 폴더를 만든다. (YOLO 와 같은 방식)
    """
    run_dir = os.path.join(base_dir, name)
    number = 2

    while os.path.exists(run_dir):
        run_dir = os.path.join(base_dir, f'{name}{number}')
        number += 1

    os.makedirs(run_dir)

    return run_dir


def train_model(dataset_dir, num_classes, save_path,
                epochs=30, batch_size=2, lr=0.005, patience=5,
                run_dir=None, device=None):
    """
    Mask R-CNN 을 학습한다.

    YOLO 와 마찬가지로
      - epoch 마다 검증 데이터로 성능(loss)을 확인하고
      - 가장 좋았던 epoch 의 가중치를 저장하며
      - patience 번 연속 나아지지 않으면 남은 epoch 을 건너뛴다 (Early Stopping)

    검증 데이터(valid)가 없으면 Early Stopping 없이 끝까지 학습하고
    마지막 epoch 을 저장한다. (이 경우 화면에 안내가 나온다)

    돌려주는 값 : {'best_epoch':…, 'best_val_loss':…, 'epochs_ran':…, 'log_path':…}
    """
    device = get_device(device)
    print(f'학습 장치 : {device}')

    train_loader = get_dataloader(dataset_dir, 'train',
                                  batch_size=batch_size, shuffle=True, augment=True)

    # 검증 데이터가 있으면 Early Stopping 을 쓴다
    use_valid = has_split(dataset_dir, 'valid')

    if use_valid:
        valid_loader = get_dataloader(dataset_dir, 'valid',
                                      batch_size=batch_size, shuffle=False, augment=False)
    else:
        valid_loader = None
        print('[안내] valid 마스크가 없어 Early Stopping 없이 끝까지 학습합니다.')

    model = load_model(num_classes)
    model.to(device)

    # 학습이 필요한 파라미터만 골라 옵티마이저에 넣는다
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    # 학습이 진행될수록 학습률을 낮춰 안정적으로 수렴시킨다.
    #
    # 코사인 곡선을 따라 매 epoch 조금씩 줄인다 (T_max epoch 에 걸쳐 0 까지).
    # 계단식(StepLR)으로 뚝뚝 떨어뜨리면, 계단과 계단 사이에서 loss 가 정체됐다가
    # 다음 계단에서 다시 내려가는 일이 잦다. 그러면 Early Stopping 이
    # '아직 수렴 안 했는데' 멈춰버린다. 부드럽게 줄이면 그런 착시가 없다.
    # (YOLO 도 연속적으로 줄이는 방식이라 두 모델의 성격이 맞는다)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    best_epoch = 0
    bad_count = 0        # 연속으로 나아지지 않은 횟수
    rows = []

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for e in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        now_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        val_loss = validate_one_epoch(model, valid_loader, device) if use_valid else None

        # 검증 loss 가 있으면 그걸로, 없으면 train loss 로 좋고 나쁨을 판단한다
        score = val_loss if val_loss is not None else train_loss
        is_best = score < best_loss

        if is_best:
            best_loss = score
            best_epoch = e
            bad_count = 0
            torch.save(model.state_dict(), save_path)
        else:
            bad_count += 1

        rows.append({
            'epoch': e,
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4) if val_loss is not None else '',
            'lr': f'{now_lr:.6f}',
            'best': 'O' if is_best else '',
        })

        text = f'[EPOCH {e}/{epochs}] train {train_loss:.4f}'
        if val_loss is not None:
            text += f'  val {val_loss:.4f}'
        text += f'  lr {now_lr:.6f}'
        if is_best:
            text += '  <- best'
        print(text)

        # Early Stopping : 검증 데이터가 있을 때만 동작한다
        if use_valid and bad_count >= patience:
            print(f'{patience} epoch 연속 나아지지 않아 학습을 멈춥니다. '
                  f'(가장 좋았던 epoch {best_epoch})')
            break

    log_path = os.path.join(run_dir, 'results.csv') if run_dir else None

    if log_path:
        save_epoch_log(log_path, rows)
        print(f'epoch 기록 저장 : {log_path}')

    print(f'가장 좋았던 epoch {best_epoch} (loss {best_loss:.4f}) 의 가중치를 저장했습니다 : {save_path}')

    return {
        'best_epoch': best_epoch,
        'best_val_loss': best_loss,
        'epochs_ran': len(rows),
        'log_path': log_path,
    }


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
