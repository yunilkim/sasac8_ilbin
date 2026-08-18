"""
역할: U-Net (시맨틱 분할) 모델 만들기 / 학습 / 평가

yolov8n-seg 와 다른 점
  yolov8n-seg : 객체를 하나씩 찾아 각각 마스크를 준다 (인스턴스 분할)
  U-Net       : 픽셀마다 클래스 하나를 찍는다 (시맨틱 분할)
                겹친 조끼 두 벌은 하나의 덩어리가 된다. 박스도 확신도도 없다.

구조
  U 자 모양이라 U-Net 이다. 왼쪽에서 이미지를 줄여가며 '무엇인지'를 파악하고,
  오른쪽에서 다시 키우며 '어디인지'를 복원한다.
  줄이는 과정에서 잃어버린 위치 정보를 살리려고, 같은 크기의 왼쪽 결과를
  오른쪽에 이어 붙인다(skip connection). 이게 U-Net 의 핵심이다.

  수축 경로 : ImageNet 으로 사전학습된 ResNet34 를 그대로 쓴다.
              (yolov8n-seg 가 COCO 사전학습을 쓰므로 공정하게 맞춘 것)
  확장 경로 : 업샘플 -> skip 연결 -> conv 2번 을 반복한다.
"""

import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34
from tqdm import tqdm

from models.unet_dataset import get_dataloader, has_split

# 입력 이미지 크기 (YOLO 와 동일하게 맞춘다)
INPUT_SIZE = 640


def double_conv(in_ch, out_ch):
    """conv -> BN -> ReLU 를 두 번 거치는 기본 블록"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
    """
    확장 경로의 한 칸.
    작은 특징맵을 2배로 키우고, 수축 경로에서 온 같은 크기 특징맵을 옆에 붙인 뒤 conv 한다.
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = double_conv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            # 크기가 1픽셀 어긋날 수 있어 맞춰준다
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

            x = torch.cat([x, skip], dim=1)   # 채널 방향으로 이어 붙인다

        return self.conv(x)


class UNet(nn.Module):
    """ResNet34 를 수축 경로로 쓰는 U-Net"""

    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)

        # --- 수축 경로 (ResNet34 를 단계별로 쪼갠다) ---
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 1/2,  64채널
        self.pool = backbone.maxpool                                            # 1/4
        self.enc1 = backbone.layer1                                             # 1/4,  64
        self.enc2 = backbone.layer2                                             # 1/8, 128
        self.enc3 = backbone.layer3                                             # 1/16, 256
        self.enc4 = backbone.layer4                                             # 1/32, 512

        # --- 확장 경로 (skip 으로 받는 채널 수를 맞춰 준다) ---
        self.up4 = UpBlock(512, 256, 256)   # 1/32 -> 1/16
        self.up3 = UpBlock(256, 128, 128)   # 1/16 -> 1/8
        self.up2 = UpBlock(128, 64, 64)     # 1/8  -> 1/4
        self.up1 = UpBlock(64, 64, 64)      # 1/4  -> 1/2
        self.up0 = UpBlock(64, 0, 32)       # 1/2  -> 1/1 (붙일 skip 이 없다)

        # 픽셀마다 클래스 점수를 낸다 (배경 포함이라 +1)
        self.head = nn.Conv2d(32, num_classes + 1, kernel_size=1)

    def forward(self, x):
        s0 = self.stem(x)          # 1/2
        s1 = self.enc1(self.pool(s0))   # 1/4
        s2 = self.enc2(s1)         # 1/8
        s3 = self.enc3(s2)         # 1/16
        s4 = self.enc4(s3)         # 1/32

        d = self.up4(s4, s3)
        d = self.up3(d, s2)
        d = self.up2(d, s1)
        d = self.up1(d, s0)
        d = self.up0(d)

        return self.head(d)


def get_device(device=None):
    """GPU 가 있으면 cuda, 없으면 cpu 를 쓴다."""
    if device is not None:
        return device

    return 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_class_weights(dataset_dir, num_classes, sample=300):
    """
    클래스마다 얼마나 자주 나오는지 세어 가중치를 만든다.

    이 데이터는 배경이 83%, 조끼 13%, 헬멧 3% 라서 그냥 학습하면
    '전부 배경'이라고 찍어도 정확도가 높게 나온다. 그러면 헬멧을 못 배운다.
    드문 클래스에 더 큰 가중치를 줘서 균형을 맞춘다.

    가중치 = 1 / sqrt(빈도)  를 평균 1 이 되도록 맞춘 값.
    (1/빈도 는 너무 극단적이라 제곱근으로 완화한다)
    """
    from PIL import Image

    semantic_dir = os.path.join(dataset_dir, 'train', 'semantic')
    files = sorted(f for f in os.listdir(semantic_dir) if f.endswith('.png'))[:sample]

    counts = np.zeros(num_classes + 1)

    for file_name in files:
        arr = np.array(Image.open(os.path.join(semantic_dir, file_name)))
        values, n = np.unique(arr, return_counts=True)

        for v, c in zip(values, n):
            if v <= num_classes:
                counts[int(v)] += int(c)

    freq = counts / counts.sum()
    weights = 1.0 / np.sqrt(np.maximum(freq, 1e-8))
    weights = weights / weights.mean()

    print(f'  클래스 픽셀 비율 : {[f"{v * 100:.1f}%" for v in freq]}')
    print(f'  클래스 가중치    : {[f"{v:.2f}" for v in weights]}')

    return torch.tensor(weights, dtype=torch.float32)


def load_model(num_classes, pretrained=True):
    """U-Net 을 만든다."""
    return UNet(num_classes, pretrained=pretrained)


def load_trained_model(weights_path, num_classes, device=None):
    """저장해둔 가중치를 불러온다."""
    device = get_device(device)

    model = load_model(num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    """1 epoch 학습하고 평균 loss 를 돌려준다."""
    model.train()
    total_loss = 0.0

    progress = tqdm(loader, desc='Train')
    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # 출력 (배치, 클래스수+1, H, W) 와 정답 (배치, H, W) 로 픽셀마다 분류한다
        output = model(images)
        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device, num_classes):
    """
    검증 데이터로 loss 와 클래스별 IoU 를 구한다.
    IoU = 맞게 예측한 픽셀 / (예측한 픽셀 + 정답 픽셀 - 맞게 예측한 픽셀)
    """
    model.eval()

    total_loss = 0.0
    inter = np.zeros(num_classes + 1)
    union = np.zeros(num_classes + 1)

    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Valid', leave=False):
            images = images.to(device)
            targets = targets.to(device)

            output = model(images)
            total_loss += criterion(output, targets).item()

            # 점수가 가장 높은 클래스를 고른다
            predict = output.argmax(dim=1)

            for c in range(num_classes + 1):
                p = predict == c
                t = targets == c
                inter[c] += float((p & t).sum())
                union[c] += float((p | t).sum())

    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)

    # 배경(0)을 뺀 실제 클래스만 평균낸다
    miou = float(np.nanmean(iou[1:]))

    return total_loss / len(loader), miou, iou


def save_epoch_log(log_path, rows):
    """epoch 별 기록을 csv 로 남긴다."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'val_mIoU',
                                               'lr', 'best'])
        writer.writeheader()
        writer.writerows(rows)


def train_model(dataset_dir, num_classes, save_path,
                epochs=30, batch_size=3, lr=0.001, patience=7,
                input_size=INPUT_SIZE, run_dir=None, device=None,
                use_class_weights=True):
    """
    U-Net 을 학습한다.

    yolov8n-seg 와 마찬가지로
      - epoch 마다 검증하고 (loss 와 mIoU)
      - 가장 좋았던 epoch 의 가중치를 저장하며
      - patience 번 연속 나아지지 않으면 멈춘다

    판단 기준은 mIoU 다. (높을수록 좋다)
    """
    device = get_device(device)
    print(f'학습 장치 : {device}')

    train_loader = get_dataloader(dataset_dir, 'train', batch_size=batch_size,
                                  shuffle=True, augment=True, input_size=input_size)

    use_valid = has_split(dataset_dir, 'valid')

    if use_valid:
        valid_loader = get_dataloader(dataset_dir, 'valid', batch_size=batch_size,
                                      shuffle=False, augment=False, input_size=input_size)
    else:
        valid_loader = None
        print('[안내] valid 지도가 없어 Early Stopping 없이 끝까지 학습합니다.')

    model = load_model(num_classes).to(device)

    # 드문 클래스(헬멧)를 놓치지 않도록 가중치를 준다
    if use_class_weights:
        weights = compute_class_weights(dataset_dir, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # yolov8n-seg / Mask R-CNN 과 같은 방식으로 학습률을 매끄럽게 줄인다
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = -1.0
    best_epoch = 0
    bad_count = 0
    rows = []

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for e in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        now_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        if use_valid:
            val_loss, miou, iou = validate_one_epoch(model, valid_loader, criterion,
                                                     device, num_classes)
            score = miou
        else:
            val_loss, miou, iou = None, None, None
            score = -train_loss      # 검증이 없으면 train loss 가 낮을수록 좋다고 본다

        is_best = score > best_score

        if is_best:
            best_score = score
            best_epoch = e
            bad_count = 0
            torch.save(model.state_dict(), save_path)
        else:
            bad_count += 1

        rows.append({
            'epoch': e,
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4) if val_loss is not None else '',
            'val_mIoU': round(miou, 4) if miou is not None else '',
            'lr': f'{now_lr:.6f}',
            'best': 'O' if is_best else '',
        })

        text = f'[EPOCH {e}/{epochs}] train {train_loss:.4f}'
        if val_loss is not None:
            text += f'  val {val_loss:.4f}  mIoU {miou:.4f}'
        text += f'  lr {now_lr:.6f}'
        if is_best:
            text += '  <- best'
        print(text)

        if use_valid and bad_count >= patience:
            print(f'{patience} epoch 연속 나아지지 않아 학습을 멈춥니다. '
                  f'(가장 좋았던 epoch {best_epoch})')
            break

    log_path = os.path.join(run_dir, 'results.csv') if run_dir else None

    if log_path:
        save_epoch_log(log_path, rows)
        print(f'epoch 기록 저장 : {log_path}')

    print(f'가장 좋았던 epoch {best_epoch} (mIoU {best_score:.4f}) 의 가중치를 저장했습니다 : {save_path}')

    return {
        'best_epoch': best_epoch,
        'best_score': best_score,
        'epochs_ran': len(rows),
        'log_path': log_path,
    }


def evaluate_model(model, dataset_dir, num_classes, split='test',
                   input_size=INPUT_SIZE, batch_size=8, device=None):
    """
    테스트 데이터로 픽셀 기준 지표를 구한다.

      IoU        : 클래스마다 겹친 정도. mIoU 는 배경을 뺀 평균
      Dice       : 2 x 겹침 / (예측 + 정답). IoU 와 비슷하지만 겹침을 더 후하게 본다
      픽셀정확도 : 전체 픽셀 중 맞힌 비율
      추론시간   : 이미지 한 장 처리 시간
    """
    import time

    device = get_device(device)
    model.to(device)
    model.eval()

    loader = get_dataloader(dataset_dir, split, batch_size=1, shuffle=False,
                            augment=False, input_size=input_size)

    inter = np.zeros(num_classes + 1)
    union = np.zeros(num_classes + 1)
    pred_sum = np.zeros(num_classes + 1)
    true_sum = np.zeros(num_classes + 1)

    correct = 0
    total = 0
    total_time = 0.0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Eval'):
            images = images.to(device)
            targets = targets.to(device)

            start = time.time()
            output = model(images)
            total_time += time.time() - start

            predict = output.argmax(dim=1)

            correct += float((predict == targets).sum())
            total += targets.numel()

            for c in range(num_classes + 1):
                p = predict == c
                t = targets == c
                inter[c] += float((p & t).sum())
                union[c] += float((p | t).sum())
                pred_sum[c] += float(p.sum())
                true_sum[c] += float(t.sum())

    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    dice = np.where(pred_sum + true_sum > 0,
                    2 * inter / np.maximum(pred_sum + true_sum, 1), np.nan)

    infer_ms = (total_time / len(loader)) * 1000 if len(loader) > 0 else 0.0

    scores = {
        'pixel_mIoU': float(np.nanmean(iou[1:])),
        'pixel_Dice': float(np.nanmean(dice[1:])),
        '픽셀정확도': correct / total if total else 0.0,
        '추론시간(ms)': infer_ms,
        'FPS': 1000 / infer_ms if infer_ms > 0 else 0.0,
    }

    # 클래스별 IoU 도 같이 넣어준다
    for c in range(1, num_classes + 1):
        scores[f'IoU_class{c}'] = float(iou[c])

    return scores


def predict_images(model, image_dir, save_dir, class_names,
                   input_size=INPUT_SIZE, device=None):
    """
    테스트 이미지를 추론해서 클래스별로 색을 칠한 결과 이미지를 저장한다.
    U-Net 은 박스가 없으므로 마스크만 그린다.
    """
    import cv2
    from PIL import Image as PILImage
    from torchvision import transforms

    from models.unet_dataset import IMAGE_EXT, NORM_MEAN, NORM_STD
    from utils.visualize import get_color

    device = get_device(device)
    model.to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    files = [f for f in sorted(os.listdir(image_dir)) if f.lower().endswith(IMAGE_EXT)]

    for file_name in tqdm(files, desc='Predict'):
        image = PILImage.open(os.path.join(image_dir, file_name)).convert('RGB')
        origin_w, origin_h = image.size

        resized = image.resize((input_size, input_size), PILImage.BILINEAR)
        x = to_tensor(resized).unsqueeze(0).to(device)

        with torch.no_grad():
            predict = model(x).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        # 원본 크기로 되돌린다 (클래스 번호라 최근접 이웃으로 키운다)
        predict = cv2.resize(predict, (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)

        colored = np.array(image).copy()
        counts = {}

        for value in sorted(set(np.unique(predict).tolist()) - {0}):
            color = np.array(get_color(value - 1), dtype=float)
            area = predict == value
            colored[area] = (colored[area] * 0.5 + color * 0.5).astype(np.uint8)

            name = class_names[value - 1] if value - 1 < len(class_names) else str(value)
            counts[name] = int(area.sum())

        cv2.imwrite(os.path.join(save_dir, file_name),
                    cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

    print(f'결과 이미지 저장 : {save_dir}')
