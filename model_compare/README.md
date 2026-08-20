# model_compare — 세 모델 학습·평가·비교

프로젝트 개요와 데이터셋은 [저장소 루트 README](../README.md) 를 본다.

## 결과

### 박스 기준

| 항목 | yolov8n | yolov8n-seg |
|------|---------|-------------|
| mAP50 | 0.9516 | 0.9457 |
| mAP50-95 | 0.7593 | 0.7723 |
| Precision | 0.9231 | 0.9190 |
| Recall | 0.9067 | 0.8909 |
| F1 | 0.9149 | 0.9048 |
| box_mIoU | 0.8092 | 0.8125 |

라벨을 폴리곤으로 바꿔도 박스 성능은 거의 그대로다.
찾는 개수는 detect 가 조금 앞서고(Recall) 박스를 정밀하게 치는 건 seg 가 조금 앞서는데
(mAP50-95, box_mIoU) 둘 다 1~2%p 안쪽이라 두 모델의 성능은 동일한것으로 봐도 무방하다.

다만 폴리곤 라벨을 쓰는 이유는 박스를 더 잘 치기 위해서가 아니라 픽셀 단위 결과를 얻기 위해서다.
만드는 비용은 박스보다 훨씬 크니, 픽셀이 필요 없으면 박스 라벨로 충분하다.

### 픽셀 기준

| 항목 | yolov8n-seg | U-Net |
|------|-------------|-------|
| pixel_mIoU | 0.7912 | 0.7785 |
| pixel_Dice | 0.8831 | 0.8753 |
| pixel_accuracy | 0.9591 | 0.9578 |
| IoU (reflective_jacket) | 0.7620 | 0.7589 |
| IoU (safety_helmet) | 0.8204 | 0.7981 |

`mask_mAP50` 은 yolov8n-seg 만 나온다 (0.9127).
U-Net 은 객체를 구분하지 않아서 이 값을 낼 수 없다. 시맨틱 분할로 바꾸면 뭘 잃는지 보여주는 자리다.

두 모델 다 헬멧 IoU 가 조끼보다 높다. 헬멧은 전체 픽셀의 3% 뿐이지만 모양과 색이 뚜렷해서
오히려 잘 잡힌다.

### 비용

| 항목 | yolov8n | yolov8n-seg | U-Net |
|------|---------|-------------|-------|
| 추론시간(ms) | 5.73 | 7.06 | 9.74 |
| FPS | 174.6 | 141.7 | 102.7 |
| 모델크기(MB) | 5.96 | 6.47 | 93.34 |

### 다른 데이터셋과 비교

Roboflow Universe 의 안전장구 데이터셋 7개와 견줘 본다.
비교 대상이 전부 detection 이라 yolov8n 만 놓고 본다.

| 지표 | 유사 7개 평균 | 최대 | 최소 | yolov8n |
|------|--------------|------|------|---------|
| mAP@50 | 91.59 | 97.50 | 87.70 | **95.16** |
| Precision | 91.73 | 96.50 | 85.20 | **92.31** |
| Recall | 87.01 | 94.00 | 81.40 | **90.67** |

세 지표 모두 평균 위, 최대 아래다. mAP 는 8개 중 2위다.
Precision 은 평균과 거의 같은데(+0.58%p) Recall 이 평균보다 3.66%p 높다.
놓치는 걸 줄인 쪽에서 점수를 벌었다.

**이 문제 영역에서 통상 나오는 범위 안에 들어왔다는 확인 용도로만 사용한다..**

| 데이터셋 | mAP@50 | Precision | Recall |
|----------|--------|-----------|--------|
| [robot-vest](https://universe.roboflow.com/politeknik-pertanian-negeri-samarinda/robot-vest) | 97.5 | 96.5 | 94.0 |
| [helmet-fkbjb](https://universe.roboflow.com/cuadrados/helmet-fkbjb) | 94.2 | 94.1 | 91.2 |
| [gsafety-helmet-2](https://universe.roboflow.com/imageprocessing-project/gsafety-helmet-2) | 92.3 | 92.2 | 88.9 |
| [helmet-6zabc](https://universe.roboflow.com/los-workspace-kfj0b/helmet-6zabc) | 92.1 | 91.6 | 87.2 |
| [vest-zsuuy](https://universe.roboflow.com/ppe1-ppe2/vest-zsuuy) | 89.5 | 85.2 | 84.7 |
| [gsafety-helmet](https://universe.roboflow.com/imageprocessing-project/gsafety-helmet) | 87.8 | 91.6 | 81.4 |
| [helmet-s8xji](https://universe.roboflow.com/gtr34-bu1er/helmet-s8xji) | 87.7 | 90.9 | 81.7 |

### 정리

박스만 놓고 보면 yolov8n 과 yolov8n-seg 는 사실상 동률이다.
**seg 를 쓰는 이유는 박스 성능이 아니라 픽셀 결과를 낼 수 있다는 것 하나다.**

U-Net 은 yolov8n-seg 보다 14배 무겁고 1.4배 느린데 픽셀 정확도는 오히려 낮다.
탐지로 대상을 먼저 좁히고 그 안에서만 마스크를 그리는 구조가,
화면 전체를 한 번에 분류하는 구조보다 이 데이터에서 유리했던 것으로 보인다.

| 필요한 것 | 고를 모델 | 근거 |
|-----------|-----------|------|
| 착용 여부만 판정 | `yolov8n` | 174.6 FPS 로 가장 빠르고 라벨링 비용도 싸다 |
| 착용 여부 + 영역 | `yolov8n-seg` | 박스 성능은 그대로면서 픽셀까지 낸다 |
| 픽셀만 필요 | `yolov8n-seg` | U-Net 보다 14배 가볍고 정확도도 높다 |

원본 수치는 `result/compare.csv`, 그래프는 `result/compare_box.jpg` 와
`result/compare_pixel.jpg` 에 있다.

### 이 결과의 조건

클래스 2개, 대상이 크고 색이 뚜렷하고, 학습 6,681장, 입력은 640×640 으로 통일했다.
클래스가 많아지거나 대상이 작아지면 결과가 달라질 수 있다.
데이터셋 구성은 [저장소 루트 README](../README.md#데이터셋) 를 본다.

체급도 같지 않다. U-Net 은 ResNet34 인코더라 93MB 인데 yolov8n-seg 는 6.5MB 다.
모델 크기를 맞춘 비교가 아니라, 두 방식의 기본 구성을 그대로 놓고 견준 것이다.
U-Net 의 인코더를 더 가벼운 것으로 바꿔 크기를 맞추면 결과가 달라질 수 있다.

---

## 주의사항

**반드시 `model_compare/` 안에서 실행한다.**
코드가 `./Data/...`, `./result/...`, `./config/detect.yaml` 처럼 전부 상대경로를 쓴다.
저장소 루트에서 돌리면 데이터도 가중치도 못 찾는다.

**학습시간(분)은 비교하면 안 된다.**
세 모델을 각각 다른 PC 에서 돌렸다. `compare.csv` 에 값은 있지만 하드웨어가 달라서 의미가 없다.
FPS 는 비교할 때 한 PC 에서 다시 재는 값이라 그대로 봐도 된다.

**`eval_scores.csv` 와 `compare.csv` 는 실행할 때마다 통째로 덮어쓴다.**
세 모델 표를 남기려면 `RUN_MODELS` 에 셋을 다 넣고 돌려야 한다.

**`train_log.csv` 의 U-Net 줄은 칸 의미가 다르다.**
박스가 없어서 mAP 를 못 내니까 `mAP50` 칸에 mIoU 가 들어간다.
`batch` 칸에도 YOLO 값이 그대로 찍히니 실제 값은 `step2_train.py` 의 `UNET_BATCH` 를 봐야 한다.

**이어 학습(resume)은 안 된다.**
U-Net 은 가중치만 저장해서 optimizer 상태와 학습률 위치가 복원되지 않는다.
yolov8n-seg 의 `resume=True` 는 중간에 끊긴 학습용이라 정상 종료된 학습에는 못 쓴다.
더 돌리려면 epoch 상한을 늘려서 처음부터 다시 학습해야 한다.

**`config/*.yaml` 의 `path` 는 절대경로다.**
Ultralytics 가 상대경로를 `DATASETS_DIR` 기준으로 해석해서 절대경로를 썼다.
PC 를 바꾸거나 폴더를 옮기면 이 값을 다시 써야 한다.

**CUDA 13 빌드는 Pascal(GTX 10xx) 이하를 지원하지 않는다.**
`python -c "import torch; print(torch.cuda.get_arch_list())"` 로 자기 GPU 의 `sm_XX` 가
목록에 있는지 확인한다. 없으면 학습 중에
`unable to find an engine to execute this computation` 이 난다.

**GPU 메모리에 맞춰 `UNET_BATCH` 를 조정한다.**
기본값 6 은 8GB 기준이다. 4GB 면 3 으로 줄인다.

**`main.py` 는 주석 상태를 보고 실행한다.**
`train_all()` 이 열려 있으면 세 모델을 처음부터 다시 학습한다. 단계별 파일을 따로 돌리는 게 안전하다.

**파일 이름에 한글이나 공백을 쓰지 않는다.**
Windows 에서 `cv2.imread` 와 `cv2.imwrite` 가 이런 경로를 열지 못한다.
step6·step7 의 격자 만들기와 U-Net 결과 저장이 조용히 실패한다.
외부 이미지를 step7 에 넣을 때 특히 걸리므로, 넣기 전에 영문 이름으로 바꾼다.

---

## 설치

```bash
cd model_compare
pip install -r requirements.txt
```

torch 와 torchvision 은 먼저 https://pytorch.org 에서 CUDA 버전에 맞게 설치한다.
`requirements.txt` 에 넣으면 CPU 빌드가 깔리면서 기존 CUDA 빌드를 덮어쓴다.

U-Net 은 torchvision 의 ResNet34 만 가져다 직접 구현했으니 따로 설치할 게 없다.
`torchmetrics` 와 `pycocotools` 는 Mask R-CNN 전용이라 주석 처리해 뒀다.

---

## 상세 내용

### 폴더 구조

파일 이름 앞의 `step` 번호가 실행 순서다.
모든 경로가 이 폴더 기준 상대경로라서 **`model_compare/` 안에서 실행해야 한다.**

```text
model_compare/
├── .gitignore              # 이 폴더의 산출물 제외 규칙 (저장소 공통은 루트에)
├── main.py                 # 전체 실행 순서 (필요한 줄의 주석을 풀어서 실행)
├── step1_check.py          # 데이터 검사 (Preprocessing 폴더의 검사를 실행)
├── step2_train.py          # 학습 - train_detect(), train_segment(), train_unet()
├── step3_eval.py           # 평가 - mAP(박스 모델), pixel mIoU/Dice(U-Net)
├── step4_predict.py        # 테스트 이미지 추론 후 결과 이미지 저장
├── step5_compare.py        # 세 모델 비교표/그래프 생성 (프로젝트 목표)
├── step6_sum_result.py     # 원본 + 세 모델 결과를 한 장으로 합치기
├── step7_external.py       # 외부 이미지로 추론 + 합치기 (폴더를 이 파일에서 정한다)
├── config/
│   ├── run_config.py       # 어떤 모델을 실행할지 (RUN_MODELS), 가중치 경로
│   ├── detect.yaml         # detection 데이터셋 경로 + 클래스
│   └── segment.yaml        # segmentation 데이터셋 경로 + 클래스
├── Data/                   # 데이터셋 (용량이 커서 git 에 올리지 않음)
│   ├── vest-helmet.v1i_roboflow_origin/  # 원본 (중복 제거 전, 비교용)
│   ├── vest-helmet_crop_dedup/           # detection 라벨 (split 간 중복 제거본)
│   └── vest-helmet_crop_dedup_seg/       # segmentation 라벨 (SAM 으로 생성)
├── Preprocessing/          # 학습 전 검사와 변환
│   ├── check_sync.py       # 이미지-라벨 짝, split 별/전체 개수 확인
│   ├── check_format.py     # 라벨 내용이 모델 형식에 맞는지 검사
│   ├── check_duplicate.py  # split 간 중복 이미지 검사 (파일명 / 이미지 유사도)
│   ├── convert_to_maskrcnn.py  # YOLO seg 폴리곤 -> 객체별 마스크(npz) 변환
│   ├── convert_to_unet.py  # 객체별 마스크(npz) -> 클래스 지도(png) 변환
│   └── check_dataset.py    # 위 검사를 실행하고 문제를 csv 로 저장 (진입점)
├── models/                 # torch 계열 모델 (YOLO 는 라이브러리가 다 해준다)
│   ├── unet.py             # U-Net 생성 / 학습 / 평가 / 결과 그리기
│   ├── unet_dataset.py     # 클래스 지도(png)를 읽는 Dataset
│   ├── seg_dataset.py      # 객체별 마스크(npz)를 읽는 Dataset (Mask R-CNN 용)
│   └── mask_rcnn.py        # Mask R-CNN 생성 / 학습 / 평가 (현재 미사용)
├── utils/
│   ├── metrics.py          # 라벨 읽기(박스/폴리곤), IoU 계산
│   ├── visualize.py        # 샘플 라벨 그리기, 비교 그래프
│   ├── table.py            # 지표를 표로 출력 / csv 저장
│   └── run_info.py         # 실행 전 설정 표시 / 폴더명 확인
└── result/                 # 검사 csv, 샘플 라벨 그림, 평가 결과, 추론 결과, 비교표
```

### 실행 순서

```bash
cd model_compare           # 모든 경로가 이 폴더 기준이다

python step1_check.py      # 1. 데이터셋 검사
python step2_train.py      # 2. 학습
python step3_eval.py       # 3. 평가
python step4_predict.py    # 4. 테스트 이미지 추론
python step5_compare.py    # 5. 세 모델 비교
python step6_sum_result.py # 6. 원본 + 세 결과를 한 장으로 합치기
python step7_external.py   # 7. 외부 이미지로 추론 + 합치기 (선택)
```

가중치가 없는 모델은 평가와 추론에서 건너뛰고 안내만 찍는다.
비교(step5)는 세 모델이 다 학습돼 있을 때만 돈다.
step5 는 평가를 자기가 다시 계산하니까 step3 을 건너뛰어도 된다.

step6 은 step4 의 결과 이미지를 원본과 나란히 붙여 `result/sum_result/` 에 저장한다.
수치로는 안 보이는 차이 - 어떤 객체를 놓쳤는지, 경계가 어떻게 다른지 - 를 눈으로 본다.

step1~6 은 이 프로젝트 데이터셋을 대상으로 돈다.
step7 은 그 흐름과 별개로 **밖에서 가져온 이미지**에 학습된 모델을 그대로 적용한다.
학습에 쓰지 않은 사진에서도 통하는지 보는 용도라 라벨이 없어도 된다.

```python
# step7_external.py 위쪽. 여기만 고치면 된다
SOURCE_DIR = './Data/external'      # 외부 이미지를 넣어둘 폴더
SAVE_ROOT  = './result/external'    # 결과가 전부 이 아래에 쌓인다
```

```text
result/external/
├── predict_detect/     yolov8n 결과
├── predict_seg/        yolov8n-seg 결과
├── predict_unet/       U-Net 결과
└── sum_result/         원본 + 셋을 한 장으로
```

추론과 격자 그리기는 step4·step6 의 함수를 그대로 빌려 쓰고, **폴더만 step7 이 정한다.**
YOLO 추론만 직접 부르는데, `step4.predict_images` 는 저장 위치가 `'./result'` 로 고정이라
step7 이 폴더를 지정할 수 없어서다.

### 실행할 모델 정하기

`config/run_config.py` 의 `RUN_MODELS` 한 줄로 정한다.
step2, step3, step4 가 모두 이 값을 본다.

```python
RUN_MODELS = ['detect', 'segment', 'unet']   # 세 모델을 한 번에
RUN_MODELS = ['unet']                        # 하나만 다시
```

일괄로 돌릴 때는 `train_all()`, `eval_all()`, `predict_all()` 을 쓰고,
하나만 돌릴 때는 `train_detect()`, `eval_one('segment')`, `predict_one('unet')` 처럼 직접 부른다.

학습과 평가, 추론은 시작할 때 어떤 설정으로 도는지 화면에 찍어준다.
`CONFIRM_RUN = True` 로 바꾸면 만들어질 폴더명까지 확인받는다.
기본이 `False` 인 건 백그라운드로 돌릴 때 입력을 기다리다 멈추기 때문이다.

### 데이터 준비

학습 전에 라벨이 제대로 만들어졌는지 확인한다.

| 구분 | 검사 내용 | 문제 유형 |
|------|-----------|-----------|
| 동기화 | 이미지에 짝이 되는 라벨 txt 가 있는가 | `라벨없음` |
| 동기화 | 라벨 txt 에 짝이 되는 이미지가 있는가 | `이미지없음` |
| 형식 | 열 개수가 모델 형식에 맞는가 | `형식오류` |
| 형식 | 폴리곤 꼭짓점이 3개 이상인가 | `꼭짓점부족` |
| 형식 | 좌표가 0~1 로 정규화되어 있는가 | `좌표범위` |
| 형식 | 클래스 번호가 `data.yaml` 범위 안인가 | `클래스번호` |
| 형식 | 라벨 내용이 비어있지 않은가 | `빈파일` |

문제가 있을 때만 `result/check_detect.csv`(또는 `check_segment.csv`)가 만들어진다.
어느 쪽이든 프로그램이 멈추지는 않는다.

**split 간 중복 검사.**
train 에서 배운 이미지가 test 에도 있으면 답을 외운 채로 시험을 보는 셈이 된다.
Roboflow 가 파일명에 해시를 붙여서 이름 비교로는 중복을 못 잡으니 이미지 내용으로 본다.
8x8 average hash 로 후보를 묶고 64x64 로 다시 확인하는 2단계다.
1단계만 쓰면 구도만 비슷한 다른 사진도 중복으로 잡힌다.

split 안쪽 중복은 보지 않는다. train 안의 중복은 성능 평가를 속이지 않는다.

```bash
python -c "from Preprocessing.check_dataset import check_current_duplicate; check_current_duplicate()"
```

**샘플 라벨 확인.**
개수와 형식이 맞아도 좌표가 엉뚱한 데 찍혀 있을 수 있다.
검사 마지막에 train 에서 무작위 3장을 뽑아 라벨을 그린 그림을 저장한다.
클래스마다 색이 달라서 클래스 순서가 뒤바뀐 것도 바로 보인다.
창으로 띄우려면 `step1_check.py` 의 `SHOW_SAMPLE = True` 로 바꾼다.

**라벨 변환.**
yolov8n-seg 는 폴리곤 txt 를 그대로 읽는다.
U-Net 은 픽셀마다 클래스 번호가 적힌 지도가 필요해서 미리 변환해 둔다.

```bash
python -m Preprocessing.convert_to_maskrcnn   # 1단계
python -m Preprocessing.convert_to_unet       # 2단계
```

```text
<split>/labels/<이름>.txt      YOLO seg 폴리곤 (원본)
            ↓ 1단계
<split>/masks/<이름>.npz       객체별 마스크 (객체수,H,W) / classes / boxes
            ↓ 2단계
<split>/semantic/<이름>.png    클래스 지도 (0=배경, 1=조끼, 2=헬멧)
```

객체별 마스크를 거치는 건 인스턴스 정보가 남아야 Mask R-CNN 도 쓸 수 있어서다.
반대 방향(지도에서 객체별로)은 복원이 안 된다.

이 데이터는 작업자들이 서로 겹쳐 있다. 2단계에서 면적이 큰 것부터 그려서 작은 객체가 위에 남게 한다.
`step5_compare.py` 의 `yolo_pixel_scores()` 도 같은 규칙을 쓴다. 정답과 예측이 같은 방식으로 눌려야 한다.

### 학습 조건 맞추기

비교 실험은 알고 싶은 것 하나만 다르게 하고 나머지는 같게 해야 한다.

| | 항목 |
|---|------|
| 같게 | 같은 이미지, 같은 train/valid/test 분할 |
| 같게 | 이미지 크기 (640) |
| 같게 | 학습 종료 규칙 (`EPOCHS=100`, `PATIENCE=15`) |
| 같게 | 평가 split(test), conf/iou 임계값 |
| 같게 | 사전학습 가중치 사용 (YOLO=COCO, U-Net=ImageNet) |
| 같게 | 증강 수준 (좌우반전, 밝기·채도·색조) |
| 다르게 | 라벨 형태와 모델 구조 |

Early Stopping 은 15 epoch 연속 나아지지 않으면 멈춘다.
저장되는 건 항상 가장 좋았던 epoch 의 가중치라서 중간에 멈춰도 손해가 없다.
epoch 수를 억지로 맞추지 않고 각자 수렴할 때까지 돌렸다.
detect 와 segment 는 100 을 채웠고 U-Net 은 96 에서 멈췄다.

### U-Net 구현

YOLO 는 ultralytics 가 데이터 읽기부터 평가까지 다 해주지만 torch 계열은 직접 만들어야 한다.

- `models/unet_dataset.py` : 클래스 지도(png)를 읽어서 이미지와 정답 지도를 같은 크기로 맞춘다.
  지도는 클래스 번호라 최근접 이웃으로 리사이즈한다. 보간하면 없던 클래스가 생긴다.
- `models/unet.py` : ResNet34 를 수축 경로로 쓰는 U-Net 을 만들고 학습 루프와 평가를 직접 돌린다.

| 선택 | 값 | 이유 |
|------|-----|------|
| 인코더 | ImageNet 사전학습 ResNet34 | yolov8n-seg 의 COCO 사전학습과 조건을 맞추려고 |
| imgsz | 640 | YOLO 와 동일 |
| batch | 6 | 8GB GPU 기준 |
| lr | 0.0015 | Adam 이라 SGD 보다 작게 |
| 학습률 스케줄 | CosineAnnealingLR | 계단식은 계단 사이 정체를 수렴으로 착각해 Early Stopping 이 일찍 걸림 |
| 판정 기준 | valid mIoU | 픽셀 정확도는 배경이 83% 라 변별력이 없음 |
| 클래스 가중치 | `1/sqrt(빈도)` | 배경 83% / 조끼 13% / 헬멧 3% 불균형 보정 (`1/빈도` 는 너무 극단적) |

epoch 별 기록은 `runs/unet/<이름>/results.csv` 에 클래스별 IoU 까지 남는다.
평균만 보면 헬멧을 못 잡고 있어도 눈치채기 어렵다.

### 학습 이력

재학습해도 이전 결과를 덮어쓰지 않는다. 실행할 때마다 `runs/` 아래에 새 폴더가 생긴다.

```text
runs/detect/vest_helmet_detect/     <- 1회차 (args.yaml, results.csv, 그래프)
runs/detect/vest_helmet_detect-2/   <- 2회차
                    ↓ 학습이 끝나면 best.pt 만 복사
result/weights/detect_best.pt       <- '지금 쓰는 모델' (평가·추론·비교는 항상 여기를 본다)
```

평가와 추론, 비교는 `result/weights/` 고정 경로만 보니까 폴더 번호가 늘어나도 설정을 고칠 필요가 없다.

U-Net 은 복사 단계가 없다. `runs/` 에는 `results.csv` 만 남고
best 가중치는 학습 중에 `result/weights/unet_best.pth` 로 바로 저장된다.

실행 조건과 결과는 `result/train_log.csv` 에 한 줄씩 쌓인다.

```csv
날짜,모델,실행폴더,epochs,patience,imgsz,batch,학습시간(분),mAP50
2026-08-19 3:53,segment,...\runs\segment\vest_helmet_seg-4,100,15,640,16,279.7,0.9557
```

### 비교 항목

U-Net 은 박스를 안 내고 yolov8n 은 픽셀을 안 칠한다.
한 지표로 셋을 줄 세울 수 없어서 두 갈래로 나눴다.

| 갈래 | 항목 | 대상 |
|------|------|------|
| 박스 | mAP50 / mAP50-95 / Precision / Recall / F1 | yolov8n, yolov8n-seg |
| 박스 | `box_mIoU` (정답 박스와 예측 박스가 겹치는 정도) | yolov8n, yolov8n-seg |
| 박스 | `mask_mAP` (마스크 정확도, 참고용) | yolov8n-seg 만 |
| 픽셀 | `pixel_mIoU` (클래스마다 겹친 픽셀 비율의 평균) | yolov8n-seg, U-Net |
| 픽셀 | `pixel_Dice` (픽셀로 계산한 F1) | yolov8n-seg, U-Net |
| 픽셀 | `pixel_accuracy`, `IoU_class1/2` | yolov8n-seg, U-Net |
| 공통 | 추론시간(ms) / FPS / 모델크기(MB) | 셋 다 |

yolov8n-seg 는 객체마다 마스크를 주고 U-Net 은 지도 한 장을 준다. 그대로는 비교가 안 된다.
그래서 yolov8n-seg 의 마스크를 지도 한 장으로 눌러서 맞췄다
(`step5_compare.py` 의 `yolo_pixel_scores()`).

`box_mIoU` 와 `pixel_mIoU` 는 이름이 비슷하지만 박스 겹침과 픽셀 겹침으로 전혀 다른 값이다.
`pixel_Dice` 는 `pixel_mIoU` 에서 계산되는 값이라(`2·IoU/(1+IoU)`) 순위를 바꾸지 않지만
시맨틱 분할에서 관례적으로 같이 싣는다.

그래프도 `compare_box.jpg` 와 `compare_pixel.jpg` 두 장으로 나눈다.
한 장에 넣으면 서로 없는 칸이 많아서 비교가 안 된다.

### 처음부터 다시 돌리기

`Data/` 폴더가 준비되면 `config/` 폴더내의 경로를 변경 후 준비되면 순서대로 실행한다.

1. `python step1_check.py` 로 라벨 검사
2. `python -m Preprocessing.convert_to_maskrcnn` → `python -m Preprocessing.convert_to_unet`
3. `config/run_config.py` 의 `RUN_MODELS` 를 `['detect', 'segment', 'unet']` 으로 둔다
4. `python step2_train.py` → `python step3_eval.py` → `python step4_predict.py`
5. `python step5_compare.py`

데이터셋 경로를 바꿀 일이 생기면 3곳만 고치면 된다.
`Preprocessing/check_dataset.py` 의 `SEGMENT_DATASET`, `config/segment.yaml` 의 `path`,
`step2_train.py` 의 `SEGMENT_DATASET`.

### 참고

- 클래스 이름(`names`)은 `config/*.yaml` 과 데이터셋의 `data.yaml` 이 순서까지 같아야 한다.
