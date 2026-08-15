# 안전장구(Vest & Helmet) 탐지 - 라벨 방식과 모델 비교

같은 이미지에 **라벨만 다르게 붙인 데이터셋**으로 모델을 각각 만들고,
어느 쪽이 더 잘 맞추는지 비교하는 미니 프로젝트입니다.

| 모델 | 라벨 | 종류 | 상태 |
|------|------|------|------|
| `yolov8n` | 바운딩 박스 | detection | 데이터 준비됨 |
| `yolov8n-seg` | 폴리곤 | segmentation | 라벨 데이터 준비 중 |
| `Mask R-CNN` (torchvision) | 폴리곤 | segmentation | 라벨 데이터 준비 중 |

---

## 폴더 구조와 역할

최상위 파일 이름 앞의 `step` 번호가 곧 실행 순서입니다.

```text
sasac8_ilbin/
├── main.py                 # 전체 실행 순서 (필요한 줄의 주석을 풀어서 실행)
├── step1_check.py          # 데이터 검사 (Preprocessing 폴더의 검사를 실행)
├── step2_train.py          # 학습 - train_detect(), train_segment(), train_maskrcnn()
├── step3_eval.py           # 평가 - mAP, Precision, Recall, 속도
├── step4_predict.py        # 테스트 이미지 추론 후 결과 이미지 저장
├── step5_compare.py        # 세 모델 비교표/그래프 생성 (프로젝트 목표)
├── config/
│   ├── run_config.py       # 어떤 모델을 실행할지 (RUN_MODELS), 가중치 경로
│   ├── detect.yaml         # detection 데이터셋 경로 + 클래스
│   └── segment.yaml        # segmentation 데이터셋 경로 + 클래스 (경로 미정)
├── Data/                   # 데이터셋 (용량이 커서 git 에 올리지 않음)
│   └── vest-helmet.v1i_roboflow/
├── Preprocessing/          # 학습 전 라벨 검사
│   ├── check_sync.py       # 이미지-라벨 짝, split 별/전체 개수 확인
│   ├── check_format.py     # 라벨 내용이 모델 형식에 맞는지 검사
│   └── check_dataset.py    # 위 검사를 실행하고 문제를 csv 로 저장 (진입점)
├── models/                 # torch 계열 모델 (YOLO 는 라이브러리가 다 해준다)
│   ├── seg_dataset.py      # 폴리곤 라벨 -> 마스크/박스 로 바꾸는 Dataset
│   └── mask_rcnn.py        # Mask R-CNN 생성 / 학습 / 평가 / 결과 그리기
├── utils/
│   ├── metrics.py          # 라벨 읽기(박스/폴리곤), IoU 계산
│   ├── visualize.py        # 샘플 라벨 그리기, 비교 그래프
│   └── table.py            # 지표를 표로 출력 / csv 저장
└── result/                 # 검사 csv, 샘플 라벨 그림, 평가 결과, 추론 결과, 비교표
```

## 설치

```bash
pip install -r requirements.txt
```

- `torch`, `torchvision` 은 GPU 를 쓰려면 https://pytorch.org 에서 CUDA 버전에 맞게 설치하세요.
- `torchmetrics`, `pycocotools` 는 Mask R-CNN 의 mAP 계산에만 씁니다.
  (YOLO 만 쓸 때는 없어도 동작합니다)

## 실행 순서

```bash
# 1. 데이터셋 검사 (라벨 형식, 이미지-라벨 짝, 개수)
python step1_check.py

# 2. 모델 학습
python step2_train.py

# 3. 평가
python step3_eval.py

# 4. 테스트 이미지 추론
python step4_predict.py

# 5. 세 모델 비교
python step5_compare.py
```

## 어떤 모델을 실행할지 정하기

`config/run_config.py` 의 `RUN_MODELS` 한 줄로 정합니다.
`step2_train.py`, `step3_eval.py`, `step4_predict.py` 가 모두 이 값을 봅니다.

```python
# segmentation 라벨이 아직 없는 지금
RUN_MODELS = ['detect']

# 라벨이 들어와서 세 모델을 한 번에 돌릴 때
RUN_MODELS = ['detect', 'segment', 'maskrcnn']
```

| 방식 | 사용법 |
|------|--------|
| 일괄 실행 | `train_all()`, `eval_all()`, `predict_all()` → `RUN_MODELS` 에 있는 모델만 실행 |
| 개별 실행 | `train_detect()`, `eval_one('segment')`, `predict_one('maskrcnn')` 처럼 직접 호출 |

- 학습이 안 끝나 가중치가 없는 모델은 평가·추론에서 **건너뛰고 안내만** 출력합니다.
- **비교(step5)는 세 모델이 모두 학습돼 있을 때만 실행**됩니다.
  하나라도 없으면 어떤 모델의 가중치가 없는지 알려주고 표를 만들지 않습니다.

## 비교하려면 학습 조건을 맞춰야 합니다

비교 실험의 기본은 **알고 싶은 것 하나만 다르게 하고 나머지는 같게** 하는 것입니다.
한쪽만 오래 학습하면 결과 차이가 라벨 덕인지 학습량 덕인지 구분할 수 없습니다.

| | 항목 |
|---|------|
| **같게** | 같은 이미지, 같은 train/valid/test 분할 |
| **같게** | 이미지 크기 (640) |
| **같게** | 학습 종료 규칙 (epoch, patience) |
| **같게** | 평가 split(test), conf/iou 임계값 |
| **다르게** | **라벨 형태 (박스 ↔ 폴리곤)** ← 알고 싶은 것 |

세 모델을 짝지어 보면 각 비교가 무엇을 말해주는지 분명해집니다.

| 비교 쌍 | 다른 점 | 알 수 있는 것 |
|---------|---------|---------------|
| yolov8n ↔ yolov8n-seg | 라벨만 (모델 계열 같음) | 라벨 방식의 효과 |
| yolov8n-seg ↔ Mask R-CNN | 모델만 (라벨 같음) | 모델 구조의 효과 |

**Early Stopping** (`step2_train.py` 의 `EPOCHS=100, PATIENCE=15`)
성능이 15 epoch 연속 나아지지 않으면 남은 epoch 을 건너뛰고 멈춥니다.
`best.pt` 는 항상 가장 좋았던 epoch 의 가중치이므로 중간에 멈춰도 손해가 없습니다.
모델마다 수렴에 걸리는 시간이 다르므로, epoch 수를 억지로 맞추기보다
**각 모델을 수렴시키고 비교표의 `학습시간(분)` 을 함께 보는 방식**을 권합니다.

## 학습 이력이 쌓이는 방식

**재학습해도 이전 결과를 덮어쓰지 않습니다.** 실행할 때마다 `runs/` 아래에 새 폴더가 생깁니다.

```text
runs/detect/vest_helmet_detect/    <- 1회차 (args.yaml, results.csv, 그래프)
runs/detect/vest_helmet_detect2/   <- 2회차
runs/detect/vest_helmet_detect3/   <- 3회차
                    ↓ 학습이 끝나면 best.pt 만 복사
result/weights/detect_best.pt      <- '지금 쓰는 모델' (평가·추론·비교는 항상 여기를 본다)
```

각 폴더에 그때 쓴 설정(`args.yaml`)과 epoch별 지표(`results.csv`)가 통째로 남아 실험 이력이 됩니다.
학습 폴더 하나가 약 20MB라 여러 번 쌓여도 부담이 없습니다.

가중치는 `result/weights/` 로 복사되므로, **폴더 번호가 늘어나도 `config/run_config.py` 를 고칠 필요가 없습니다.**

### 실험 노트 — `result/train_log.csv`

학습이 끝날 때마다 실행 조건과 결과가 한 줄씩 쌓입니다. 폴더를 열어보지 않아도 한눈에 비교됩니다.

```csv
날짜,모델,실행폴더,epochs,patience,imgsz,batch,학습시간(분),mAP50
2026-08-15 15:56,detect,./runs/detect/vest_helmet_detect,50,없음,640,16,95.4,0.9625
```

`mAP50` 은 그 실행의 `results.csv` 에서 가장 좋았던 값을 읽어 적습니다. 비교표의 `학습시간(분)` 도 이 파일에서 가져옵니다.

> **Mask R-CNN은 예외입니다.** `.pth` 파일이 약 170MB로 커서 이력을 쌓지 않고
> `result/weights/mask_rcnn.pth` 하나만 덮어씁니다. 조건과 결과는 `train_log.csv` 에 남습니다.

## 데이터셋 검사 (Preprocessing)

학습 전에 라벨이 제대로 만들어졌는지 확인합니다. 검사 항목은 다음과 같습니다.

| 구분 | 검사 내용 | 문제 유형 |
|------|-----------|-----------|
| 동기화 | 이미지에 짝이 되는 라벨 txt 가 있는가 | `라벨없음` |
| 동기화 | 라벨 txt 에 짝이 되는 이미지가 있는가 | `이미지없음` |
| 동기화 | split 별 / 전체 이미지·라벨 개수 | (개수 출력) |
| 형식 | 열 개수가 모델 형식에 맞는가 (detection 5칸 / segmentation 클래스+짝수 좌표) | `형식오류` |
| 형식 | 폴리곤 꼭짓점이 3개 이상인가 | `꼭짓점부족` |
| 형식 | 좌표가 0~1 로 정규화되어 있는가 | `좌표범위` |
| 형식 | 클래스 번호가 `data.yaml` 범위 안인가 | `클래스번호` |
| 형식 | 라벨 내용이 비어있지 않은가 | `빈파일` |

문제가 있으면 `result/check_detect.csv` (또는 `check_segment.csv`) 로 저장됩니다.
문제가 없으면 csv 는 만들어지지 않고, 어느 경우든 프로그램이 멈추지는 않습니다.

```text
split,image_file,label_file,problem_type,detail
train,a_001.jpg,,라벨없음,짝이 되는 txt 파일 없음
valid,c_003.jpg,c_003.txt,형식오류,3번째 줄 열 4개 (detection 은 5개)
```

### 샘플 라벨 눈으로 확인

개수와 형식이 맞아도 라벨 좌표가 엉뚱한 곳에 찍혀 있을 수 있습니다.
그래서 검사 마지막에 `train` 에서 무작위로 **3장**을 뽑아 라벨을 그린 그림을 저장합니다.

- detection : 바운딩 **박스**로 그림 → `result/label_sample_detect.jpg`
- segmentation : 폴리곤 **외곽선**으로 그림 → `result/label_sample_segment.jpg`
- 클래스마다 색이 다르고 이름이 함께 표시되므로, 클래스 순서가 뒤바뀌었는지도 바로 보입니다.
- 실행할 때마다 다른 이미지가 뽑히므로 여러 번 돌려보면 좋습니다.

기본은 파일 저장만 합니다. 창으로도 띄우고 싶으면 `step1_check.py` 의
`SHOW_SAMPLE = True` 로 바꾸세요. (창을 닫아야 다음 코드가 실행됩니다)
뽑는 장수나 폴더는 `Preprocessing/check_dataset.py` 의
`SAMPLE_COUNT`, `SAMPLE_SPLIT` 에서 바꿉니다.

## Mask R-CNN 은 왜 따로 만드나

YOLO 는 ultralytics 가 데이터 읽기·학습·평가를 전부 해주지만,
torch 계열 모델은 직접 만들어야 합니다.

- `models/seg_dataset.py` : YOLO 폴리곤 라벨(0~1 정규화)을 읽어
  **마스크 이미지**와 **박스 좌표(픽셀)** 로 바꿔 줍니다. 클래스 번호는 0이 배경이라 +1 합니다.
- `models/mask_rcnn.py` : 사전학습 Mask R-CNN 을 불러와 ROI Head 를 우리 클래스 수에 맞게
  교체하고, 학습 루프와 평가를 직접 돌립니다.

평가 지표는 YOLO 와 같은 기준으로 맞췄습니다.
mAP 는 `torchmetrics`(COCO 방식), Precision/Recall/mIoU 는 IoU 0.5 기준으로 직접 계산합니다.

## 비교 항목

| 항목 | 설명 |
|------|------|
| mAP50 / mAP50-95 | 대표적인 객체 탐지 정확도 지표 |
| Precision / Recall / F1 | 정밀도, 재현율과 그 조화평균 |
| mIoU | 정답 박스와 예측 박스가 겹치는 정도의 평균 |
| mask mAP | 마스크 정확도 (yolov8n-seg 만 나옴, 참고용) |
| 추론시간(ms) / FPS | 이미지 한 장 처리 속도 |
| 학습시간(분) / 모델크기(MB) | 학습 비용과 모델 무게 |

학습이 끝난 모델만 표에 들어갑니다. 결과는 `result/compare.csv`, `result/compare.jpg` 입니다.

## segmentation 데이터셋이 들어오면 할 일

1. `Data/` 아래에 segmentation 라벨 데이터셋 폴더를 넣는다.
2. 경로를 **3곳**만 수정한다. (나머지는 여기서 자동으로 만들어진다)
   - `Preprocessing/check_dataset.py` 의 `SEGMENT_DATASET`
   - `config/segment.yaml` 의 `path` (yolov8n-seg 용)
   - `step2_train.py` 의 `SEGMENT_DATASET` (Mask R-CNN 용)
3. `config/run_config.py` 의 `RUN_MODELS` 를 세 모델 전부로 바꾼다.
4. `python step1_check.py` 로 라벨 검사 (`check_segment_dataset()` 주석 해제)
5. `python step2_train.py` → `python step3_eval.py` → `python step4_predict.py` 실행
6. `python step5_compare.py` 로 비교표를 만든다.

## 참고

- 클래스 이름(`names`)은 `config/*.yaml` 과 데이터셋의 `data.yaml` 이 **순서까지** 같아야 합니다.
- 경로는 Ultralytics 특성상 `config/*.yaml` 만 절대경로를 사용합니다.
- 설계 참고 문서: `ref/yolov8n_vest_helmet_implementation_plan.md`
