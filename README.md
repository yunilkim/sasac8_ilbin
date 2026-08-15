# 안전장구(Vest & Helmet) 탐지 - detection vs segmentation 비교

같은 이미지에 **라벨만 다르게 붙인 두 데이터셋**으로 YOLOv8n 모델을 각각 만들고,
어느 쪽이 더 잘 맞추는지 비교하는 미니 프로젝트입니다.

| 구분 | 라벨 | 모델 | 상태 |
|------|------|------|------|
| detection | 바운딩 박스 | `yolov8n.pt` | 데이터 준비됨 |
| segmentation | 폴리곤 | `yolov8n-seg.pt` | 라벨 데이터 준비 중 |

---

## 폴더 구조와 역할

```text
sasac8_ilbin/
├── main.py                 # 전체 실행 순서 (필요한 줄의 주석을 풀어서 실행)
├── train.py                # 학습 - train_detect(), train_segment()
├── eval.py                 # 평가 - mAP, Precision, Recall, 속도
├── predict.py              # 테스트 이미지 추론 후 결과 이미지 저장
├── compare.py              # 두 모델 비교표/그래프 생성 (프로젝트 목표)
├── config/
│   ├── detect.yaml         # detection 데이터셋 경로 + 클래스
│   └── segment.yaml        # segmentation 데이터셋 경로 + 클래스 (경로 미정)
├── Data/                   # 데이터셋 (용량이 커서 git 에 올리지 않음)
│   └── vest-helmet.v1i_roboflow/
├── utils/
│   ├── dataset.py          # 데이터셋 개수/클래스 확인
│   ├── metrics.py          # IoU 계산 (라벨 -> 박스 변환 포함)
│   └── visualize.py        # 라벨 확인, 비교 그래프
└── result/                 # 추론 결과, 비교표, 그래프 저장
```

## 설치

```bash
pip install -r requirements.txt
```

## 실행 순서

```bash
# 1. 데이터셋 확인 (개수, 클래스 이름)
python -m utils.dataset

# 2. detection 모델 학습
python train.py

# 3. 평가
python eval.py

# 4. 테스트 이미지 추론
python predict.py

# 5. segmentation 데이터가 준비되면 두 모델 비교
python compare.py
```

## segmentation 데이터셋이 들어오면 할 일

1. `Data/` 아래에 segmentation 라벨 데이터셋 폴더를 넣는다.
2. `config/segment.yaml` 의 `path` 를 그 폴더 경로로 수정한다.
3. `compare.py` 의 `SEGMENT_TEST_DIR` 도 같은 폴더 기준으로 수정한다.
4. `train.py` 에서 `train_segment()` 주석을 풀고 실행한다.
5. `python compare.py` 로 비교표를 만든다.

## 비교 항목

| 항목 | 설명 |
|------|------|
| mAP50 / mAP50-95 | 대표적인 객체 탐지 정확도 지표 |
| Precision / Recall / F1 | 정밀도, 재현율과 그 조화평균 |
| mIoU | 정답 박스와 예측 박스가 겹치는 정도의 평균 |
| mask mAP | 마스크 정확도 (segmentation 모델만 나옴, 참고용) |
| 추론시간(ms) / FPS | 이미지 한 장 처리 속도 |
| 학습시간(분) / 모델크기(MB) | 학습 비용과 모델 무게 |

## 참고

- 클래스 이름(`names`)은 `config/*.yaml` 과 데이터셋의 `data.yaml` 이 **순서까지** 같아야 합니다.
- 경로는 Ultralytics 특성상 절대경로를 사용합니다. 다른 PC 에서는 `config/*.yaml` 의 `path` 한 줄만 바꾸면 됩니다.
- 설계 참고 문서: `ref/yolov8n_vest_helmet_implementation_plan.md`
