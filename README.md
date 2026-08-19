# sasac8_ilbin

안전장구(반사조끼·안전모) 탐지 미니 프로젝트.
라벨을 만드는 도구와, 그 라벨로 모델을 학습·비교하는 코드를 폴더로 나눠 둔다.

| 폴더 | 하는 일 |
|------|---------|
| [`labeling_studio/`](labeling_studio/) | 원본 이미지에서 segmentation 라벨을 만들고 검수하는 도구 |
| [`model_compare/`](model_compare/) | 만들어진 데이터셋으로 세 모델을 학습·평가·비교 |

두 폴더는 독립적으로 돈다. 각자 `README.md`, `requirements.txt`, `.gitignore` 를 갖고 있으니
쓰려는 쪽 폴더로 들어가서 그 안의 README 를 보면 된다.

```bash
cd labeling_studio    # 라벨을 만들 때
cd model_compare      # 모델을 학습·비교할 때
```

## 결과 미리보기

![세 모델 비교](docs/sample_result.jpg)

수치 비교는 [`model_compare/README.md`](model_compare/README.md#결과) 를 본다.

## 데이터셋

[vest-helmet (Roboflow Universe)](https://universe.roboflow.com/uhhh/vest-helmet-wlbch) 를 쓴다.
원본 이미지는 640×640, 클래스는 `reflective_jacket` 과 `safety_helmet` 둘이다.

원본을 손봐서 세 벌을 둔다.

| 데이터셋 | train | valid | test | 합계 | 라벨 |
|----------|-------|-------|------|------|------|
| `vest-helmet.v1i_roboflow_origin` | 6,681 | 1,427 | 1,418 | 9,526 | 원본 (비교용 보관) |
| `vest-helmet_crop_dedup` | 6,681 | 1,258 | 1,230 | 9,169 | 박스 (detection) |
| `vest-helmet_crop_dedup_seg` | 6,681 | 1,258 | 1,229 | 9,168 | 폴리곤 (segmentation) |

- **crop** — 이미지에 붙어 있던 반사 패딩을 잘라냈다. 그래서 640×640 이 아닌 이미지가 섞인다.
- **dedup** — split 사이에 겹치는 이미지를 뺐다. valid·test 가 원본보다 줄어든 이유다.

segmentation 쪽 test 가 1장 적다. 비교는 양쪽에 다 있는 이미지로만 하므로 결과에는 영향이 없다.

객체 수는 `vest-helmet_crop_dedup_seg` 기준이다.

| split | reflective_jacket | safety_helmet | 합계 |
|-------|-------------------|---------------|------|
| train | 10,419 | 12,661 | 23,080 |
| valid | 1,885 | 2,355 | 4,240 |
| test | 1,886 | 2,405 | 4,291 |

헬멧이 조끼보다 개수는 많지만 하나하나가 작아서, 픽셀로 세면 전체의 3% 밖에 되지 않는다.

`labeling_studio/db`, `labeling_studio/work`, `model_compare/Data` 는 용량이 커서 git 에 없다.
각자 로컬에 두고 쓴다.
