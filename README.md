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

## 데이터

`labeling_studio/db`, `labeling_studio/work`, `model_compare/Data` 는 용량이 커서 git 에 없다.
각자 로컬에 두고 쓴다.
