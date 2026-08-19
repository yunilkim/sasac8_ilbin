"""
지표 딕셔너리를 표로 출력하고 csv 로 저장

step3_eval(평가 결과)과 step5_compare(모델 비교) 두 곳에서 쓰기 때문에 여기 모아둠

받는 값 형태 : {모델이름: {지표이름: 값, ...}, ...}
    예) {'yolov8n(detect)': {'mAP50': 0.96, 'Recall': 0.91, ...}}
"""

import csv
import os


# 모델 지표값 가져오기
def get_all_keys(scores):
    keys = []

    for model_scores in scores.values():
        for key in model_scores:
            if key not in keys:
                keys.append(key)

    return keys


# 표 출력
def print_table(scores):
    names = list(scores.keys())
    keys = get_all_keys(scores)

    print()
    print(f'{"항목":<16}' + ''.join(f'{name:>18}' for name in names))

    for key in keys:
        row = f'{key:<16}'

        for name in names:
            value = scores[name].get(key)
            row += f'{value:>18.4f}' if value is not None else f'{"-":>18}'

        print(row)

    print()


# csv 저장
def save_table_csv(scores, save_path):
    names = list(scores.keys())
    keys = get_all_keys(scores)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['항목'] + names)

        for key in keys:
            row = [key]

            for name in names:
                value = scores[name].get(key)
                row.append(round(value, 4) if value is not None else '')

            writer.writerow(row)

    print(f'표 저장 : {save_path}')
