"""
역할: 지표 딕셔너리를 표로 출력하고 csv 로 저장한다.

step3_eval(평가 결과)과 step5_compare(모델 비교) 두 곳에서 쓰기 때문에 여기 모아두었다.

받는 값 형태 : {모델이름: {지표이름: 값, ...}, ...}
  예) {'yolov8n(detect)': {'mAP50': 0.96, 'Recall': 0.91, ...}}
"""

import csv
import os


def get_all_keys(scores):
    """
    모델마다 나오는 지표가 조금씩 달라서(예: mask mAP) 표에 쓸 항목을 모아 정리한다.
    먼저 나온 순서를 그대로 유지한다.
    """
    keys = []

    for model_scores in scores.values():
        for key in model_scores:
            if key not in keys:
                keys.append(key)

    return keys


def print_table(scores):
    """표를 화면에 출력한다. 없는 항목은 - 로 표시한다."""
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


def save_table_csv(scores, save_path):
    """표를 csv 로 저장한다. (엑셀에서 열 수 있도록 utf-8-sig)"""
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
