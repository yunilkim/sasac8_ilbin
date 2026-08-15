"""
역할: 라벨 txt 파일의 내용이 모델 형식에 맞게 만들어졌는지 검사한다.

라벨 한 줄의 형식
  detection    : <클래스> <cx> <cy> <w> <h>              -> 항상 5칸
  segmentation : <클래스> <x1> <y1> <x2> <y2> ...        -> 클래스 1칸 + 좌표 짝수칸(꼭짓점 3개 이상)

검사 항목
  1. 열 개수가 모델 형식(detection / segmentation)에 맞는가
  2. 좌표값이 0 ~ 1 사이로 정규화되어 있는가
  3. 클래스 번호가 data.yaml 의 클래스 개수 범위 안인가
  4. 내용이 빈 파일은 아닌가 / 폴리곤 꼭짓점이 3개 미만은 아닌가
"""

import os

import yaml

SPLITS = ['train', 'valid', 'test']
IMAGE_EXT = ('.jpg', '.jpeg', '.png')

# 폴리곤은 최소 삼각형(꼭짓점 3개)은 되어야 영역이 만들어진다
MIN_POINTS = 3


def load_class_count(dataset_dir):
    """데이터셋의 data.yaml 을 읽어 클래스 개수를 돌려준다. 파일이 없으면 None."""
    yaml_path = os.path.join(dataset_dir, 'data.yaml')

    if not os.path.exists(yaml_path):
        print(f'  [주의] data.yaml 이 없어 클래스 번호 검사는 건너뜁니다 : {yaml_path}')
        return None

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return len(data['names'])


def check_line(parts, task, class_count):
    """
    라벨 한 줄을 검사한다.
    문제가 있으면 (문제유형, 설명), 문제가 없으면 None 을 돌려준다.
    task : 'detect' 또는 'segment'
    """
    # 숫자가 아닌 값이 섞여 있는지 먼저 확인
    try:
        numbers = [float(v) for v in parts]
    except ValueError:
        return ('형식오류', '숫자가 아닌 값이 있음')

    class_no = int(numbers[0])
    values = numbers[1:]

    if task == 'detect':
        # detection 은 클래스 + 좌표 4개 = 5칸 고정
        if len(parts) != 5:
            return ('형식오류', f'열 {len(parts)}개 (detection 은 5개)')
    else:
        # segmentation 은 좌표가 (x, y) 짝으로 들어가므로 개수가 짝수여야 한다
        if len(values) % 2 != 0:
            return ('형식오류', f'좌표 {len(values)}개 (x,y 짝이 맞지 않음)')

        if len(values) // 2 < MIN_POINTS:
            return ('꼭짓점부족', f'꼭짓점 {len(values) // 2}개 (최소 {MIN_POINTS}개)')

    # 좌표는 0 ~ 1 로 정규화되어 있어야 한다
    for v in values:
        if v < 0 or v > 1:
            return ('좌표범위', f'0~1 을 벗어난 값 {v}')

    # 클래스 번호는 0 부터 (클래스 개수 - 1) 까지
    if class_count is not None and not (0 <= class_no < class_count):
        return ('클래스번호', f'클래스 {class_no} (0 ~ {class_count - 1} 이어야 함)')

    return None


def check_label_file(txt_path, task, class_count):
    """
    라벨 파일 하나를 검사해서 문제 목록을 돌려준다.
    돌려주는 값 : [(문제유형, 설명), ...]
    """
    problems = []
    line_count = 0

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()

            # 빈 줄은 그냥 넘어간다
            if not parts:
                continue

            line_count += 1
            result = check_line(parts, task, class_count)

            if result is not None:
                problem_type, detail = result
                problems.append((problem_type, f'{line_no}번째 줄 {detail}'))

    # 라벨이 한 줄도 없는 파일 (배경 이미지가 아니라면 라벨링이 빠진 것)
    if line_count == 0:
        problems.append(('빈파일', '내용이 없음'))

    return problems


def check_format(dataset_dir, task):
    """
    데이터셋 전체의 라벨 형식을 검사한다.
    돌려주는 값 : 문제 목록 (csv 로 저장할 딕셔너리 리스트)
    """
    class_count = load_class_count(dataset_dir)
    all_problems = []

    print('[라벨 형식 검사]')

    for split in SPLITS:
        image_dir = os.path.join(dataset_dir, split, 'images')
        label_dir = os.path.join(dataset_dir, split, 'labels')

        if not os.path.isdir(label_dir):
            print(f'  {split:5s} : 폴더 없음')
            continue

        # 문제가 생긴 라벨의 이미지 이름도 같이 적어주기 위해 이미지 목록을 미리 만든다
        image_names = {}
        if os.path.isdir(image_dir):
            for file_name in os.listdir(image_dir):
                if file_name.lower().endswith(IMAGE_EXT):
                    image_names[os.path.splitext(file_name)[0]] = file_name

        split_problems = []

        for label_file in os.listdir(label_dir):
            if not label_file.endswith('.txt'):
                continue

            name = os.path.splitext(label_file)[0]
            problems = check_label_file(os.path.join(label_dir, label_file), task, class_count)

            for problem_type, detail in problems:
                split_problems.append({
                    'split': split,
                    'image_file': image_names.get(name, ''),
                    'label_file': label_file,
                    'problem_type': problem_type,
                    'detail': detail,
                })

        all_problems += split_problems

        state = 'OK' if len(split_problems) == 0 else f'문제 {len(split_problems)}건'
        print(f'  {split:5s} : {state}')

    return all_problems
