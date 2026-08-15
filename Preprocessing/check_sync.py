"""
역할: 이미지 파일과 라벨 파일이 서로 짝이 맞는지(동기화) 확인한다.

  - 이미지는 있는데 라벨 txt 가 없는 경우
  - 라벨 txt 는 있는데 이미지가 없는 경우
  - split(train/valid/test) 별 개수와 전체 총 개수

이미지 a_001.jpg 의 짝은 같은 이름의 a_001.txt 이다.
(확장자를 뗀 이름을 '파일 이름'이라고 부른다)
"""

import os

SPLITS = ['train', 'valid', 'test']
IMAGE_EXT = ('.jpg', '.jpeg', '.png')


def get_name_dict(folder, exts):
    """
    폴더 안의 파일을 {확장자 뺀 이름: 전체 파일명} 딕셔너리로 만든다.
    예) {'a_001': 'a_001.jpg', 'a_002': 'a_002.jpg'}
    """
    names = {}

    if not os.path.isdir(folder):
        return names

    for file_name in os.listdir(folder):
        if file_name.lower().endswith(exts):
            name = os.path.splitext(file_name)[0]
            names[name] = file_name

    return names


def check_split_sync(dataset_dir, split):
    """
    split 하나에 대해 이미지와 라벨의 짝을 확인한다.
    돌려주는 값 : (문제 목록, 이미지 개수, 라벨 개수)
    """
    image_dir = os.path.join(dataset_dir, split, 'images')
    label_dir = os.path.join(dataset_dir, split, 'labels')

    images = get_name_dict(image_dir, IMAGE_EXT)
    labels = get_name_dict(label_dir, ('.txt',))

    problems = []

    # 이미지에 짝이 되는 라벨이 없는 경우
    for name in images:
        if name not in labels:
            problems.append({
                'split': split,
                'image_file': images[name],
                'label_file': '',
                'problem_type': '라벨없음',
                'detail': '짝이 되는 txt 파일 없음',
            })

    # 라벨에 짝이 되는 이미지가 없는 경우
    for name in labels:
        if name not in images:
            problems.append({
                'split': split,
                'image_file': '',
                'label_file': labels[name],
                'problem_type': '이미지없음',
                'detail': '짝이 되는 이미지 없음',
            })

    return problems, len(images), len(labels)


def check_sync(dataset_dir):
    """
    데이터셋 전체(train/valid/test)의 동기화를 확인하고 개수를 출력한다.
    돌려주는 값 : (문제 목록, 개수 요약 딕셔너리)
    """
    all_problems = []
    counts = {}

    total_image = 0
    total_label = 0

    print('[동기화 검사]')

    for split in SPLITS:
        image_dir = os.path.join(dataset_dir, split, 'images')

        if not os.path.isdir(image_dir):
            print(f'  {split:5s} : 폴더 없음')
            continue

        problems, image_count, label_count = check_split_sync(dataset_dir, split)

        all_problems += problems
        counts[split] = {'images': image_count, 'labels': label_count}

        total_image += image_count
        total_label += label_count

        # 개수가 같아도 파일 이름이 다를 수 있으므로 짝이 안 맞는 개수도 같이 보여준다
        state = 'OK' if len(problems) == 0 else f'짝 안맞음 {len(problems)}건'
        print(f'  {split:5s} : 이미지 {image_count}개 / 라벨 {label_count}개 -> {state}')

    counts['total'] = {'images': total_image, 'labels': total_label}
    print(f'  전체  : 이미지 {total_image}개 / 라벨 {total_label}개')

    return all_problems, counts
