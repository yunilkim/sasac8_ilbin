"""
역할: train / valid / test 사이에 같은 이미지가 들어있는지(split 간 중복) 검사한다.

왜 중요한가
  train 에서 배운 이미지가 test 에도 있으면, 모델은 답을 외운 채로 시험을 보는 셈이 된다.
  성능이 실제보다 높게 나오므로 학습 전에 반드시 확인해야 한다.

두 가지 방법을 모두 돌려 결과를 비교한다.
  1) 파일명 기준   : 확장자를 뗀 이름이 같으면 중복
                     -> 즉시 끝나지만, Roboflow 는 이름 뒤에 해시를 붙이기 때문에
                        같은 사진이라도 이름이 달라서 거의 못 잡는다.
  2) 이미지 유사도 : 사진을 8x8 흑백으로 줄여 '평균보다 밝은가'를 64비트로 적는다(average hash).
                     크기나 압축이 달라도 같은 사진이면 같은 값이 나온다.
                     다만 8x8 은 너무 거칠어서 '구도만 비슷한 다른 사진'도 같다고 나온다.
                     그래서 해시가 같은 것끼리 64x64 로 다시 비교해서(픽셀 평균 차이, MAD)
                     정말 같은 사진인지 확인한다. 이 두 단계를 거쳐야 오탐이 걸러진다.

split 내부의 중복은 보지 않는다. (train 안의 중복은 성능 평가를 속이지 않는다)
"""

import csv
import os

import cv2
import numpy as np
from tqdm import tqdm

SPLITS = ['train', 'valid', 'test']
IMAGE_EXT = ('.jpg', '.jpeg', '.png')

# 중복이 여러 split 에 걸쳐 있을 때 어느 split 을 남길지 정하는 순서.
# 앞에 있을수록 우선 남긴다. (학습 데이터를 남기고 평가 데이터를 버린다)
KEEP_PRIORITY = ['train', 'valid', 'test']

# average hash 를 만들 때 줄일 크기 (8x8 -> 64비트)
HASH_SIZE = 8

# 해시가 같을 때 진짜 같은 사진인지 확인할 크기와 기준
# 64x64 로 줄여 픽셀 밝기 차이의 평균(MAD)을 구하고, 이 값보다 크면 다른 사진으로 본다.
VERIFY_SIZE = 64
MAD_MAX = 18.0


def list_images(dataset_dir):
    """
    데이터셋의 모든 이미지를 [(split, 파일명, 전체경로), ...] 로 모은다.
    """
    images = []

    for split in SPLITS:
        image_dir = os.path.join(dataset_dir, split, 'images')

        if not os.path.isdir(image_dir):
            continue

        for file_name in sorted(os.listdir(image_dir)):
            if file_name.lower().endswith(IMAGE_EXT):
                images.append((split, file_name, os.path.join(image_dir, file_name)))

    return images


def load_small_gray(image_path, size=VERIFY_SIZE):
    """이미지를 흑백 64x64 로 줄여서 읽는다. 읽지 못하면 None."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def average_hash(small, hash_size=HASH_SIZE):
    """
    줄여둔 이미지를 다시 8x8 로 줄이고, 각 칸이 평균보다 밝으면 1 어두우면 0 으로 적어
    64자리 이진수(정수)를 만든다. 같은 사진이면 크기가 달라도 같은 값이 나온다.
    """
    if small is None:
        return None

    tiny = cv2.resize(small, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    bits = tiny > tiny.mean()

    # True/False 64개를 하나의 정수로 합친다
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)

    return value


def mean_abs_diff(small_a, small_b):
    """두 축소 이미지의 밝기 차이 평균(MAD). 0에 가까울수록 같은 사진이다."""
    return float(np.abs(small_a.astype(float) - small_b.astype(float)).mean())


def verify_group(members, smalls, mad_max=MAD_MAX):
    """
    해시가 같아서 묶인 그룹에서, 첫 번째 이미지와 실제로 비슷한 것만 남긴다.
    8x8 해시는 거칠어서 구도만 비슷한 다른 사진도 같이 묶이기 때문이다.

    돌려주는 값 : 확인을 통과한 멤버 목록
    """
    base = smalls[members[0][2]]

    if base is None:
        return []

    kept = [members[0]]

    for member in members[1:]:
        other = smalls[member[2]]

        if other is None:
            continue

        if mean_abs_diff(base, other) <= mad_max:
            kept.append(member)

    return kept


def group_by_key(images, keys, smalls=None):
    """
    같은 key 를 가진 이미지끼리 묶는다.
    smalls 를 주면 묶인 것들이 정말 같은 사진인지 한 번 더 확인한다.
    split 이 2개 이상 섞인 묶음만 '중복 그룹'으로 돌려준다.

    돌려주는 값 : [[(split, 파일명, 순번), ...], ...]
    """
    buckets = {}

    for index, ((split, file_name, _), key) in enumerate(zip(images, keys)):
        if key is None:
            continue

        buckets.setdefault(key, []).append((split, file_name, index))

    groups = []

    for members in buckets.values():
        if len(members) < 2:
            continue

        # 이미지 유사도로 묶은 경우에는 진짜 같은 사진인지 확인한다
        if smalls is not None:
            members = verify_group(members, smalls)

        # 한 split 안에서만 뭉친 것은 건너뛴다 (split 간 중복만 본다)
        if len({split for split, _, _ in members}) < 2:
            continue

        groups.append(members)

    return groups


def decide_keep(members):
    """
    중복 그룹에서 어느 파일을 남기고 어느 것을 버릴지 정한다.
    KEEP_PRIORITY 가 앞선 split 을 남기고, 나머지 split 의 파일은 버린다.

    돌려주는 값 : [(split, 파일명, 남길지 여부), ...]
    """
    # 그룹에 들어있는 split 중 우선순위가 가장 높은 것
    keep_split = min({split for split, _, _ in members},
                     key=lambda s: KEEP_PRIORITY.index(s) if s in KEEP_PRIORITY else 99)

    return [(split, file_name, split == keep_split) for split, file_name, _ in members]


def count_by_split(images):
    """split 별 이미지 개수를 센다."""
    counts = {split: 0 for split in SPLITS}

    for split, _, _ in images:
        counts[split] = counts.get(split, 0) + 1

    return counts


def check_one_method(images, keys, method_name, smalls=None):
    """
    한 가지 기준으로 중복을 찾고 결과를 정리한다.

    돌려주는 값 :
      {'method': 이름, 'groups': 그룹 수, 'duplicate': 중복 파일 수,
       'drop': 버릴 파일 수, 'before': {split: 수}, 'after': {split: 수},
       'rows': csv 로 저장할 목록}
    """
    groups = group_by_key(images, keys, smalls=smalls)

    before = count_by_split(images)
    after = dict(before)

    rows = []
    duplicate_count = 0
    drop_count = 0

    for group_no, members in enumerate(groups, start=1):
        decided = decide_keep(members)
        duplicate_count += len(decided)

        for split, file_name, keep in decided:
            if not keep:
                after[split] -= 1
                drop_count += 1

            rows.append({
                '기준': method_name,
                '그룹번호': group_no,
                'split': split,
                '파일명': file_name,
                '처리': '유지' if keep else '삭제대상',
            })

    return {
        'method': method_name,
        'groups': len(groups),
        'duplicate': duplicate_count,
        'drop': drop_count,
        'before': before,
        'after': after,
        'rows': rows,
    }


def print_result(result):
    """검사 결과 하나를 보기 좋게 출력한다."""
    before, after = result['before'], result['after']
    total_before = sum(before.values())
    total_after = sum(after.values())

    print(f"[{result['method']}]")
    print(f"  split 간 중복 그룹 : {result['groups']}개")
    print(f"  중복에 걸린 파일   : {result['duplicate']}개 (그 중 삭제 대상 {result['drop']}개)")
    print('  split       현재      중복처리 후')

    for split in SPLITS:
        print(f'  {split:<8} {before.get(split, 0):>8} {after.get(split, 0):>14}')

    print(f'  {"합계":<7} {total_before:>8} {total_after:>14}')


def print_sample_files(result, limit=10):
    """중복으로 잡힌 파일 이름을 몇 개만 보여준다."""
    drops = [row for row in result['rows'] if row['처리'] == '삭제대상']

    if not drops:
        return

    print(f"  삭제 대상 파일 예시 (상위 {min(limit, len(drops))}개)")

    for row in drops[:limit]:
        print(f"    [{row['split']}] {row['파일명']}")


def save_duplicate_csv(results, save_path):
    """두 기준의 중복 목록을 하나의 csv 로 저장한다."""
    rows = []
    for result in results:
        rows += result['rows']

    if not rows:
        return None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['기준', '그룹번호', 'split', '파일명', '처리'])
        writer.writeheader()
        writer.writerows(rows)

    print(f'중복 목록 저장 : {save_path}')

    return save_path


def check_duplicate(dataset_dir, save_path=None):
    """
    데이터셋의 split 간 중복을 두 가지 기준으로 검사하고 결과를 비교한다.
    돌려주는 값 : [파일명 기준 결과, 이미지 유사도 기준 결과]
    """
    print(f'===== 중복 검사 : {dataset_dir} =====')

    if not os.path.isdir(dataset_dir):
        print('데이터셋 폴더가 없습니다.')
        return []

    images = list_images(dataset_dir)

    if not images:
        print('이미지를 찾지 못했습니다.')
        return []

    print(f'이미지 {len(images)}장을 검사합니다.')

    # 1) 파일명 기준 : 확장자를 뗀 이름
    name_keys = [os.path.splitext(file_name)[0] for _, file_name, _ in images]
    name_result = check_one_method(images, name_keys, '파일명 기준')

    # 2) 이미지 유사도 기준 (이미지를 전부 읽어야 해서 시간이 걸린다)
    #    한 번 읽어 64x64 로 줄여두고, 그것으로 8x8 해시도 만들고 확인용 비교도 한다
    smalls = []
    hash_keys = []

    for _, _, path in tqdm(images, desc='이미지 해시 계산'):
        small = load_small_gray(path)
        smalls.append(small)
        hash_keys.append(average_hash(small))

    failed = hash_keys.count(None)
    if failed:
        print(f'[주의] 이미지 {failed}장을 읽지 못해 유사도 검사에서 제외했습니다.')

    hash_result = check_one_method(images, hash_keys, '이미지 유사도 기준', smalls=smalls)

    print()
    print_result(name_result)
    print_sample_files(name_result)
    print()
    print_result(hash_result)
    print_sample_files(hash_result)

    # 두 방법 비교
    print()
    print('[두 기준 비교]')
    print(f"  파일명 기준     : 중복 그룹 {name_result['groups']:>4}개 / 삭제 대상 {name_result['drop']:>5}개")
    print(f"  이미지 유사도   : 중복 그룹 {hash_result['groups']:>4}개 / 삭제 대상 {hash_result['drop']:>5}개")

    only_hash = hash_result['drop'] - name_result['drop']
    if only_hash > 0:
        print(f'  -> 파일명만 봤다면 {only_hash}개를 놓쳤을 것입니다.')

    if save_path:
        save_duplicate_csv([name_result, hash_result], save_path)

    return [name_result, hash_result]


if __name__ == '__main__':
    # 원본(중복 제거 전) 데이터셋을 검사해 본다
    check_duplicate('./Data/vest-helmet.v1i_roboflow_origin',
                    save_path='./result/duplicate_origin.csv')
