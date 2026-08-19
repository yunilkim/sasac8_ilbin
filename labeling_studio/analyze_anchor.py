#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 analyze_anchor.py  --  앵커링 측정 결과 대조
================================================================================

 실행:
     .venv\\Scripts\\python.exe analyze_anchor.py

 같은 30장을 두 번 라벨한 결과를 비교한다.
   1차 anchor_1_blank   빈 화면에서 직접      -> 수동 단가
   2차 anchor_2_seeded  모델 예측을 보고 고침 -> 검수 단가 + 앵커링

--------------------------------------------------------------------------------
 [ 무엇을 앵커링이라 부르는가 ]
--------------------------------------------------------------------------------
 1차에는 없는데 2차에만 있는 박스 = 내가 직접 그릴 때는 안 그렸는데, 모델이
 그려주니 "그럴듯하네" 하고 통과시킨 것. 이게 답지에 들어가면 그 모델의
 체계적 오류가 정답으로 굳는다.

 반대로 1차에만 있는 박스 = 모델이 못 봤고 나도 검수 때 못 살린 것(누락).

 이 측정은 하한이다. 2차를 할 때 1차의 기억이 남아 모델을 1차 답으로 되돌리게
 되므로, 실제 앵커링은 여기 나온 값보다 클 수 있다.
================================================================================
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yolo_dataset_studio as S  # noqa: E402

DATA = os.path.dirname(os.path.abspath(__file__))
A_NAME, B_NAME = "anchor_1_blank", "anchor_2_seeded"
IOU = 0.5


def fmt(sec: float) -> str:
    return S.fmt_hms(sec)


def main() -> int:
    lay = S.Layout(DATA)
    try:
        A = S.Workspace.load(os.path.join(lay.work_dir, A_NAME))
        B = S.Workspace.load(os.path.join(lay.work_dir, B_NAME))
    except OSError as e:
        print(f"[!] 작업공간을 못 읽었습니다: {e}")
        return 1

    pairs = A.worklist_pairs()
    done_a = [k for k in pairs if k in A.records]
    done_b = [k for k in pairs if k in B.records]
    print(f"대상 {len(pairs)}장 · 1차 완료 {len(done_a)} · 2차 완료 {len(done_b)}")
    both = [k for k in pairs if k in A.records and k in B.records]
    if not both:
        print("\n[!] 두 라운드가 모두 끝난 이미지가 없습니다.")
        print("    1차 anchor_1_blank 를 먼저 끝내고 2차 anchor_2_seeded 를 하세요.")
        return 1
    print(f"둘 다 끝난 {len(both)}장으로 비교합니다.\n")

    ta, tb = A.totals(), B.totals()
    oa, ob = ta["ops"], tb["ops"]

    print("=" * 66)
    print("1. 시간과 노동")
    print("=" * 66)
    print(f"{'':<22}{'1차 blank':>16}{'2차 seeded':>16}")
    print(f"{'실측 소요':<22}{fmt(oa.seconds):>16}{fmt(ob.seconds):>16}")
    print(f"{'장당 중앙값':<22}{ta['median_sec']:>14.0f}초{tb['median_sec']:>14.0f}초")
    print(f"{'그린 박스':<22}{oa.drawn:>16}{ob.drawn:>16}")
    print(f"{'지운 박스':<22}{oa.deleted:>16}{ob.deleted:>16}")
    print(f"{'고친 박스':<22}{oa.adjusted:>16}{ob.adjusted:>16}")
    print(f"{'손대지 않은 박스':<22}{oa.kept:>16}{ob.kept:>16}")
    if oa.seconds > 0:
        print(f"\n2차가 1차 대비 {ob.seconds / oa.seconds:.0%} 시간 "
              f"({'빠름' if ob.seconds < oa.seconds else '느림'})")
    n_box_a = sum(len(A.records[k].get("src", [])) for k in both)
    if n_box_a:
        print(f"박스당 실측 단가: 1차 {oa.seconds / n_box_a:.1f}초 "
              f"(합의값 {S.CostModel().draw:.0f}초)")

    print("\n" + "=" * 66)
    print("2. 두 결과가 얼마나 다른가")
    print("=" * 66)
    only_a = only_b = matched = cls_diff = 0
    only_b_conf = []
    per_img = []
    for k in both:
        sp, fn = k
        sa, _ = S.parse_label_file(A.label_path(sp, fn))
        sb, _ = S.parse_label_file(B.label_path(sp, fn))
        # 2차 박스의 출처(C=모델이 그린 것을 유지 / A=사람이 그림)
        srcs_b = (B.records[k].get("src") or [])
        m = S.match_greedy(sa, sb, IOU, same_class=False)
        ma = {i for _v, i, _j in m}
        mb = {j for _v, _i, j in m}
        for v, i, j in m:
            matched += 1
            if sa[i].cls != sb[j].cls:
                cls_diff += 1
        oa_n = len(sa) - len(ma)
        ob_n = len(sb) - len(mb)
        only_a += oa_n
        only_b += ob_n
        for j in range(len(sb)):
            if j not in mb:
                only_b_conf.append(srcs_b[j] if j < len(srcs_b) else "?")
        per_img.append((fn, len(sa), len(sb), oa_n, ob_n))

    tot_a = matched + only_a
    tot_b = matched + only_b
    print(f"1차 박스 {tot_a}개 · 2차 박스 {tot_b}개 · 짝지어짐 {matched}")
    print(f"클래스가 다른 짝 {cls_diff}개")
    print()
    print(f"★ 2차에만 있는 박스 {only_b}개 "
          f"({only_b / max(tot_b, 1):.1%} of 2차)")
    print(f"   = 직접 그릴 땐 안 그렸는데 모델이 그려주니 통과시킨 것 = 앵커링")
    if only_b_conf:
        from collections import Counter
        c = Counter(only_b_conf)
        print(f"   출처 분해: " + " · ".join(f"{k}={v}" for k, v in sorted(c.items())))
        n_model = sum(v for k, v in c.items() if k in ("C", "D"))
        print(f"   그중 모델이 만든 것 {n_model}개 "
              f"({n_model / max(only_b, 1):.0%})")
    print()
    print(f"  1차에만 있는 박스 {only_a}개 "
          f"({only_a / max(tot_a, 1):.1%} of 1차)")
    print(f"   = 모델이 못 봤고 검수 때도 못 살린 것 = 누락")

    print("\n" + "=" * 66)
    print("3. 판단 근거")
    print("=" * 66)
    rate = only_b / max(tot_b, 1)
    if rate < 0.03:
        verdict = ("앵커링이 3% 미만. 모델 시드로 답지를 만들어도 "
                   "무리가 없어 보인다.")
    elif rate < 0.08:
        verdict = ("앵커링이 3~8%. 방법 비교에는 무해하지만 "
                   "절대값 주장(자동승인 precision 등)에는 영향이 있다.")
    else:
        verdict = ("앵커링이 8% 이상. 채점용 답지는 빈 화면에서 만드는 편이 "
                   "낫다. 나머지 대량 구간에만 모델 시드를 쓰자.")
    print(f"  {verdict}")
    print(f"  ※ 이 값은 하한이다. 2차 때 1차의 기억이 남아 실제보다 작게 나온다.")

    print("\n" + "=" * 66)
    print("4. 장별 (2차에만 있는 박스가 많은 순)")
    print("=" * 66)
    per_img.sort(key=lambda t: -t[4])
    print(f"{'파일':<44}{'1차':>5}{'2차':>5}{'1차만':>7}{'2차만':>7}")
    for fn, na, nb, oan, obn in per_img[:12]:
        print(f"{fn[:42]:<44}{na:>5}{nb:>5}{oan:>7}{obn:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
