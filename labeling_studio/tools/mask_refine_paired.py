"""중앙값만 보면 속는다 — 객체별로 짝지어 비교한다.

'평균이 조금 나아졌다'는 60%를 개선하고 40%를 망친 결과일 수도 있다. 그러면
전량에 적용하는 것은 동전 던지기다. 짝지은 차이를 봐야 결정할 수 있다.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "refine_test.json")))
keys = ("base", "close", "crop", "crop_lanczos")
n = {k: len(d[k]["miou"]) for k in keys}
print("표본 수:", n)
if len(set(n.values())) != 1:
    print("[!] 길이가 달라 짝짓기 불가 — 일부 객체에서 변형이 빠졌다")
    raise SystemExit(1)

base_m = np.array(d["base"]["miou"])
base_s = np.array(d["base"]["stab"])
base_p = np.array(d["base"]["pts"])

print(f"\n{'방식':<16}{'모델IoU 개선':>14}{'악화':>8}{'무변':>8}"
      f"{'평균 변화':>11}{'안정성 개선':>12}{'악화':>8}")
print("-" * 78)
for k in ("close", "crop", "crop_lanczos"):
    m = np.array(d[k]["miou"]); s = np.array(d[k]["stab"])
    dm = m - base_m
    ds = s - base_s
    up = int(np.sum(dm > 0.02)); dn = int(np.sum(dm < -0.02))
    same = len(dm) - up - dn
    su = int(np.sum(ds > 0.02)); sd = int(np.sum(ds < -0.02))
    print(f"{k:<16}{up:>14}{dn:>8}{same:>8}{dm.mean():>+11.3f}{su:>12}{sd:>8}")

print("\n크게 나빠진 경우가 있나 (모델IoU −0.2 이상 하락)")
for k in ("close", "crop", "crop_lanczos"):
    m = np.array(d[k]["miou"])
    dm = m - base_m
    bad = int(np.sum(dm < -0.2))
    good = int(np.sum(dm > 0.2))
    print(f"  {k:<16} 크게 악화 {bad:>3}개 · 크게 개선 {good:>3}개")

print("\n지그재그(점>200) 가 없어진 비율")
for k in keys:
    p = np.array(d[k]["pts"])
    print(f"  {k:<16} 점 중앙값 {np.median(p):>6.0f} · 여전히 >200 {np.mean(p > 200):>6.1%}")

print("\n기준자의 한계")
print(f"  기준으로 쓴 모델은 150장짜리 (마스크 mAP50 0.751).")
print(f"  base 의 모델IoU 중앙값이 {np.median(base_m):.3f} 인데, 이 값이 낮은 것이")
print(f"  라벨이 나빠서인지 모델이 약해서인지는 이 자로 가릴 수 없다.")
