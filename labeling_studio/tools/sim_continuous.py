# -*- coding: utf-8 -*-
"""연속형(Human-in-the-loop) 라벨링 시나리오를 시간축으로 돌려본다.

사람이 라벨링하는 동안 GPU 가 뒤에서 학습하고, 새 모델이 나오면 아직 안 연
이미지를 다시 예측해 준다. 사람은 한 번도 안 기다린다. 그래서 이 시뮬레이션이
답하는 것은 '얼마나 기다리나'가 아니라 두 가지다.

  1. 전체 몇 시간 걸리나
  2. 낡음(staleness) -- 지금 보는 예측을 만든 모델이 몇 장 뒤처져 있나

낡음이 이 시스템의 진짜 비용이다. 예측은 176 img/s 라 6,681장 전량을 37초에
다시 만든다. 큐가 빌 일이 없다. 느린 것은 학습뿐이고, 그래서 사람이 항상
'조금 낡은' 모델의 예측을 본다.

--------------------------------------------------------------------------
 [ 어디까지 실측인가 -- 이 구분이 중요하다 ]
--------------------------------------------------------------------------
 실측
   빈 화면 12.9초/장          work/manual_bbox 26장 · 88박스 · 335.5초
   학습시간 143 + 0.385n 초   tools/bench_train_time.py (검증 400장·5에폭)
   재예측 37초                make_preds.py 6,531장 실측
   절감 150장 63.8%           GT 계측 6,531장
   절감 500장 80.5%           GT 계측

 추정 (실측 아님)
   제로샷 25%                 비용모델 추정치
   50 / 300 / 1000장 지점     위 실측점을 이은 보간

 -> 50장·300장 지점이 보간이라 총 소요 숫자가 여기 걸려 있다.
    y11s_n50_random 모델이 이미 있으므로 예측을 만들어 GT 로 재면 확정된다.
--------------------------------------------------------------------------

 실행:
     .venv\\Scripts\\python.exe tools\\sim_continuous.py
     .venv\\Scripts\\python.exe tools\\sim_continuous.py --target 500 --first 50
"""
import argparse
import sys

BLANK = 12.9        # 빈 화면 초/장 (실측)
REPRED = 37.0       # 남은 이미지 전량 재예측 (실측). 백그라운드라 사람은 안 멈춘다.

# (모델 학습분, 편집비용 절감률). 0 은 시드 없음, 1 은 제로샷.
CURVE = [(0, 0.00), (1, 0.25), (50, 0.45), (150, 0.638),
         (300, 0.73), (500, 0.805), (1000, 0.85)]


def train_sec(n):
    """검증 400장 · 5에폭마다 · 80에폭 기준. bench_train_time.py 에서 유도."""
    return 143.0 + 0.385 * n


def sec_per_img(n):
    """학습분 n장짜리 모델이 시드해 줄 때 사람이 한 장에 쓰는 초."""
    if n <= 0:
        return BLANK
    for (a, av), (b, bv) in zip(CURVE, CURVE[1:]):
        if a <= n <= b:
            f = 0.0 if b == a else (n - a) / (b - a)
            return BLANK * (1.0 - (av + f * (bv - av)))
    return BLANK * (1.0 - CURVE[-1][1])


def run(target, first_batch=50, zero_shot=True):
    """사건 기반 시뮬레이션. 반환: (총초, [(시각, 학습분, 사람진도, 낡음, 초당)])"""
    t = done = 0.0
    model_n = 1 if zero_shot else 0
    train_n = None
    train_done = None
    log = []
    while done < target - 1e-9:
        rate = sec_per_img(model_n)
        cand = [(t + (target - done) * rate, "target")]
        if train_done is not None:
            cand.append((train_done, "model"))
        elif done < first_batch:
            cand.append((t + (first_batch - done) * rate, "launch"))
        tn, kind = min(cand, key=lambda e: e[0])
        done += (tn - t) / rate
        t = tn
        if kind == "target":
            break
        if kind == "model":
            log.append((t, train_n, int(done), int(done) - train_n, sec_per_img(train_n)))
            model_n = train_n
        # 학습이 끝났거나(=다음 판을 바로 건다) 첫 배치에 도달했다
        train_n = int(round(done))
        train_done = t + train_sec(train_n)
    return t, log


def show(target, first_batch, zero_shot):
    t, log = run(target, first_batch, zero_shot)
    seed0 = "제로샷" if zero_shot else "빈화면"
    print("\n" + "=" * 74)
    print("목표 %d장 · 첫 학습 %d장에서 · 시작 시드 %s" % (target, first_batch, seed0))
    print("=" * 74)
    print("%9s  %-9s%7s%9s%7s%9s" % ("시각", "모델", "학습분", "사람진도", "낡음", "초/장"))
    print("%9s  %-9s%7s%9s%7s%7.1f초" % ("0.0분", seed0, "-", "0장", "-",
                                         sec_per_img(1 if zero_shot else 0)))
    shown = log if len(log) <= 14 else log[:12] + [None] + log[-1:]
    for i, row in enumerate(shown, 1):
        if row is None:
            print("%9s  %-9s" % ("", "..."))
            continue
        tt, tn, dn, st, rate = row
        idx = i if len(log) <= 14 else (i if i <= 12 else len(log))
        print("%8.1f분  %-9s%6d장%8d장%6d장%7.1f초" % (tt / 60, "v%d" % idx, tn, dn, st, rate))
    manual = target * BLANK
    zs = target * sec_per_img(1)
    print("\n  총 소요      %.0f분 (%.1f시간) · 모델 %d번 갱신" % (t / 60, t / 3600, len(log)))
    print("  빈 화면이면  %.1f시간   -> 절감 %.0f%%" % (manual / 3600, (1 - t / manual) * 100))
    print("  제로샷 고정  %.1f시간   -> 추가 절감 %.0f%%" % (zs / 3600, (1 - t / zs) * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, nargs="*", default=[500, 1000, 6681])
    ap.add_argument("--first", type=int, default=50, help="첫 학습을 거는 장수")
    ap.add_argument("--blank-start", action="store_true", help="제로샷 없이 빈 화면에서 시작")
    a = ap.parse_args()
    for n in a.target:
        show(n, a.first, not a.blank_start)
    print("\n※ 50·300장 지점은 보간이다. docstring 의 '어디까지 실측인가' 참조.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
