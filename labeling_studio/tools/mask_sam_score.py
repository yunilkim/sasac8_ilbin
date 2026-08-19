"""SAM 의 예측 IoU 가 내 대리 지표보다 나은 신호인가 — 알려진 불량으로 검증.

불량 표본: 채움 비율(마스크넓이/박스넓이) 상위 = 마스크가 사실상 화면 전체.
           눈으로 6개 확인했고 전부 진짜 불량이었다.
대조군  : 무작위 표본.

두 집단에서 SAM 예측 IoU 분포가 갈리면 그게 원리적인 신호다. 안 갈리면
내 대리 지표를 계속 써야 한다. 어느 쪽이든 측정으로 정한다.
"""
import json
import os
import random

os.environ["YOLO_AUTOINSTALL"] = "false"
import cv2  # noqa: E402
import numpy as np  # noqa: E402

# 작업 루트는 이 스크립트 위치에서 유도한다(tools/ 의 부모).
# 절대경로를 박아 두면 저장소를 받은 사람 환경에서 바로 깨진다.
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOX = os.path.join(D, "db", "vest-helmet_crop_dedup")
SEG = os.path.join(D, "db", "vest-helmet_crop_dedup_seg")
HERE = os.path.dirname(os.path.abspath(__file__))


def boxes_of(lp):
    out = []
    for line in open(lp, encoding="utf-8"):
        t = line.split()
        if len(t) >= 5:
            out.append((int(float(t[0])), *[float(x) for x in t[1:5]]))
    return out


def poly_area(t):
    p = np.array([float(x) for x in t[1:]], np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def collect():
    """(fill, split, file, obj_index) 목록."""
    rows = []
    for sp in ("train", "valid", "test"):
        bl, sl = os.path.join(BOX, sp, "labels"), os.path.join(SEG, sp, "labels")
        for f in sorted(os.listdir(sl)):
            bx = boxes_of(os.path.join(bl, f))
            sg = [l.split() for l in open(os.path.join(sl, f), encoding="utf-8")
                  if l.strip()]
            if len(bx) != len(sg):
                continue
            for i, (b, s) in enumerate(zip(bx, sg)):
                fill = poly_area(s) / max(b[3] * b[4], 1e-12)
                rows.append((fill, sp, f, i))
    return rows


def main():
    from ultralytics import SAM
    m = SAM(os.path.join(D, "sam2.1_b.pt"))

    rows = collect()
    rows.sort(key=lambda r: -r[0])
    bad = rows[:60]                                  # 채움 최상위 = 알려진 불량
    rng = random.Random(0)
    ctrl = rng.sample([r for r in rows if r[0] < 0.9], 200)
    print(f"객체 {len(rows)}개 · 불량 표본 {len(bad)} (채움 "
          f"{bad[-1][0]:.2f}~{bad[0][0]:.2f}) · 대조군 {len(ctrl)}")

    def score(items, tag):
        # 파일 단위로 묶어서 한 번만 추론
        by_file = {}
        for fill, sp, f, i in items:
            by_file.setdefault((sp, f), []).append((i, fill))
        out = []
        for n, ((sp, f), objs) in enumerate(by_file.items(), 1):
            stem = f[:-4]
            ip = os.path.join(SEG, sp, "images", stem + ".jpg")
            img = cv2.imdecode(np.fromfile(ip, np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            bx = boxes_of(os.path.join(BOX, sp, "labels", f))
            xyxy = [[(b[1] - b[3] / 2) * w, (b[2] - b[4] / 2) * h,
                     (b[1] + b[3] / 2) * w, (b[2] + b[4] / 2) * h] for b in bx]
            try:
                r = m(img, bboxes=xyxy, verbose=False, device=0)[0]
            except Exception:  # noqa: BLE001
                continue
            if r.boxes is None or len(r.boxes) != len(bx):
                continue
            conf = r.boxes.conf.cpu().numpy()
            for i, fill in objs:
                if i < len(conf):
                    out.append((float(conf[i]), fill))
            if n % 25 == 0:
                print(f"    {tag} {n}/{len(by_file)}")
        return out

    print("\n불량 표본 채점")
    b = score(bad, "bad")
    print("대조군 채점")
    c = score(ctrl, "ctrl")

    def stat(v, name):
        a = np.array([x[0] for x in v])
        print(f"  {name:<8} n={len(a):<5} 중앙값 {np.median(a):.4f} · "
              f"평균 {a.mean():.4f} · 10%tile {np.percentile(a, 10):.4f} · "
              f"최소 {a.min():.4f} · 최대 {a.max():.4f}")
        return a

    print("\n" + "=" * 70)
    print("SAM 예측 IoU (r.boxes.conf)")
    print("=" * 70)
    ab, ac = stat(b, "불량"), stat(c, "대조군")

    # 이 신호로 불량을 얼마나 골라낼 수 있나
    print("\n문턱값별 걸러짐 (불량을 몇 % 잡고 대조군을 몇 % 버리나)")
    for th in (0.90, 0.93, 0.95, 0.96, 0.97):
        print(f"  conf < {th:.2f}   불량 {np.mean(ab < th):>6.1%} 잡음 · "
              f"대조군 {np.mean(ac < th):>6.1%} 버림")

    # AUC (순위 기반, 낮을수록 불량이면 좋은 신호)
    allv = np.concatenate([ab, ac])
    lbl = np.concatenate([np.ones_like(ab), np.zeros_like(ac)])
    order = np.argsort(allv)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    n1, n0 = lbl.sum(), (1 - lbl).sum()
    auc = 1 - (ranks[lbl == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    print(f"\n  AUC(낮은 conf = 불량) {auc:.3f}   "
          f"(0.5=무의미 · 0.7 이상이면 쓸 만함)")

    json.dump({"bad": [list(x) for x in b], "ctrl": [list(x) for x in c]},
              open(os.path.join(HERE, "sam_score.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
