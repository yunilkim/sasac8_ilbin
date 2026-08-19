"""프롬프트 흔들기 안정성이 불량 마스크를 잡아내는가.

공식 SAM 의 AutomaticMaskGenerator 는 두 신호로 마스크를 거른다.
  pred_iou_thresh        모델이 스스로 매긴 IoU 예측
  stability_score_thresh 확률맵의 이진화 문턱을 0.5±δ 로 흔들었을 때 마스크가 유지되는가

우리는 박스 프롬프트를 쓰므로 두 번째를 '프롬프트 흔들기'로 옮긴다.
박스를 ±5% 키우고 줄여서 다시 물어보고, 세 마스크가 서로 얼마나 겹치는지 본다.
진짜 물체를 잡았다면 박스가 조금 달라져도 같은 것을 딴다. 배경을 잡았다면
경계가 없으므로 크게 흔들린다.

'SAM on Medical Images'(2305.00035) 가 박스 크기 섭동이 예측 정확도를 크게
바꾼다고 보고한 것과 같은 성질을 이용한다.

불량 표본은 채움 비율 상위 60개(눈으로 6개 확인, 전부 진짜 불량),
대조군은 무작위 200개. 채움 비율로 고른 집단이므로 채움 비율 자신과는
공정한 비교가 안 된다 -- 여기서 재는 것은 '독립적인 신호가 같은 것을 지목하는가'다.
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
JIT = 0.06        # 박스를 이만큼 키우고 줄인다


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


def main():
    from ultralytics import SAM
    m = SAM(os.path.join(D, "sam2.1_b.pt"))

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
                rows.append((poly_area(s) / max(b[3] * b[4], 1e-12), sp, f, i))
    rows.sort(key=lambda r: -r[0])
    bad = rows[:60]
    ctrl = random.Random(0).sample([r for r in rows if r[0] < 0.9], 200)

    def run(items, tag):
        by_file = {}
        for fill, sp, f, i in items:
            by_file.setdefault((sp, f), []).append(i)
        out = []
        for n, ((sp, f), idxs) in enumerate(by_file.items(), 1):
            stem = f[:-4]
            img = cv2.imdecode(np.fromfile(
                os.path.join(SEG, sp, "images", stem + ".jpg"), np.uint8),
                cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            bx = boxes_of(os.path.join(BOX, sp, "labels", f))

            def xyxy(scale):
                v = []
                for _, cx, cy, bw, bh in bx:
                    sw, sh = bw * scale, bh * scale
                    v.append([(cx - sw / 2) * w, (cy - sh / 2) * h,
                              (cx + sw / 2) * w, (cy + sh / 2) * h])
                return v

            masks = []
            confs = None
            try:
                for sc in (1.0, 1.0 + JIT, 1.0 - JIT):
                    r = m(img, bboxes=xyxy(sc), verbose=False, device=0)[0]
                    if r.masks is None or len(r.masks.data) != len(bx):
                        raise RuntimeError
                    masks.append(r.masks.data.cpu().numpy().astype(bool))
                    if confs is None and r.boxes is not None:
                        confs = r.boxes.conf.cpu().numpy()
            except Exception:  # noqa: BLE001
                continue
            for i in idxs:
                a, b_, c = masks[0][i], masks[1][i], masks[2][i]
                def iou(u, v):
                    inter = np.logical_and(u, v).sum()
                    uni = np.logical_or(u, v).sum()
                    return float(inter) / max(float(uni), 1.0)
                stab = min(iou(a, b_), iou(a, c))
                out.append({"stability": stab,
                            "sam_conf": float(confs[i]) if confs is not None else -1,
                            "fill": next(r[0] for r in items
                                         if r[1] == sp and r[2] == f and r[3] == i)})
            if n % 25 == 0:
                print(f"    {tag} {n}/{len(by_file)}", flush=True)
        return out

    print(f"불량 {len(bad)} · 대조군 {len(ctrl)} · 박스 섭동 ±{JIT:.0%}")
    print("불량 표본")
    B = run(bad, "bad")
    print("대조군")
    C = run(ctrl, "ctrl")

    def auc(pos, neg):
        a = np.concatenate([pos, neg])
        l = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        o = np.argsort(a)
        rk = np.empty(len(a))
        rk[o] = np.arange(1, len(a) + 1)
        n1, n0 = l.sum(), (1 - l).sum()
        return 1 - (rk[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    print("\n" + "=" * 72)
    print(f"{'신호':<22}{'불량 중앙값':>12}{'대조군 중앙값':>14}{'AUC':>8}")
    print("=" * 72)
    for key, nm in (("stability", "프롬프트 안정성"), ("sam_conf", "SAM 예측 IoU")):
        p = np.array([x[key] for x in B])
        q = np.array([x[key] for x in C])
        print(f"{nm:<22}{np.median(p):>12.4f}{np.median(q):>14.4f}"
              f"{auc(p, q):>8.3f}")

    # 두 신호를 합치면
    ps = np.array([x["stability"] for x in B])
    pc = np.array([x["sam_conf"] for x in B])
    qs = np.array([x["stability"] for x in C])
    qc = np.array([x["sam_conf"] for x in C])
    print(f"{'둘의 곱':<22}{np.median(ps * pc):>12.4f}"
          f"{np.median(qs * qc):>14.4f}{auc(ps * pc, qs * qc):>8.3f}")

    print("\n프롬프트 안정성 문턱값별")
    for th in (0.7, 0.8, 0.85, 0.9, 0.95):
        print(f"  stability < {th:.2f}   불량 {np.mean(ps < th):>6.1%} 잡음 · "
              f"대조군 {np.mean(qs < th):>6.1%} 버림")

    json.dump({"bad": B, "ctrl": C},
              open(os.path.join(HERE, "stability.json"), "w"), indent=1)
    print("\n-> stability.json")


if __name__ == "__main__":
    main()
