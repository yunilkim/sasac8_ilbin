"""마스크 -> 폴리곤 변환 방식을 비교한다.

문제: ultralytics 의 masks2segments 기본값은 strategy="all" 이고, 마스크가 여러
조각으로 나뉘면 merge_multi_segment 로 조각들을 '선으로 이어' 하나의 폴리곤을
만든다. 메시 소재 조끼처럼 구멍이 많으면 그 연결선들이 갈지자 낙서가 된다.

정답표 없이 잴 수 있는 지표가 있다. 폴리곤을 다시 칠했을 때 SAM 의 원본 이진
마스크를 얼마나 재현하는가 (IoU). 변환 단계만 따로 재는 것이라 GT 가 필요없다.

  fidelity = IoU( fillPoly(폴리곤), SAM 이진마스크 )

비교 대상
  all       현재 방식 (ultralytics 기본)
  largest   가장 큰 외곽선 하나만
  close     구멍 메우기(모폴로지 닫기) 후 가장 큰 외곽선
  close_all 구멍 메우기 후 all

YOLO seg 라벨은 객체당 폴리곤 하나라 구멍을 표현할 수 없다. 그러니 구멍을
메우는 쪽이 형식과도 맞다.
"""
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
SIMPLIFY = 0.0015
KERNEL = 5          # 구멍 메우기 커널


def boxes_of(lp):
    return [(int(float(t[0])), *[float(x) for x in t[1:5]])
            for t in (l.split() for l in open(lp, encoding="utf-8"))
            if len(t) >= 5]


def simplify(pts, w, h):
    eps = SIMPLIFY * float(np.hypot(w, h))
    ap = cv2.approxPolyDP(pts.astype(np.int32).reshape(-1, 1, 2), eps, True)
    return ap.reshape(-1, 2)


def poly_from(mask, mode):
    """이진 마스크 -> 폴리곤(픽셀 좌표). 없으면 None."""
    m = mask.astype(np.uint8)
    if mode.startswith("close"):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL, KERNEL))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    if mode in ("largest", "close"):
        c = max(cs, key=cv2.contourArea)
        return c.reshape(-1, 2)
    # all / close_all : ultralytics 와 같은 방식으로 이어 붙인다
    from ultralytics.data.converter import merge_multi_segment
    if len(cs) == 1:
        return cs[0].reshape(-1, 2)
    return np.concatenate(merge_multi_segment([x.reshape(-1, 2) for x in cs]))


def main():
    from ultralytics import SAM
    sam = SAM(os.path.join(D, "sam2.1_b.pt"))

    rng = random.Random(11)
    cand = [(sp, f) for sp in ("train", "valid", "test")
            for f in os.listdir(os.path.join(SEG, sp, "labels"))]
    rng.shuffle(cand)

    modes = ["all", "largest", "close", "close_all"]
    fid = {m: [] for m in modes}
    npts = {m: [] for m in modes}
    n_obj = 0
    worst = []

    for sp, f in cand:
        if n_obj >= 400:
            break
        img = cv2.imdecode(np.fromfile(
            os.path.join(SEG, sp, "images", f[:-4] + ".jpg"), np.uint8),
            cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        bx = boxes_of(os.path.join(BOX, sp, "labels", f))
        if not bx or len(bx) > 6:
            continue
        xyxy = [[(cx - bw / 2) * w, (cy - bh / 2) * h,
                 (cx + bw / 2) * w, (cy + bh / 2) * h] for _, cx, cy, bw, bh in bx]
        try:
            r = sam(img, bboxes=xyxy, verbose=False, device=0)[0]
            if r.masks is None or len(r.masks.data) != len(bx):
                continue
            md = r.masks.data.cpu().numpy().astype(np.uint8)
        except Exception:  # noqa: BLE001
            continue
        for i in range(len(bx)):
            m = md[i]
            if m.sum() < 50:
                continue
            n_obj += 1
            per = {}
            for mode in modes:
                p = poly_from(m, mode)
                if p is None or len(p) < 3:
                    fid[mode].append(0.0); npts[mode].append(0)
                    per[mode] = 0.0
                    continue
                sp_ = simplify(p, w, h)
                canv = np.zeros_like(m)
                cv2.fillPoly(canv, [sp_.astype(np.int32)], 1)
                inter = int(np.logical_and(canv, m).sum())
                uni = int(np.logical_or(canv, m).sum())
                v = inter / max(uni, 1)
                fid[mode].append(v); npts[mode].append(len(sp_))
                per[mode] = v
            if per["all"] < 0.6:
                worst.append((per["all"], per["close"], sp, f, i))

    print(f"객체 {n_obj}개\n")
    print(f"{'방식':<12}{'충실도 중앙값':>14}{'하위10%':>10}{'0.8미만':>10}"
          f"{'0.5미만':>10}{'점 중앙값':>11}{'점 최대':>9}")
    print("-" * 68)
    for mode in modes:
        v = np.array(fid[mode]); p = np.array(npts[mode])
        print(f"{mode:<12}{np.median(v):>14.4f}{np.percentile(v, 10):>10.4f}"
              f"{np.mean(v < 0.8):>10.1%}{np.mean(v < 0.5):>10.1%}"
              f"{np.median(p):>11.0f}{p.max():>9}")

    print(f"\n현재 방식(all)이 0.6 미만인 객체 {len(worst)}개 "
          f"({len(worst) / max(n_obj, 1):.1%})")
    if worst:
        imp = [w_[1] - w_[0] for w_ in worst]
        print(f"  그것들을 close 로 바꾸면 충실도 "
              f"{np.mean([w_[0] for w_ in worst]):.3f} → "
              f"{np.mean([w_[1] for w_ in worst]):.3f} "
              f"(평균 {np.mean(imp):+.3f})")

    # 눈으로 볼 수 있게 최악 6건을 all vs close 로 나란히
    worst.sort(key=lambda t: t[0])
    cells = []
    for a_, c_, sp, f, i in worst[:6]:
        img = cv2.imdecode(np.fromfile(
            os.path.join(SEG, sp, "images", f[:-4] + ".jpg"), np.uint8),
            cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        bx = boxes_of(os.path.join(BOX, sp, "labels", f))
        xyxy = [[(cx - bw / 2) * w, (cy - bh / 2) * h,
                 (cx + bw / 2) * w, (cy + bh / 2) * h] for _, cx, cy, bw, bh in bx]
        r = sam(img, bboxes=xyxy, verbose=False, device=0)[0]
        m = r.masks.data.cpu().numpy().astype(np.uint8)[i]
        row = []
        for mode, col, lab in (("all", (60, 60, 255), f"all {a_:.2f}"),
                               ("close", (80, 240, 80), f"close {c_:.2f}")):
            vis = img.copy()
            p = poly_from(m, mode)
            if p is not None and len(p) >= 3:
                cv2.polylines(vis, [simplify(p, w, h).astype(np.int32)], True,
                              col, 2, cv2.LINE_AA)
            vis = cv2.resize(vis, (260, 260))
            cv2.putText(vis, lab, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2, cv2.LINE_AA)
            row.append(vis)
        cells.append(np.hstack(row))
    if cells:
        g = [np.hstack(cells[i:i + 2]) for i in range(0, len(cells), 2)]
        wmax = max(x.shape[1] for x in g)
        g = [np.pad(x, ((0, 0), (0, wmax - x.shape[1]), (0, 0))) for x in g]
        cv2.imwrite(os.path.join(HERE, "poly_strategy.jpg"), np.vstack(g))
        print("-> poly_strategy.jpg (왼쪽 빨강=현재 all · 오른쪽 초록=close)")


if __name__ == "__main__":
    main()
