"""지그재그 마스크를 고치는 네 가지 방법을 같은 자로 비교한다.

대상: 폴리곤 점 > 200 인 객체 (전량의 4.17%, 실질 불량의 대부분이 여기 있다)

  A base        지금 방식 -- 전체 이미지 + 박스 프롬프트
  B crop        박스 둘레를 잘라 SAM 에 다시 (SAM 이 1024 로 확대하므로
                물체가 인코더 안에서 2~3배 커진다)
  C crop_lanczos 자른 뒤 Lanczos 로 미리 1024 까지 키워서 넣음
                (SAM 내부 보간 대신 더 좋은 보간을 쓰는 셈)
  D close       지금 마스크에 모폴로지 닫기 (사장님 1단계).
                앞서 무작위 표본에서는 손해였지만 지그재그에는 4% 밖에 없었다.
                조각난 마스크를 메우는 게 정확히 모폴로지가 하는 일이니
                이 부분집합에서는 다를 수 있다.

자는 셋. 서로 독립적이라 한쪽만 좋아지는 것에 속지 않는다.
  points     폴리곤 점 개수 -- 지그재그가 실제로 사라졌는가
  stability  박스 ±6% 섭동에서 마스크가 유지되는가 (AUC 0.966 로 검증된 신호)
  model_iou  학습된 seg 모델의 예측과 얼마나 맞는가
"""
import json
import os
import random
import time

os.environ["YOLO_AUTOINSTALL"] = "false"
import cv2  # noqa: E402
import numpy as np  # noqa: E402

# 작업 루트는 이 스크립트 위치에서 유도한다(tools/ 의 부모).
# 절대경로를 박아 두면 저장소를 받은 사람 환경에서 바로 깨진다.
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOX = os.path.join(D, "db", "vest-helmet_crop_dedup")
SEG = os.path.join(D, "db", "vest-helmet_crop_dedup_seg")
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(D, "models", "y11sseg_n150_dedup")
SIMPLIFY, MARGIN, JIT, KERNEL = 0.0015, 0.25, 0.06, 5
NEED = 60


def iou(a, b):
    return float(np.logical_and(a, b).sum()) / max(
        float(np.logical_or(a, b).sum()), 1.0)


def poly_of(mask, w, h):
    """ultralytics 와 같은 방식(all + merge)으로 폴리곤을 만든다."""
    from ultralytics.data.converter import merge_multi_segment
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    p = (cs[0].reshape(-1, 2) if len(cs) == 1
         else np.concatenate(merge_multi_segment([c.reshape(-1, 2) for c in cs])))
    eps = SIMPLIFY * float(np.hypot(w, h))
    return cv2.approxPolyDP(p.astype(np.int32).reshape(-1, 1, 2),
                            eps, True).reshape(-1, 2)


def main():
    from ultralytics import SAM, YOLO
    sam = SAM(os.path.join(D, "sam2.1_b.pt"))
    det = YOLO(os.path.join(MODEL, "weights", "best.pt"))
    trained = set(json.load(open(os.path.join(MODEL, "model.json"),
                                 encoding="utf-8"))["train_image_list"])

    # 지그재그 객체 모으기 (학습분 제외)
    targets = []
    for sp in ("train",):
        ldir = os.path.join(SEG, sp, "labels")
        for f in sorted(os.listdir(ldir)):
            img_name = f[:-4] + ".jpg"
            if img_name in trained:
                continue
            for i, l in enumerate(open(os.path.join(ldir, f), encoding="utf-8")):
                t = l.split()
                if len(t) >= 7 and (len(t) - 1) // 2 > 200:
                    targets.append((sp, f, i))
    random.Random(3).shuffle(targets)
    targets = targets[:NEED]
    print(f"지그재그 객체 {len(targets)}개로 비교\n")

    res = {k: {"pts": [], "stab": [], "miou": []}
           for k in ("base", "crop", "crop_lanczos", "close")}
    t0 = time.perf_counter()

    for n, (sp, f, idx) in enumerate(targets, 1):
        stem = f[:-4]
        img = cv2.imdecode(np.fromfile(
            os.path.join(SEG, sp, "images", stem + ".jpg"), np.uint8),
            cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        bx = [(int(float(t[0])), *[float(x) for x in t[1:5]])
              for t in (l.split() for l in
                        open(os.path.join(BOX, sp, "labels", f), encoding="utf-8"))
              if len(t) >= 5]
        if idx >= len(bx):
            continue
        cls, cx, cy, bw, bh = bx[idx]

        # 모델 예측 (기준자) -- 같은 클래스 중 가장 겹치는 것
        pr = det.predict(img, verbose=False, device=0, conf=0.25,
                         imgsz=640, retina_masks=True)[0]
        pmasks = []
        if pr.masks is not None:
            pm = pr.masks.data.cpu().numpy().astype(np.uint8)
            pc = pr.boxes.cls.cpu().numpy().astype(int)
            for k in range(len(pm)):
                mk = pm[k] if pm[k].shape == (h, w) else cv2.resize(
                    pm[k], (w, h), interpolation=cv2.INTER_NEAREST)
                if int(pc[k]) == cls:
                    pmasks.append(mk)

        def score(mask_full, tag):
            p = poly_of(mask_full, w, h)
            if p is None or len(p) < 3:
                return
            res[tag]["pts"].append(len(p))
            res[tag]["miou"].append(max([iou(mask_full, q) for q in pmasks],
                                        default=0.0))

        def run_full(scale):
            b = [[(cx - bw * scale / 2) * w, (cy - bh * scale / 2) * h,
                  (cx + bw * scale / 2) * w, (cy + bh * scale / 2) * h]]
            r = sam(img, bboxes=b, verbose=False, device=0)[0]
            return r.masks.data.cpu().numpy().astype(np.uint8)[0]

        def run_crop(scale, lanczos):
            mw, mh = bw * (1 + 2 * MARGIN), bh * (1 + 2 * MARGIN)
            x0 = int(max(0, (cx - mw / 2) * w)); x1 = int(min(w, (cx + mw / 2) * w))
            y0 = int(max(0, (cy - mh / 2) * h)); y1 = int(min(h, (cy + mh / 2) * h))
            if x1 - x0 < 8 or y1 - y0 < 8:
                return None
            sub = img[y0:y1, x0:x1]
            sh_, sw_ = sub.shape[:2]
            k = 1.0
            if lanczos:
                k = min(1024 / max(sw_, sh_), 4.0)
                if k > 1.05:
                    sub = cv2.resize(sub, (int(sw_ * k), int(sh_ * k)),
                                     interpolation=cv2.INTER_LANCZOS4)
            bxs = [[((cx - bw * scale / 2) * w - x0) * k,
                    ((cy - bh * scale / 2) * h - y0) * k,
                    ((cx + bw * scale / 2) * w - x0) * k,
                    ((cy + bh * scale / 2) * h - y0) * k]]
            r = sam(sub, bboxes=bxs, verbose=False, device=0)[0]
            m = r.masks.data.cpu().numpy().astype(np.uint8)[0]
            if m.shape != (sub.shape[0], sub.shape[1]):
                m = cv2.resize(m, (sub.shape[1], sub.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            if k > 1.05:
                m = cv2.resize(m, (sw_, sh_), interpolation=cv2.INTER_NEAREST)
            out = np.zeros((h, w), np.uint8)
            out[y0:y1, x0:x1] = m
            return out

        try:
            variants = {}
            variants["base"] = run_full(1.0)
            variants["crop"] = run_crop(1.0, False)
            variants["crop_lanczos"] = run_crop(1.0, True)
            kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL, KERNEL))
            variants["close"] = cv2.morphologyEx(variants["base"],
                                                 cv2.MORPH_CLOSE, kk)
            # 안정성: 각 방식마다 섭동 두 번 더
            stab = {}
            for tag, fn_ in (("base", lambda s: run_full(s)),
                             ("crop", lambda s: run_crop(s, False)),
                             ("crop_lanczos", lambda s: run_crop(s, True))):
                a = variants[tag]
                if a is None:
                    continue
                v = []
                for s in (1 + JIT, 1 - JIT):
                    b_ = fn_(s)
                    if b_ is not None:
                        v.append(iou(a, b_))
                stab[tag] = min(v) if v else 0.0
            ck = cv2.morphologyEx(run_full(1 + JIT), cv2.MORPH_CLOSE, kk)
            ck2 = cv2.morphologyEx(run_full(1 - JIT), cv2.MORPH_CLOSE, kk)
            stab["close"] = min(iou(variants["close"], ck),
                                iou(variants["close"], ck2))
        except Exception as e:  # noqa: BLE001
            print(f"  실패 {stem[:24]}: {type(e).__name__}")
            continue

        for tag, m in variants.items():
            if m is None:
                continue
            score(m, tag)
            res[tag]["stab"].append(stab.get(tag, 0.0))
        if n % 10 == 0:
            print(f"  {n}/{len(targets)}  {time.perf_counter() - t0:.0f}s")

    print(f"\n{'방식':<16}{'점 중앙값':>10}{'점>200 비율':>12}"
          f"{'안정성 중앙값':>14}{'모델IoU 중앙값':>15}{'모델IoU<0.5':>12}")
    print("-" * 79)
    for tag in ("base", "close", "crop", "crop_lanczos"):
        d = res[tag]
        if not d["pts"]:
            continue
        p = np.array(d["pts"]); s = np.array(d["stab"]); m = np.array(d["miou"])
        print(f"{tag:<16}{np.median(p):>10.0f}{np.mean(p > 200):>12.1%}"
              f"{np.median(s):>14.3f}{np.median(m):>15.3f}{np.mean(m < 0.5):>12.1%}")

    json.dump({k: {kk: list(map(float, vv)) for kk, vv in v.items()}
               for k, v in res.items()},
              open(os.path.join(HERE, "refine_test.json"), "w"), indent=1)
    print(f"\n{time.perf_counter() - t0:.0f}초 · -> refine_test.json")


if __name__ == "__main__":
    main()
