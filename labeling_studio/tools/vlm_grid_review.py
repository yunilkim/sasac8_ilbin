"""박스로 자른 조각을 격자로 묶어 VLM 에게 한 번에 물어본다.

발상(사장님): 어차피 박스로 잘라서 조각이 작다. N 개를 번호 붙여 격자로 묶으면
한 번의 호출로 N 개를 판정할 수 있고, VLM 은 '이 중에 3번이 이상하다' 같은
비교 판단을 잘한다. 처리량이 N 배가 된다.

이 스크립트는 그 발상이 실제로 되는지 잰다. 세 가지를 확인한다.
  1) 검출력   -- 안정성 검사가 이미 불량으로 지목한 것을 VLM 도 찾아내는가
  2) 위치 편향 -- 격자에서 특정 자리에 있는 것을 더/덜 찍는가 (알려진 함정)
  3) 처리량   -- 객체당 실제 몇 초인가

정답표는 프롬프트 섭동 안정성으로 만든다 (AUC 0.966 으로 검증된 신호).
  불량 = stability < 0.20   거의 확실한 실패
  정상 = stability > 0.95
'VLM 이 안정성을 재현하는가'를 재는 것이지 '진짜 정답'을 재는 게 아니다.
그 한계는 결과에 함께 적는다. 대신 안정성이 정상이라 한 것 중 VLM 이 잡은 것은
따로 모아 둔다 -- 기하 신호가 못 보는 의미 오류가 거기 있을 수 있다.
"""
from __future__ import annotations

import base64
import json
import os
import random
import time

os.environ["YOLO_AUTOINSTALL"] = "false"
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402

# 작업 루트는 이 스크립트 위치에서 유도한다(tools/ 의 부모).
# 절대경로를 박아 두면 저장소를 받은 사람 환경에서 바로 깨진다.
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOX = os.path.join(D, "db", "vest-helmet_crop_dedup")
SEG = os.path.join(D, "db", "vest-helmet_crop_dedup_seg")
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["reflective_jacket", "safety_helmet"]
KO = {"reflective_jacket": "안전 조끼", "safety_helmet": "안전모"}

URL = "http://localhost:1234/v1/chat/completions"
MODEL = "gemma-4-12b-it-qat-uncensored-heretic-nvfp4"
CELL = 300          # 격자 한 칸 픽셀
PER_GRID = 5        # 한 번에 몇 개
MARGIN = 0.25       # 박스 둘레 여유
JIT = 0.06

PROMPT = """이미지는 {n}개의 칸으로 나뉘어 있고 각 칸 위에 번호와 물체 이름이 적혀 있다.
각 칸은 사진의 일부를 잘라낸 것이고, 그 안의 물체에 초록색 윤곽선을 그려 놓았다.

할 일: 초록색 윤곽선이 적힌 이름의 물체를 제대로 감싸고 있는지 칸마다 판단하라.

'잘못됨'으로 볼 것:
- 윤곽선이 물체가 아니라 배경이나 화면 전체를 감싸고 있다
- 윤곽선이 엉뚱한 물체(사람 얼굴, 옷, 벽 등)를 감싸고 있다
- 윤곽선이 물체의 아주 일부만 감싸고 있다

'괜찮음'으로 볼 것:
- 윤곽선이 대체로 물체를 따라간다 (경계가 몇 픽셀 어긋나는 것은 괜찮다)

잘못된 칸이 없을 수도 있고, 여러 개일 수도 있다. 억지로 고르지 마라.

JSON 만 출력하라. 설명하지 마라.
{{"bad": [잘못된 칸 번호들], "reason": {{"번호": "짧은 이유"}}}}"""


def boxes_of(lp):
    out = []
    for line in open(lp, encoding="utf-8"):
        t = line.split()
        if len(t) >= 5:
            out.append((int(float(t[0])), *[float(x) for x in t[1:5]]))
    return out


def polys_of(lp):
    out = []
    for line in open(lp, encoding="utf-8"):
        t = line.split()
        if len(t) >= 7:
            out.append((int(float(t[0])),
                        np.array([float(x) for x in t[1:]], np.float64).reshape(-1, 2)))
    return out


def crop_cell(sp, f, idx):
    """객체 하나를 잘라 마스크 윤곽을 그린 CELL x CELL 조각."""
    stem = f[:-4]
    img = cv2.imdecode(np.fromfile(os.path.join(SEG, sp, "images", stem + ".jpg"),
                                   np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    bx = boxes_of(os.path.join(BOX, sp, "labels", f))
    pl = polys_of(os.path.join(SEG, sp, "labels", f))
    if idx >= len(bx) or idx >= len(pl):
        return None, None
    cls, cx, cy, bw, bh = bx[idx]
    vis = img.copy()
    p = (pl[idx][1] * [w, h]).astype(np.int32)
    cv2.polylines(vis, [p], True, (80, 240, 80), 2, cv2.LINE_AA)
    # 박스 둘레로 자른다. 여유를 줘야 '윤곽선이 밖으로 새는' 것이 보인다.
    mw, mh = bw * (1 + MARGIN * 2), bh * (1 + MARGIN * 2)
    x0 = int(max(0, (cx - mw / 2) * w)); x1 = int(min(w, (cx + mw / 2) * w))
    y0 = int(max(0, (cy - mh / 2) * h)); y1 = int(min(h, (cy + mh / 2) * h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, None
    return cv2.resize(vis[y0:y1, x0:x1], (CELL, CELL),
                      interpolation=cv2.INTER_AREA), NAMES[cls]


def make_grid(items):
    """items = [(cell_img, name)] -> 번호 붙은 격자 이미지."""
    n = len(items)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    hd = 34
    sheet = np.full((rows * (CELL + hd), cols * CELL, 3), 24, np.uint8)
    for i, (cell, name) in enumerate(items):
        r, c = divmod(i, cols)
        oy, ox = r * (CELL + hd), c * CELL
        sheet[oy + hd:oy + hd + CELL, ox:ox + CELL] = cell
        cv2.rectangle(sheet, (ox, oy), (ox + CELL, oy + hd), (40, 40, 40), -1)
        cv2.putText(sheet, f"{i + 1}. {name}", (ox + 8, oy + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(sheet, (ox + 1, oy + 1), (ox + CELL - 1, oy + hd + CELL - 1),
                      (90, 90, 90), 1)
    return sheet


def ask(img, n):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    b64 = base64.b64encode(buf.tobytes()).decode()
    body = {
        "model": MODEL, "temperature": 0.0, "max_tokens": 400,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(n=n)},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
    }
    t = time.perf_counter()
    r = requests.post(URL, json=body, timeout=300)
    el = time.perf_counter() - t
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    s = txt.find("{")
    e = txt.rfind("}")
    try:
        d = json.loads(txt[s:e + 1])
        bad = [int(x) for x in d.get("bad", []) if 1 <= int(x) <= n]
    except (ValueError, TypeError, KeyError):
        return None, el, txt
    return (bad, d.get("reason", {})), el, txt


def main():
    from ultralytics import SAM
    sam = SAM(os.path.join(D, "sam2.1_b.pt"))

    # ---- 후보 모으고 안정성으로 정답표 만들기 ---------------------------
    rng = random.Random(7)
    cand = []
    for sp in ("train", "valid", "test"):
        sl = os.path.join(SEG, sp, "labels")
        for f in os.listdir(sl):
            cand.append((sp, f))
    rng.shuffle(cand)

    print("안정성 계산 중 (정답표 만들기)")
    labeled = []
    seen_bad = seen_good = 0
    for sp, f in cand:
        if seen_bad >= 40 and seen_good >= 80:
            break
        stem = f[:-4]
        ip = os.path.join(SEG, sp, "images", stem + ".jpg")
        img = cv2.imdecode(np.fromfile(ip, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        bx = boxes_of(os.path.join(BOX, sp, "labels", f))
        if not bx or len(bx) > 6:
            continue

        def xyxy(s):
            return [[(cx - bw * s / 2) * w, (cy - bh * s / 2) * h,
                     (cx + bw * s / 2) * w, (cy + bh * s / 2) * h]
                    for _, cx, cy, bw, bh in bx]

        try:
            ms = []
            for s in (1.0, 1 + JIT, 1 - JIT):
                r = sam(img, bboxes=xyxy(s), verbose=False, device=0)[0]
                if r.masks is None or len(r.masks.data) != len(bx):
                    raise RuntimeError
                ms.append(r.masks.data.cpu().numpy().astype(bool))
        except Exception:  # noqa: BLE001
            continue
        for i in range(len(bx)):
            a, b, c = ms[0][i], ms[1][i], ms[2][i]
            def iou(u, v):
                return float(np.logical_and(u, v).sum()) / max(
                    float(np.logical_or(u, v).sum()), 1.0)
            st = min(iou(a, b), iou(a, c))
            if st < 0.20 and seen_bad < 40:
                labeled.append((sp, f, i, st, "bad")); seen_bad += 1
            elif st > 0.95 and seen_good < 80:
                labeled.append((sp, f, i, st, "good")); seen_good += 1
    print(f"  불량 {seen_bad} · 정상 {seen_good}")

    # ---- 격자 구성 (자리 무작위, 불량 개수 0~2 로 섞음) ------------------
    bad = [x for x in labeled if x[4] == "bad"]
    good = [x for x in labeled if x[4] == "good"]
    rng.shuffle(bad); rng.shuffle(good)
    grids = []
    bi = gi = 0
    while gi < len(good):
        k = rng.choice([0, 1, 1, 2]) if bi < len(bad) else 0
        k = min(k, len(bad) - bi)
        take_g = PER_GRID - k
        if gi + take_g > len(good):
            break
        items = bad[bi:bi + k] + good[gi:gi + take_g]
        bi += k; gi += take_g
        rng.shuffle(items)
        grids.append(items)
    print(f"  격자 {len(grids)}개 · 칸 {sum(len(g) for g in grids)}개")

    # ---- 질의 ------------------------------------------------------------
    TP = FP = FN = TN = 0
    pos_hit = {i: [0, 0] for i in range(1, PER_GRID + 1)}   # [잡음, 전체불량]
    pos_fp = {i: 0 for i in range(1, PER_GRID + 1)}
    times = []
    extra = []
    fails = 0
    for gi_, items in enumerate(grids, 1):
        cells = []
        keep = []
        for sp, f, i, st, lab in items:
            c, nm = crop_cell(sp, f, i)
            if c is None:
                continue
            cells.append((c, nm)); keep.append((sp, f, i, st, lab))
        if len(cells) < 2:
            continue
        sheet = make_grid(cells)
        if gi_ == 1:
            cv2.imwrite(os.path.join(HERE, "grid_sample.jpg"), sheet)
        res, el, raw = ask(sheet, len(cells))
        times.append(el)
        if res is None:
            fails += 1
            print(f"  격자 {gi_}: 파싱 실패 · {raw[:70]}")
            continue
        picked, reasons = res
        for k, (sp, f, i, st, lab) in enumerate(keep, 1):
            got = k in picked
            if lab == "bad":
                pos_hit[k][1] += 1
                if got:
                    TP += 1; pos_hit[k][0] += 1
                else:
                    FN += 1
            else:
                if got:
                    FP += 1; pos_fp[k] += 1
                    extra.append({"split": sp, "file": f, "obj": i,
                                  "stability": round(st, 4),
                                  "reason": reasons.get(str(k), "")})
                else:
                    TN += 1
        if gi_ % 5 == 0:
            print(f"    {gi_}/{len(grids)}  {el:.1f}s")

    n_obj = TP + FP + FN + TN
    print("\n" + "=" * 72)
    print(f"격자 {len(times)}개 · 칸 {n_obj}개 · 파싱 실패 {fails}")
    print("=" * 72)
    print(f"  안정성이 '불량'이라 한 것을 VLM 이 잡은 비율 (재현율) "
          f"{TP / max(TP + FN, 1):.1%}  ({TP}/{TP + FN})")
    print(f"  안정성이 '정상'이라 한 것을 VLM 이 찍은 비율        "
          f"{FP / max(FP + TN, 1):.1%}  ({FP}/{FP + TN})")
    print(f"  정밀도 {TP / max(TP + FP, 1):.1%}")
    if times:
        t = np.array(times)
        print(f"\n  격자당 {t.mean():.1f}초 (중앙값 {np.median(t):.1f}) · "
              f"칸당 {t.sum() / max(n_obj, 1):.2f}초")
        print(f"  단건 처리(8.9초/장) 대비 {8.9 / max(t.sum() / max(n_obj, 1), 1e-9):.1f}배")
        print(f"  31,612개 환산 {31612 * t.sum() / max(n_obj, 1) / 3600:.1f}시간 (1병렬)")

    print("\n  위치별 (격자 안 자리에 따라 다르게 보는가)")
    for k in range(1, PER_GRID + 1):
        hit, tot = pos_hit[k]
        print(f"    {k}번 자리  불량 {tot}개 중 {hit}개 잡음"
              + (f" ({hit / tot:.0%})" if tot else "        ")
              + f"   · 오탐 {pos_fp[k]}개")

    json.dump({"TP": TP, "FP": FP, "FN": FN, "TN": TN,
               "pos_hit": pos_hit, "pos_fp": pos_fp,
               "sec_per_grid": float(np.mean(times)) if times else 0,
               "extra_flags": extra},
              open(os.path.join(HERE, "grid_vlm.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n  안정성은 정상인데 VLM 이 잡은 것 {len(extra)}개 -> grid_vlm.json")
    print("  -> grid_sample.jpg (VLM 이 본 격자 예시)")


if __name__ == "__main__":
    main()
