# -*- coding: utf-8 -*-
"""연속형 라벨링 세션 분석 — 사람 시간이 실제로 어떻게 줄었는가.

실행:
    .venv\\Scripts\\python.exe tools\\analyze_live.py [작업공간이름]

이 프로젝트의 성공 기준은 mAP 가 아니라 사람 편집 비용이다. 그런데 지금까지
절감률은 전부 비용모델 추정치(그리기 5초·삭제 1초·조정 3초)였고, 시드를 받았을
때 사람이 실제로 몇 초를 쓰는지는 한 번도 재지 못했다.

연속형 세션은 그 측정을 공짜로 준다. 같은 사람이 같은 자리에서 한 판을 하는 동안
모델이 v1, v2, v3 ... 로 바뀌므로, 레코드의 pred_model 로 구간을 갈라
**빈 화면 대 v1 대 v2 ...** 의 장당 실측 초를 짝지어 비교할 수 있다.

읽는 법 주의:
  - 구간마다 이미지가 다르다. 어려운 장이 몰리면 구간 평균이 흔들린다.
    그래서 장당 초와 함께 **박스당 초**와 표본 수를 같이 본다.
  - 뒤로 갈수록 사람이 손에 익는 학습 효과가 섞인다. 모델 개선분과 분리되지 않는다.
  - GT 대조는 참고용이다. GT 는 B 태그(익명 주석자)이고 오류가 확인된 자다.
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
sys.path.insert(0, DATA)

import yolo_dataset_studio as S  # noqa: E402


def fmt(sec):
    return S.fmt_hms(sec)


def bar(v, vmax, w=22):
    n = int(round(w * v / vmax)) if vmax > 0 else 0
    return "█" * n + "·" * (w - n)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "live_0818"
    lay = S.Layout(DATA)
    ws = S.Workspace.load(os.path.join(lay.work_dir, name))
    # ts 는 '마지막 저장' 시각이다. 나중에 s 로 훑고 지나가면 갱신되므로 세션
    # 순서가 아니다. 작업목록(사람에게 제시된) 순서를 쓴다 -- w/s 로 오간 만큼은
    # 어긋나지만 ts 보다 훨씬 실제에 가깝다.
    pos = {}
    for sp_name, files in ws.worklist.items():
        for i, f in enumerate(files):
            pos[(sp_name, f)] = i
    recs = sorted(ws.records.values(),
                  key=lambda r: pos.get((r["split"], r["image"]), 10 ** 9))
    if not recs:
        print("검수 기록이 없습니다.")
        return 1

    # ---------------------------------------------------------------- 모델 이력
    jd = os.path.join(ws.path, "jobs", "done")
    jobs = []
    for fn in sorted(os.listdir(jd)) if os.path.isdir(jd) else []:
        j = json.load(open(os.path.join(jd, fn), encoding="utf-8"))
        if j.get("ok"):
            jobs.append(j)

    print("=" * 74)
    print("연속형 라벨링 세션 분석 — %s" % ws.name)
    print("=" * 74)
    print("데이터셋 %s · 작업목록 %d장"
          % (os.path.basename(ws.source),
             sum(len(v) for v in ws.worklist.values())))

    # ---------------------------------------------------------------- 총계
    t = ws.totals()
    ops = t["ops"]
    real = ops.seconds
    boxes = t["boxes"]
    n = t["images"]
    print("\n" + "-" * 74)
    print("1. 총계")
    print("-" * 74)
    print("  검수      %d장 · 박스 %d개 · 장당 %.1f개" % (n, boxes, boxes / max(n, 1)))
    print("  실측 시간 %s  (장당 %.1f초 · 박스당 %.1f초)"
          % (fmt(real), real / max(n, 1), real / max(boxes, 1)))
    print("  편집 연산 그리기 %d · 삭제 %d · 조정 %d · 유지 %d"
          % (ops.drawn, ops.deleted, ops.adjusted, ops.kept))
    print("  승격      %d개 (기각 티어를 클릭으로 살린 것)" % t["promoted"])
    tot = ops.drawn + ops.deleted + ops.adjusted + ops.kept
    if tot:
        print("  → 모델 예측 처리:  유지 %.0f%% · 조정 %.0f%% · 삭제 %.0f%% · 새로 그림 %.0f%%"
              % (100 * ops.kept / tot, 100 * ops.adjusted / tot,
                 100 * ops.deleted / tot, 100 * ops.drawn / tot))
    prov = t["prov"]
    human = prov.get("A", 0) + prov.get("B", 0)
    auto = prov.get("C", 0) + prov.get("D", 0)
    print("  출처      사람 A+B %d · 기계 C+D %d  (자동화율 %.0f%%)"
          % (human, auto, 100 * auto / max(human + auto, 1)))
    print("  비용모델  편집 %s vs 수동 %s → 절감 %.1f%%"
          % (fmt(t["cost_sec"]), fmt(t["manual_sec"]), (1 - t["ratio"]) * 100))

    # ---------------------------------------------------------------- 모델 이력
    print("\n" + "-" * 74)
    print("2. 모델 이력")
    print("-" * 74)
    print("  %-16s %6s %8s %9s %8s %8s" %
          ("모델", "학습분", "mAP50", "mAP50-95", "학습", "예측"))
    for j in jobs:
        m = j.get("metrics") or {}
        print("  %-16s %5d장 %8.4f %9.4f %7.0f초 %7.0f초"
              % (j.get("model"), j.get("trained_images", 0),
                 m.get("mAP50", 0), m.get("mAP50-95", 0),
                 j.get("train_seconds", 0), j.get("pred_seconds", 0)))

    # ---------------------------------------------------------------- 구간 비교
    print("\n" + "-" * 74)
    print("3. 시드를 만든 모델별 사람 비용  ← 이 세션의 핵심 측정")
    print("-" * 74)
    seg = defaultdict(list)
    for r in recs:
        seg[r.get("pred_model") or "(없음 · 빈 화면)"].append(r)

    order = ["(없음 · 빈 화면)"] + [j["model"] for j in jobs]
    rows = []
    for k in order:
        rs = seg.get(k)
        if not rs:
            continue
        sec = sum(r["ops"]["seconds"] for r in rs)
        bx = sum(len(r["src"]) for r in rs)
        o = defaultdict(int)
        for r in rs:
            for kk in ("drawn", "deleted", "adjusted", "kept"):
                o[kk] += r["ops"][kk]
        rows.append((k, len(rs), bx, sec, o))

    vmax = max((s / max(c, 1) for _, c, _, s, _ in rows), default=1)
    print("  %-22s %4s %5s %8s %8s  %s" %
          ("시드 출처", "장", "박스", "장당초", "박스당", "장당 초"))
    for k, c, bx, sec, o in rows:
        print("  %-22s %3d장 %4d개 %7.1f초 %7.1f초  %s"
              % (k, c, bx, sec / c, sec / max(bx, 1), bar(sec / c, vmax)))

    MIN_N = 10          # 표본이 이보다 작으면 평균을 말하지 않는다
    solid = [r for r in rows if r[1] >= MIN_N]
    thin = [r for r in rows if r[1] < MIN_N]
    if thin:
        print("\n  ※ 표본 %d장 미만이라 평균을 말할 수 없는 구간: %s"
              % (MIN_N, ", ".join("%s(%d장)" % (r[0], r[1]) for r in thin)))
    if len(solid) >= 2:
        base = solid[0][3] / solid[0][1]
        last = solid[-1][3] / solid[-1][1]
        d = (1 - last / base) * 100
        print("\n  %s %.1f초/장  →  %s %.1f초/장   =  %s %.0f%%"
              % (solid[0][0], base, solid[-1][0], last,
                 "단축" if d > 0 else "오히려 증가", abs(d)))
        print("  ※ 구간마다 이미지가 다르고, 사람이 손에 익는 효과가 섞인다.")

    print("\n  구간별 편집 연산 (모델 예측을 어떻게 처리했나)")
    print("  %-22s %7s %7s %7s %7s" % ("시드 출처", "유지", "조정", "삭제", "그리기"))
    for k, c, bx, sec, o in rows:
        tt = sum(o.values())
        if not tt:
            continue
        print("  %-22s %6.0f%% %6.0f%% %6.0f%% %6.0f%%"
              % (k, 100 * o["kept"] / tt, 100 * o["adjusted"] / tt,
                 100 * o["deleted"] / tt, 100 * o["drawn"] / tt))

    # ---------------------------------------------------------------- 시간 추이
    print("\n" + "-" * 74)
    print("4. 시간 추이 (작업목록 순서 · 10장 묶음)")
    print("-" * 74)
    CH = 10
    chunks = [recs[i:i + CH] for i in range(0, len(recs), CH)]
    vmax = max((sum(r["ops"]["seconds"] for r in c) / len(c)) for c in chunks)
    for i, c in enumerate(chunks):
        sec = sum(r["ops"]["seconds"] for r in c)
        bx = sum(len(r["src"]) for r in c)
        mods = {r.get("pred_model") or "빈화면" for r in c}
        tag = sorted(mods)[-1].replace(ws.name + "_", "")
        print("  %3d~%-3d %7.1f초/장 %5.1f박스 %-10s %s"
              % (i * CH + 1, i * CH + len(c), sec / len(c), bx / len(c),
                 tag, bar(sec / len(c), vmax)))

    # ---------------------------------------------------------------- GT 대조
    print("\n" + "-" * 74)
    print("5. GT 대조 (참고용 — GT 는 오류가 확인된 B 태그다)")
    print("-" * 74)
    proj = S.Project(ws.source)
    tp = fp = fn_ = 0
    for r in recs:
        sp = proj.split_by_name(r["split"])
        if not sp:
            continue
        mine, _ = S.parse_label_file(ws.label_path(r["split"], r["image"]))
        gt, _ = S.parse_label_file(sp.label_path_for(r["image"]))
        m = S.match_greedy(gt, mine, 0.5, same_class=False)
        ok = sum(1 for _v, i, j in m if gt[i].cls == mine[j].cls)
        tp += ok
        fp += len(mine) - ok
        fn_ += len(gt) - ok
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn_, 1)
    print("  내 라벨 %d개 vs GT %d개 · 일치 %d" % (tp + fp, tp + fn_, tp))
    print("  GT 기준 정밀도 %.3f · 재현율 %.3f · F1 %.3f"
          % (P, R, 2 * P * R / max(P + R, 1e-9)))
    print("  ※ 낮다고 내 라벨이 틀린 것은 아니다. 규약이 다르면 이렇게 갈린다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
