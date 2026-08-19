#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 check_contrast.py  --  화면에서 안 보이는 글씨를 기계적으로 찾는다
================================================================================

 실행:
     .venv\\Scripts\\python.exe check_contrast.py
     .venv\\Scripts\\python.exe check_contrast.py --min 4.5

--------------------------------------------------------------------------------
 [ 왜 필요한가 ]
--------------------------------------------------------------------------------
 2026-08-17 에 시작 메뉴의 콤보박스(대상·스플릿·클래스)가 통째로 안 보인다는
 지적을 받았다. 베이지 바탕에 흰 글씨였다.

 원인은 ttk 의 상태맵이다. style.configure() 는 '보통 상태'의 색만 정하고,
 clam 테마는 상태별 기본값을 따로 갖고 있는데 그게 전부 밝은 색이다.

     disabled -> #dcdad5      active -> #eeebe7

 둘 다 베이지다. 어두운 테마를 씌워도 이 두 상태는 clam 기본값이 그대로 나온다.
 이 앱의 콤보박스는 전부 state="readonly" 라서 늘 그 경로를 탔다.

 눈으로 확인하는 방식으로는 이런 걸 못 잡는다. 어떤 상태는 마우스를 올려야만
 나타나고, 새 다이얼로그를 추가할 때마다 다시 생긴다. 그래서 기계로 훑는다.

--------------------------------------------------------------------------------
 [ 무엇을 재는가 ]
--------------------------------------------------------------------------------
 WCAG 상대휘도 대비. (밝은쪽+0.05) / (어두운쪽+0.05).
     4.5:1 이상   본문 글씨로 적합
     3.0~4.5      큰 글씨나 비활성 표시로만
     3.0 미만     사실상 안 보임

 두 갈래로 본다.
   1) ttk 스타일 -- 상태(readonly/disabled/focus/active/pressed)별 실제 색
   2) 실제 위젯  -- 만들어진 tk 위젯의 최종 색 (스타일이 안 닿는 곳)

 입력계 위젯(Combobox/Entry/Spinbox/Treeview)만 fieldbackground 로 그린다.
 라벨·버튼·프레임에까지 그걸 물으면 루트(".")의 값이 딸려 나와 엉뚱한 바탕과
 비교하게 된다 -- 처음 이 검사를 짤 때 그 실수로 없는 문제를 6건 만들어냈다.
================================================================================
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import tkinter as tk
from tkinter import ttk

DATA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DATA)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import yolo_dataset_studio as S  # noqa: E402

STATES = ("", "readonly", "disabled", "focus", "active", "pressed", "selected")
STYLES = ("TLabel", "TButton", "TCheckbutton", "TCombobox", "TEntry",
          "TSpinbox", "Treeview", "Treeview.Heading", "Status.TLabel",
          "Head.TLabel", "Session.TLabel", "Card.TLabel", "TFrame")
FIELD = {"TCombobox", "TEntry", "TSpinbox", "Treeview"}


def to_rgb(root, name: str):
    if not name:
        return None
    try:
        r, g, b = root.winfo_rgb(name)
    except tk.TclError:
        return None
    return (r / 257, g / 257, b / 257)


def luminance(c) -> float:
    def f(x: float) -> float:
        x /= 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])


def contrast(fg, bg) -> float:
    a, b = luminance(fg), luminance(bg)
    if a < b:
        a, b = b, a
    return (a + 0.05) / (b + 0.05)


def main() -> int:
    ap = argparse.ArgumentParser(description="안 보이는 글씨 찾기")
    ap.add_argument("--min", type=float, default=4.5, help="합격선 (기본 4.5:1)")
    ap.add_argument("--quiet", action="store_true", help="문제만 출력")
    a = ap.parse_args()

    bad, checked = [], 0
    app = S.App()
    app.withdraw()
    app._start_menu_shown = True
    app.update()
    st = ttk.Style(app)

    def check(where: str, fgc: str, bgc: str) -> None:
        nonlocal checked
        f, b = to_rgb(app, fgc), to_rgb(app, bgc)
        if not f or not b:
            return
        checked += 1
        r = contrast(f, b)
        mark = "OK  " if r >= a.min else ("낮음" if r >= 3.0 else "안보임")
        line = (f"  {mark}  {r:5.2f}:1  {where:<44} "
                f"글자 {fgc:<9} 바탕 {bgc}")
        if r < a.min:
            bad.append(line)
        if not a.quiet or r < a.min:
            print(line)

    if not a.quiet:
        print("=" * 88)
        print("1. ttk 스타일 — 상태별")
        print("=" * 88)
    for style in STYLES:
        for stt in STATES:
            stl = (stt,) if stt else ()
            fg = (st.lookup(style, "foreground", stl)
                  or st.lookup(style, "foreground"))
            if style in FIELD:
                bg = (st.lookup(style, "fieldbackground", stl)
                      or st.lookup(style, "fieldbackground")
                      or st.lookup(style, "background", stl))
            else:
                bg = (st.lookup(style, "background", stl)
                      or st.lookup(style, "background"))
            if fg and bg:
                check(f"{style} [{stt or 'normal'}]", fg, bg)
            # readonly 콤보는 값을 '선택된 텍스트'로 그린다. 그 조합도 봐야 한다.
            if style == "TCombobox":
                sf = st.lookup(style, "selectforeground", stl)
                sb = st.lookup(style, "selectbackground", stl)
                if sf and sb:
                    check(f"{style} [{stt or 'normal'}] 선택텍스트", sf, sb)

    if not a.quiet:
        print("\n" + "=" * 88)
        print("2. 실제로 만들어진 tk 위젯 (스타일이 안 닿는 곳)")
        print("=" * 88)

    app.deiconify()
    app.on_start_menu()
    app.update()
    time.sleep(0.3)
    app.update()

    def walk(w):
        out = [w]
        try:
            for c in w.winfo_children():
                out += walk(c)
        except tk.TclError:
            pass
        return out

    tops = [app] + [w for w in app.winfo_children()
                    if w.winfo_class() == "Toplevel"]
    for top in tops:
        for w in walk(top):
            cls = w.winfo_class()
            if cls not in ("Listbox", "Text", "Entry", "Spinbox", "Label",
                           "Canvas"):
                continue
            try:
                keys = w.keys()
                fg = w.cget("fg") if "fg" in keys else None
                bg = w.cget("bg") if "bg" in keys else None
                tag = str(w)[-34:]
                if fg and bg:
                    check(f"tk.{cls} {tag}", str(fg), str(bg))
                sf = w.cget("selectforeground") if "selectforeground" in keys else None
                sb = w.cget("selectbackground") if "selectbackground" in keys else None
                if sf and sb:
                    check(f"tk.{cls} 선택 {tag}", str(sf), str(sb))
            except tk.TclError:
                continue

    app.destroy()
    print("\n" + "=" * 88)
    print(f"조합 {checked}건 검사 · 합격선 {a.min}:1")
    if bad:
        print(f"미달 {len(bad)}건")
        for line in bad:
            print(line)
        return 1
    print("전부 통과 — 안 보이는 조합 없음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
