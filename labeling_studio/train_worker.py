#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 train_worker.py  --  연속형 라벨링의 백그라운드 학습 워커
================================================================================

 실행:
     .venv\\Scripts\\python.exe train_worker.py            (모든 작업공간 감시)
     .venv\\Scripts\\python.exe train_worker.py --workspace live_helmet
     .venv\\Scripts\\python.exe train_worker.py --once     (큐를 한 번만 비우고 종료)

 사람이 라벨링하는 동안 이 프로세스가 뒤에서 학습을 돌리고, 끝나면 아직 안 연
 이미지에 예측을 새로 깔아준다. 사람은 학습을 기다리지 않는다.

--------------------------------------------------------------------------------
 [ 왜 별도 프로세스인가 ]
--------------------------------------------------------------------------------
 스튜디오는 Tkinter 다. 학습을 스레드로 돌리면 ultralytics 가 GIL 을 잡고 있는
 동안 UI 가 얼어붙는다. 프로세스를 나누면 GPU 작업과 화면이 완전히 분리된다.

--------------------------------------------------------------------------------
 [ 왜 소켓이 아니라 파일인가 ]
--------------------------------------------------------------------------------
 이 프로젝트는 이미 파일 배치(db / work / export)로 상태를 주고받는다. 큐도 같은
 규약을 쓰면 새로 배울 개념이 없고, 워커가 죽어도 요청이 폴더에 남아 있어 다시
 띄우면 이어서 한다. 스튜디오는 폴더만 보면 되므로 통신 코드가 아예 필요 없다.

     work/<작업공간>/jobs/queued/<작업id>.json    스튜디오가 떨어뜨린 요청
     work/<작업공간>/jobs/running/<작업id>.json   워커가 집어간 것
     work/<작업공간>/jobs/done/<작업id>.json      끝난 것 (결과가 덧붙는다)

 집어가기는 os.rename 한 번이다. 파일시스템이 원자적으로 보장하므로 워커가 둘
 떠 있어도 같은 작업을 두 번 하지 않는다.

--------------------------------------------------------------------------------
 [ 밀린 요청은 가장 최근 것만 한다 ]
--------------------------------------------------------------------------------
 사람이 계속 라벨링하는 동안 요청이 여러 개 쌓일 수 있다. 그때 오래된 것부터
 처리하면 이미 낡은 라벨로 학습하게 된다. 한 작업공간에서 큐에 여럿이 있으면
 **가장 최근 것만 실행하고 나머지는 superseded 로 넘긴다.** 어차피 최신 요청이
 그 시점까지 확정된 라벨을 전부 포함한다.

--------------------------------------------------------------------------------
 [ 예측은 전량 다시 만든다 ]
--------------------------------------------------------------------------------
 '아직 안 연 이미지만' 골라 예측하면 사람이 그 사이에 연 이미지와 경합이 생긴다.
 그런데 예측은 176 img/s 라 6,531장이 37초다. 그냥 스플릿 전량을 다시 만들고,
 이미 확정된 이미지의 예측은 스튜디오가 무시하게 두는 편이 단순하고 안전하다.
 학습에 쓴 이미지는 make_preds.py 가 train_image_list 로 알아서 뺀다.
================================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from typing import List, Optional, Tuple

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

DATA = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(DATA, ".venv", "Scripts", "python.exe")
if not os.path.isfile(PYTHON):
    PYTHON = sys.executable

QUEUED, RUNNING, DONE = "queued", "running", "done"
POLL_SEC = 2.0
HEARTBEAT = "worker.json"      # work/worker.json — 살아 있음을 알리는 파일
HEARTBEAT_STALE = 30.0         # 이보다 오래되면 죽은 것으로 본다
STOPFILE = "worker.stop"       # 스튜디오가 닫힐 때 남긴다 — 곱게 멈추라는 신호


def stop_path(root: str) -> str:
    return os.path.join(root, "work", STOPFILE)


def ask_stop(root: str) -> None:
    """스튜디오가 닫힐 때 부른다. 죽이지 않고 '현재 작업을 마치고 멈춰라'로 둔다.

    프로세스를 강제로 죽이면 몇 분짜리 학습이 통째로 날아간다. 워커는 폴링
    간격마다 이 파일을 보고, 돌던 작업이 끝난 뒤에 스스로 나간다.
    """
    try:
        os.makedirs(os.path.join(root, "work"), exist_ok=True)
        with open(stop_path(root), "w", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    except OSError:
        pass


def _consume_stop(root: str) -> bool:
    """멈추라는 신호가 있으면 지우고 True. 지우는 이유는 다음 기동을 막지 않기 위해서."""
    p = stop_path(root)
    if not os.path.exists(p):
        return False
    try:
        os.remove(p)
    except OSError:
        pass
    return True


# --------------------------------------------------------------- 심장박동
#  워커가 안 떠 있으면 요청이 큐에 쌓이기만 하고 학습은 영원히 안 된다. 사람은
#  500장을 라벨하고 나서야 모델이 없다는 걸 알게 된다 -- 조용한 실패다.
#  그래서 워커가 살아 있다는 사실 자체를 파일로 남기고 스튜디오가 이를 읽는다.

_beat = {"state": "idle", "detail": "", "started": time.time()}


def beat(root: str, state: Optional[str] = None, detail: str = "") -> None:
    """상태를 갱신하고 즉시 한 번 찍는다. 이후는 아래 스레드가 이어서 찍는다."""
    if state is not None:
        _beat.update(state=state, detail=detail, started=time.time())
    d = os.path.join(root, "work")
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, HEARTBEAT + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "ts": time.time(),
                       "state": _beat["state"], "detail": _beat["detail"],
                       "elapsed": round(time.time() - _beat["started"], 1)}, f)
        os.replace(tmp, os.path.join(d, HEARTBEAT))
    except OSError:
        pass


def start_beating(root: str) -> None:
    """학습은 subprocess 로 몇 분씩 블로킹되므로 심장박동은 별도 스레드가 찍는다.
    안 그러면 학습 중에 박동이 끊겨 스튜디오가 '워커 꺼짐'으로 오해한다."""
    def loop():
        while True:
            beat(root)
            time.sleep(5.0)
    t = threading.Thread(target=loop, daemon=True)
    t.start()


def worker_alive(root: str) -> Optional[dict]:
    """워커가 살아 있으면 그 상태, 아니면 None. 스튜디오가 부른다."""
    try:
        with open(os.path.join(root, "work", HEARTBEAT), "r", encoding="utf-8") as f:
            h = json.load(f)
    except (OSError, ValueError):
        return None
    return h if (time.time() - float(h.get("ts", 0))) < HEARTBEAT_STALE else None


# ------------------------------------------------------------------ 큐 다루기
def jobs_dir(ws_dir: str, state: str) -> str:
    d = os.path.join(ws_dir, "jobs", state)
    os.makedirs(d, exist_ok=True)
    return d


def read_job(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_job(path: str, job: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def submit(ws_dir: str, job: dict) -> str:
    """스튜디오 쪽에서 부르는 함수. 요청 하나를 큐에 넣는다."""
    job.setdefault("id", time.strftime("%Y%m%d-%H%M%S")
                   + "_n%d" % int(job.get("n_confirmed", 0)))
    job.setdefault("requested", time.strftime("%Y-%m-%d %H:%M:%S"))
    p = os.path.join(jobs_dir(ws_dir, QUEUED), job["id"] + ".json")
    write_job(p, job)
    return p


def pending(ws_dir: str) -> int:
    """큐에 있거나 돌고 있는 작업 수. 스튜디오가 중복 요청을 막을 때 쓴다."""
    n = 0
    for st in (QUEUED, RUNNING):
        d = os.path.join(ws_dir, "jobs", st)
        if os.path.isdir(d):
            n += len([f for f in os.listdir(d) if f.endswith(".json")])
    return n


def latest_model(ws_dir: str) -> Optional[dict]:
    """가장 최근에 끝난 작업의 결과. 스튜디오가 새 모델을 감지할 때 쓴다."""
    d = os.path.join(ws_dir, "jobs", DONE)
    if not os.path.isdir(d):
        return None
    fs = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    for fn in reversed(fs):
        j = read_job(os.path.join(d, fn))
        if j and j.get("ok"):
            return j
    return None


def requeue_stale(ws_dir: str) -> int:
    """워커가 죽어서 running 에 남은 것을 큐로 되돌린다 (기동 시 1회)."""
    src, dst = jobs_dir(ws_dir, RUNNING), jobs_dir(ws_dir, QUEUED)
    n = 0
    for fn in os.listdir(src):
        if not fn.endswith(".json"):
            continue
        try:
            os.rename(os.path.join(src, fn), os.path.join(dst, fn))
            n += 1
        except OSError:
            pass
    return n


def claim(ws_dir: str) -> Optional[Tuple[str, dict]]:
    """큐에서 가장 최근 요청 하나를 집어간다. 나머지는 superseded 로 보낸다.

    os.rename 이 원자적이라 워커가 둘이어도 한 쪽만 성공한다.
    """
    qd = jobs_dir(ws_dir, QUEUED)
    fs = sorted(f for f in os.listdir(qd) if f.endswith(".json"))
    if not fs:
        return None
    newest, older = fs[-1], fs[:-1]
    for fn in older:
        job = read_job(os.path.join(qd, fn)) or {"id": fn[:-5]}
        job.update(ok=False, superseded_by=newest[:-5],
                   finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        write_job(os.path.join(jobs_dir(ws_dir, DONE), fn), job)
        try:
            os.remove(os.path.join(qd, fn))
        except OSError:
            pass
    dst = os.path.join(jobs_dir(ws_dir, RUNNING), newest)
    try:
        os.rename(os.path.join(qd, newest), dst)
    except OSError:
        return None  # 다른 워커가 먼저 집어갔다
    job = read_job(dst)
    return (dst, job) if job else None


# ------------------------------------------------------------------ 실행
def run(cmd: List[str], log_path: str) -> int:
    env = dict(os.environ, YOLO_AUTOINSTALL="false", PYTHONIOENCODING="utf-8")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        return subprocess.call(cmd, cwd=DATA, env=env, stdout=log,
                               stderr=subprocess.STDOUT)


def model_name(job: dict, ws_name: str) -> str:
    return job.get("model_name") or "%s_v%d" % (ws_name, int(job.get("version", 1)))


def execute(ws_dir: str, ws_name: str, path: str, job: dict) -> dict:
    """학습 -> 예측 생성. 결과를 job 에 채워 done 으로 옮긴다."""
    t0 = time.perf_counter()
    name = model_name(job, ws_name)
    logs = os.path.join(ws_dir, "jobs", "logs")
    os.makedirs(logs, exist_ok=True)
    log_path = os.path.join(logs, job["id"] + ".log")
    job["model"] = name
    job["log"] = log_path

    cmd = [PYTHON, "train_model.py",
           "--labels", ws_name,
           "--name", name,
           "--split", job.get("split", "train"),
           "--mode", "all",
           "--base", job.get("base", "yolo11s.pt"),
           "--epochs", str(job.get("epochs", 80)),
           "--seed", str(job.get("seed", 0)),
           "--val-fast-n", str(job.get("val_fast_n", 400)),
           "--val-period", str(job.get("val_period", 5)),
           "--overwrite"]
    if job.get("dataset"):
        cmd += ["--dataset", job["dataset"]]
    if job.get("note"):
        cmd += ["--note", job["note"]]

    rc = run(cmd, log_path)
    job["train_seconds"] = round(time.perf_counter() - t0, 1)
    if rc != 0:
        job.update(ok=False, error="train_model.py 종료코드 %d" % rc)
        return job

    mj = os.path.join(DATA, "models", name, "model.json")
    meta = read_job(mj) or {}
    job["metrics"] = meta.get("metrics")
    job["trained_images"] = meta.get("trained_images")

    # ---- 예측 생성 (스플릿 전량. 학습에 쓴 이미지는 make_preds 가 뺀다) ----
    t1 = time.perf_counter()
    rc = run([PYTHON, "make_preds.py", name, "--split", job.get("split", "train")],
             log_path)
    job["pred_seconds"] = round(time.perf_counter() - t1, 1)
    if rc != 0:
        job.update(ok=False, error="make_preds.py 종료코드 %d" % rc)
        return job

    job["pred_dir"] = os.path.join(DATA, "models", name, "preds",
                                   job.get("split", "train"))
    job["ok"] = True
    return job


def finish(ws_dir: str, path: str, job: dict) -> None:
    job["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_job(os.path.join(jobs_dir(ws_dir, DONE), os.path.basename(path)), job)
    try:
        os.remove(path)
    except OSError:
        pass


# ------------------------------------------------------------------ 감시 루프
def workspaces(root: str, only: Optional[str]) -> List[Tuple[str, str]]:
    wd = os.path.join(root, "work")
    if not os.path.isdir(wd):
        return []
    out = []
    for n in sorted(os.listdir(wd)):
        if only and n != only:
            continue
        p = os.path.join(wd, n)
        if os.path.isfile(os.path.join(p, "workspace.json")):
            out.append((n, p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=None, help="이 작업공간만 감시")
    ap.add_argument("--once", action="store_true", help="큐를 한 번 비우고 종료")
    ap.add_argument("--poll", type=float, default=POLL_SEC)
    ap.add_argument("--root", default=None,
                    help="감시할 작업 루트 (기본: 이 스크립트가 있는 폴더)")
    a = ap.parse_args()

    #  스튜디오가 다른 루트를 열고 있을 수 있다. 그때는 스튜디오가 --root 로
    #  자기 루트를 넘긴다. .venv 는 스크립트 옆에 있으므로 PYTHON 은 그대로 둔다.
    global DATA
    if a.root:
        DATA = os.path.abspath(a.root)

    wss = workspaces(DATA, a.workspace)
    if not wss:
        print("[!] 감시할 작업공간이 없습니다.")
        return 1
    print("train_worker 시작 — 작업공간 %d개 · %.1f초마다 확인" % (len(wss), a.poll))
    for n, p in wss:
        k = requeue_stale(p)
        if k:
            print("  [%s] 중단된 작업 %d개를 큐로 되돌렸습니다." % (n, k))
    print("  " + ", ".join(n for n, _ in wss))
    sys.stdout.flush()

    idle = True
    _consume_stop(DATA)          # 지난 세션이 남긴 신호가 새 워커를 죽이면 안 된다
    start_beating(DATA)
    beat(DATA, "idle")
    while True:
        if _consume_stop(DATA):
            print("\n멈추라는 신호를 받았습니다. 종료합니다.", flush=True)
            try:
                os.remove(os.path.join(DATA, "work", HEARTBEAT))
            except OSError:
                pass
            return 0
        did = False
        for ws_name, ws_dir in workspaces(DATA, a.workspace):
            got = claim(ws_dir)
            if not got:
                continue
            did, idle = True, False
            path, job = got
            beat(DATA, "training", "%s · %s장" % (ws_name, job.get("n_confirmed", "?")))
            print("\n[%s] %s 시작 — 확정 %s장"
                  % (ws_name, job["id"], job.get("n_confirmed", "?")), flush=True)
            try:
                job = execute(ws_dir, ws_name, path, job)
            except Exception as e:                      # 워커는 죽으면 안 된다
                job.update(ok=False, error="%s: %s" % (type(e).__name__, e))
            finish(ws_dir, path, job)
            if job.get("ok"):
                m = (job.get("metrics") or {}).get("mAP50")
                print("[%s] %s 완료 — %s · mAP50 %s · 학습 %.1f분 · 예측 %.0f초"
                      % (ws_name, job["id"], job.get("model"), m,
                         job["train_seconds"] / 60, job.get("pred_seconds", 0)),
                      flush=True)
            else:
                print("[%s] %s 실패 — %s (로그: %s)"
                      % (ws_name, job["id"], job.get("error"), job.get("log")),
                      flush=True)
        if did:
            beat(DATA, "idle")
        if a.once and not did:
            return 0
        if not did:
            if not idle:
                print("대기 중…", flush=True)
                idle = True
            time.sleep(a.poll)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단합니다.")
        sys.exit(130)
