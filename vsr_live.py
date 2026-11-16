"""
사전학습 립리딩 모델로 웹캠 입모양 → 문장 받아쓰기

  python vsr_live.py --vsr-root ./Visual_Speech_Recognition_for_Multiple_Languages
  python vsr_live.py --vsr-root ... --lang en --cpu
  python vsr_live.py --vsr-root ... --video my_clip.mp4      # 녹화된 영상 파일로 실행

흐름
  웹캠 녹화(SPACE~SPACE) → MediaPipe로 얼굴 기준점 4개 → 25fps로 맞춤
  → 저장소의 전처리(얼굴 정렬 + 입 주변 96x96 흑백 자르기 → 88x88)
  → 사전학습 모델(Conformer 인코더 + Transformer 디코더 + 언어모델, 빔서치)로 상위 N개 문장
  → (선택) GPT가 입모양 혼동을 고려해 문장 보정 + 한국어 번역
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from face_points import FacePointTracker, resample_to_fps  # noqa: E402
from ui_text import draw_lines  # noqa: E402

CONFIGS = {
    "en": ("English (LRS3, 단어 오류율 약 19%)", "configs/LRS3_V_WER19.1.ini"),
    "es": ("Español (오류율 약 45%)", "configs/CMUMOSEAS_V_ES_WER44.5.ini"),
    "pt": ("Português (오류율 약 51%)", "configs/CMUMOSEAS_V_PT_WER51.4.ini"),
    "fr": ("Français (오류율 약 59%)", "configs/CMUMOSEAS_V_FR_WER58.6.ini"),
    "zh": ("中文 (CMLR, 글자 오류율 약 8% — 뉴스 아나운서 데이터)", "configs/CMLR_V_WER8.0.ini"),
}
MODEL_ZOO = "https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages#model-zoo"
MAX_SEC = 8.0
WHITE, YELLOW, GREEN, RED, CYAN, GRAY = (255, 255, 255), (0, 220, 255), (80, 220, 80), (60, 60, 255), (255, 220, 0), (180, 180, 180)


# ---------------------------------------------------------------------------
# 사전학습 모델 불러오기
# ---------------------------------------------------------------------------
def check_files(vsr_root: Path, config_rel: str) -> list[str]:
    cfg = ConfigParser()
    cfg.read(vsr_root / config_rel)
    missing = []
    for key in ("model_path", "model_conf", "rnnlm", "rnnlm_conf"):
        rel = cfg.get("model", key, fallback="")
        if rel and not (vsr_root / rel).exists():
            missing.append(rel)
    return missing


class VSRModel:
    def __init__(self, vsr_root: Path, config_rel: str, device: str, beam: int | None):
        os.chdir(vsr_root)                 # 설정 파일 안의 경로가 저장소 기준 상대경로라서
        sys.path.insert(0, str(vsr_root))
        import torch
        from espnet.asr.asr_utils import parse_hypothesis
        from pipelines.pipeline import InferencePipeline

        self.torch = torch
        self.parse_hypothesis = parse_hypothesis
        # face_track=False → 저장소의 옛 MediaPipe 검출기를 쓰지 않고, 우리가 만든 기준점을 직접 넘긴다
        self.pipe = InferencePipeline(config_rel, detector="mediapipe", face_track=False, device=device)
        if beam:
            self.pipe.model.beam_search.beam_size = beam
        self.device = device

    def mouth_crops(self, video_rgb: np.ndarray, points: list) -> np.ndarray:
        """(T,H,W,3) RGB + 프레임별 4점 → (T,96,96) 흑백 입 영역"""
        return self.pipe.dataloader.video_process(video_rgb, list(points))

    def transcribe(self, crops: np.ndarray, nbest: int = 5) -> list[tuple[str, float]]:
        torch = self.torch
        data = self.pipe.dataloader.video_transform(torch.tensor(crops))
        m = self.pipe.model
        with torch.no_grad():
            enc = m.model.encode(data.to(self.device))
            hyps = m.beam_search(enc)
        out, seen = [], set()
        for h in hyps[: max(nbest * 3, nbest)]:
            text, _, _, score = self.parse_hypothesis(h.asdict(), m.token_list)
            text = text.replace("▁", " ").replace("<eos>", "").strip()
            if text and text not in seen:
                seen.add(text)
                out.append((text, score))
            if len(out) >= nbest:
                break
        return out


# ---------------------------------------------------------------------------
# 녹화 / 영상 파일 읽기
# ---------------------------------------------------------------------------
def read_video_file(path: str, tracker: FacePointTracker):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, points, times = [], [], []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        p4, _ = tracker.process(rgb, ts_ms=i * 1000.0 / fps)
        frames.append(rgb)
        points.append(p4)
        times.append(i / fps)
        i += 1
    cap.release()
    return frames, points, times


def open_camera(index: int):
    import platform
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if platform.system() == "Windows" else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        sys.exit(f"카메라 {index}번을 열 수 없어요. --camera 1 처럼 다른 번호를 시도해 보세요.")
    return cap


def mirror_pts(pts, w):
    p = pts.copy()
    p[:, 0] = w - p[:, 0]
    return p


# ---------------------------------------------------------------------------
# 한 번 받아쓰기
# ---------------------------------------------------------------------------
def run_once(model: VSRModel, frames, points, times, lang, refiner, nbest, save_dir: Path | None):
    if len(frames) < 10 or all(p is None for p in points):
        return {"error": "녹화가 너무 짧거나 얼굴을 찾지 못했어요."}
    video, pts25 = resample_to_fps(frames, points, times, 25.0)
    t0 = time.time()
    crops = model.mouth_crops(video, pts25)
    if crops is None:
        return {"error": "입 영역을 자르지 못했어요."}
    hyps = model.transcribe(crops, nbest)
    result = {"nbest": hyps, "frames": len(crops), "infer_sec": round(time.time() - t0, 2), "crop_preview": crops}
    print(f"\n[모델 결과] {len(crops)}프레임, {result['infer_sec']}초")
    for i, (t, s) in enumerate(hyps, 1):
        print(f"  {i}. {t}   (score {s:.2f})")
    if refiner and hyps:
        try:
            r = refiner.refine(lang, hyps)
            result["gpt"] = r
            print("[후보 번역]")
            for i, ((t, _), k) in enumerate(zip(hyps, r["nbest_korean"]), 1):
                print(f"  {i}. {t}  →  {k}")
            print(f"[GPT 보정] {r['best_sentence']}  → {r['korean_translation']}  (신뢰도 {r['confidence']}: {r['reason']})")
        except Exception as e:
            result["gpt_error"] = str(e)
            print(f"[GPT 오류] {e}")
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        h, w = video.shape[1:3]
        vw = cv2.VideoWriter(str(save_dir / f"{stamp}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
        for f in video:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        with open(save_dir / "transcripts.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"file": f"{stamp}.mp4", "lang": lang, "nbest": hyps, "gpt": result.get("gpt")},
                               ensure_ascii=False) + "\n")
    return result


def crop_strip(crops: np.ndarray, n: int = 6, size: int = 72) -> np.ndarray:
    idx = np.linspace(0, len(crops) - 1, n).astype(int)
    tiles = [cv2.resize(crops[i], (size, size)) for i in idx]
    return cv2.cvtColor(np.hstack(tiles), cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# 실시간 화면
# ---------------------------------------------------------------------------
def live(model, tracker, lang, refiner, args, save_dir):
    cap = open_camera(args.camera)
    win = "Pretrained Lip Reader"
    recording = False
    frames, points, times = [], [], []
    start = 0.0
    last = None
    print("\n창이 뜨면: SPACE = 녹화 시작/끝 (한 문장씩, 최대 8초), Q = 종료")
    print("팁: 얼굴 정면, 밝은 조명, 입술이 잘 보이게, 또박또박 (소리는 안 내도 됩니다)")
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            p4, lips = tracker.process(rgb)
            now = time.time()
            if recording:
                frames.append(rgb)
                points.append(p4)
                times.append(now)
            view = cv2.flip(bgr, 1)  # 보기 편하게 거울 화면 (모델에는 원본 방향을 넣음)
            w = view.shape[1]
            if lips is not None:
                color = (0, 0, 255) if recording else (0, 255, 0)
                for p in mirror_pts(lips, w).astype(int):
                    cv2.circle(view, tuple(p), 2, color, -1)
            if recording:
                cv2.circle(view, (w - 30, 30), 12, (0, 0, 255), -1)

            lines = [(f"● 녹화 중 {now - start:.1f}s ... 끝나면 SPACE" if recording else "SPACE를 누르고 한 문장을 말하세요",
                      RED if recording else GREEN)]
            if last:
                if "error" in last:
                    lines.append((last["error"], RED))
                else:
                    g = last.get("gpt")
                    if last["nbest"]:
                        lines.append((f"모델: {last['nbest'][0][0]}", YELLOW))
                        if g and g.get("nbest_korean"):
                            lines.append((f"  └ 번역: {g['nbest_korean'][0]}", YELLOW))
                    if g:
                        lines.append((f"GPT 보정: {g['best_sentence']}", CYAN))
                        lines.append((f"한국어: {g['korean_translation']}  (신뢰도 {g['confidence']})", WHITE))
            if lips is None:
                lines.append(("얼굴이 안 보여요!", RED))
            lines.append(("Q 종료", GRAY))
            canvas = draw_lines(view, lines, "ko", size=22)
            if last and "crop_preview" in last:
                strip = crop_strip(last["crop_preview"])
                sh, sw = strip.shape[:2]
                if sw <= canvas.shape[1]:
                    canvas[-sh:, :sw] = strip  # 모델이 실제로 본 입 영역 미리보기
            cv2.imshow(win, canvas)

            key = cv2.waitKey(1) & 0xFF
            stop = recording and now - start > MAX_SEC
            if key == ord(" ") or stop:
                if not recording and not stop:
                    recording, frames, points, times, start = True, [], [], [], now
                else:
                    recording = False
                    cv2.imshow(win, draw_lines(view, [("받아쓰는 중... (CPU면 수십 초 걸릴 수 있어요)", YELLOW)], "ko", size=22))
                    cv2.waitKey(1)
                    last = run_once(model, frames, points, times, lang, refiner, args.nbest, save_dir)
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def ensure_utf8_mode():
    """
    한국어 Windows는 기본 인코딩이 cp949라서, 사전학습 저장소가 UTF-8 파일(토큰 목록 등)을
    encoding 없이 open() 할 때 UnicodeDecodeError가 난다.
    → Python UTF-8 모드(-X utf8)로 자기 자신을 다시 실행한다.
    """
    if sys.flags.utf8_mode or os.environ.get("VSR_UTF8_RELAUNCHED"):
        return
    import subprocess
    env = dict(os.environ, PYTHONUTF8="1", VSR_UTF8_RELAUNCHED="1")
    rc = subprocess.call([sys.executable, "-X", "utf8", os.path.abspath(sys.argv[0]), *sys.argv[1:]], env=env)
    sys.exit(rc)


def main():
    ensure_utf8_mode()
    ap = argparse.ArgumentParser(description="사전학습 립리딩 모델 웹캠 데모")
    ap.add_argument("--vsr-root", required=True, help="Visual_Speech_Recognition_for_Multiple_Languages 저장소 경로")
    ap.add_argument("--lang", choices=list(CONFIGS))
    ap.add_argument("--video", help="웹캠 대신 영상 파일로 실행")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="GPU가 있어도 CPU로 실행")
    ap.add_argument("--beam", type=int, default=None, help="빔 크기 (기본 40, CPU에서 느리면 10 정도로)")
    ap.add_argument("--nbest", type=int, default=5)
    ap.add_argument("--save", action="store_true", help="녹화 영상과 결과를 recordings/ 에 저장")
    args = ap.parse_args()

    vsr_root = Path(args.vsr_root).resolve()
    video_path = Path(args.video).resolve() if args.video else None  # 모델을 불러오면 작업 폴더가 바뀌므로 미리 절대경로로
    if video_path and not video_path.exists():
        sys.exit(f"영상 파일이 없어요: {video_path}")
    if not (vsr_root / "pipelines").exists():
        sys.exit(f"저장소를 찾을 수 없어요: {vsr_root}\nREADME의 설치 2단계를 확인하세요.")

    lang = args.lang
    if not lang:
        print("언어를 고르세요 (한국어 공개 모델은 아직 없어요)")
        keys = list(CONFIGS)
        for i, k in enumerate(keys, 1):
            print(f"  {i}) {CONFIGS[k][0]}")
        c = input("선택 (Enter=1): ").strip()
        lang = keys[int(c) - 1] if c.isdigit() and 1 <= int(c) <= len(keys) else "en"
    label, config_rel = CONFIGS[lang]

    missing = check_files(vsr_root, config_rel)
    if missing:
        print("\n모델 파일이 없어요. 아래 위치에 압축을 풀어 주세요:")
        for m in missing:
            print("  -", vsr_root / m)
        sys.exit(f"다운로드: {MODEL_ZOO}  (README 3단계 참고)")

    from gpt_refine import Refiner
    refiner = Refiner.ask()

    import torch
    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    beam = args.beam or (10 if device == "cpu" else None)
    print(f"\n{label} 모델을 불러오는 중... (장치: {device}, 빔: {beam or 40})")
    save_dir = (HERE / "recordings") if args.save else None
    model = VSRModel(vsr_root, config_rel, device, beam)
    tracker = FacePointTracker()
    try:
        if args.video:
            frames, points, times = read_video_file(str(video_path), tracker)
            run_once(model, frames, points, times, lang, refiner, args.nbest, save_dir)
        else:
            live(model, tracker, lang, refiner, args, save_dir)
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
