"""
사전학습 립리딩 웹사이트 (Gradio)

실행 (서버를 여는 사람만):
  python web_app.py                 # 내 컴퓨터에서만 접속 (http://127.0.0.1:7860)
  python web_app.py --share         # 다른 사람도 접속할 수 있는 임시 공개 링크(https://xxxx.gradio.live, 72시간)
  python web_app.py --cpu --beam 10 # CPU에서 빠르게

접속하는 사람은:
  1) 링크 열기  2) 자기 OpenAI API 키 입력 (또는 키 없이 시작)  3) 웹캠으로 녹화 → 결과 확인
  설치할 것이 없다.

데이터 흐름
  브라우저 웹캠 영상 → 이 앱을 실행 중인 컴퓨터(서버)로 업로드 → 얼굴 기준점·입 영역·립리딩 모델 (서버에서 처리)
  → 후보 문장 '텍스트'만 OpenAI로 전송 (보정 + 한국어 번역)
  업로드된 영상은 서버 임시 폴더에서 몇 분 안에 자동 삭제된다.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from face_points import FacePointTracker, resample_to_fps  # noqa: E402
from gpt_refine import Refiner  # noqa: E402
from vsr_live import CONFIGS, MODEL_ZOO, VSRModel, check_files, crop_strip, ensure_utf8_mode  # noqa: E402

MAX_SEC = 10.0
MAX_WIDTH = 640
NBEST = 5
MODEL: VSRModel | None = None
LANG = "en"
LANG_LABEL = ""

GREEN = (34, 197, 94)     # RGB
LIGHT = (187, 247, 208)


# ---------------------------------------------------------------------------
# 영상 파일 → 프레임 (브라우저 녹화는 webm, 업로드는 mp4 등 → PyAV로 통일해서 읽음)
# ---------------------------------------------------------------------------
def decode_video(path: str):
    import av

    frames, times = [], []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate) if stream.average_rate else 25.0
        for i, frame in enumerate(container.decode(stream)):
            t = frame.time if frame.time is not None else i / fps
            if t > MAX_SEC + 0.5:
                break
            rgb = frame.to_ndarray(format="rgb24")
            h, w = rgb.shape[:2]
            if w > MAX_WIDTH:
                rgb = cv2.resize(rgb, (MAX_WIDTH, int(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
            times.append(float(t))
    if times:
        times = list(np.maximum.accumulate(np.asarray(times)))  # 간혹 뒤섞인 타임스탬프 보정
        if times[-1] - times[0] < 1e-6:  # 타임스탬프가 없는 영상
            times = [i / fps for i in range(len(frames))]
    return frames, times


# ---------------------------------------------------------------------------
# FFmpeg 없이 쓰는 웹캠 영상 입력
#   gr.Video(include_audio=False)는 소리를 빼려고 FFmpeg 프로그램을 호출하는데 Windows에는 보통 없다.
#   브라우저가 이미 소리 없이 녹화하고, 우리는 PyAV로 프레임만 읽으므로 변환 없이 경로만 받는다.
# ---------------------------------------------------------------------------
def make_video_input(**kwargs):
    import types

    import gradio as gr

    video = gr.Video(**kwargs)

    def preprocess(self, payload):  # FFmpeg 변환 없이 업로드된 파일 경로를 그대로 넘김
        if payload is None:
            return None
        return str(payload.video.path)

    # 클래스를 상속하면 Gradio가 같은 폴더에 web_app.pyi 파일을 자동으로 만들어서,
    # 상속 대신 이 컴포넌트 하나의 preprocess만 바꾼다.
    video.preprocess = types.MethodType(preprocess, video)
    return video


# ---------------------------------------------------------------------------
# 입모양 인식 화면 (초록색 입술 표시) 만들기
# ---------------------------------------------------------------------------
def annotate(frame: np.ndarray, lips: np.ndarray | None) -> np.ndarray:
    out = frame.copy()
    if lips is None:
        cv2.putText(out, "NO FACE", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (239, 68, 68), 2, cv2.LINE_AA)
        return out
    pts = lips.astype(np.int32)
    x0, y0 = pts.min(0) - 18
    x1, y1 = pts.max(0) + 18
    t = max(2, out.shape[1] // 320)
    cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), GREEN, t, cv2.LINE_AA)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, LIGHT, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(out, tuple(int(v) for v in p), t + 1, GREEN, -1, cv2.LINE_AA)
    return out


OVERLAY_WIDTH = 480  # 결과 영상 폭 (작을수록 브라우저로 돌려보내는 용량이 줄어듦)


def encode_webm(frames: list[np.ndarray], fps: int = 25) -> bytes | None:
    """브라우저에서 바로 재생되는 webm(VP8)으로 인코딩. FFmpeg 프로그램 없이 PyAV만 사용."""
    import av

    try:
        h0, w0 = frames[0].shape[:2]
        if w0 > OVERLAY_WIDTH:
            nh = int(h0 * OVERLAY_WIDTH / w0)
            frames = [cv2.resize(f, (OVERLAY_WIDTH, nh), interpolation=cv2.INTER_AREA) for f in frames]
        h, w = frames[0].shape[:2]
        h2, w2 = h - h % 2, w - w % 2
        buf = io.BytesIO()
        with av.open(buf, mode="w", format="webm") as c:
            s = c.add_stream("libvpx", rate=fps)
            s.width, s.height, s.pix_fmt = w2, h2, "yuv420p"
            s.bit_rate = 500_000
            s.options = {"deadline": "realtime", "cpu-used": "8"}
            for f in frames:
                vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:h2, :w2]), format="rgb24")
                for p in s.encode(vf):
                    c.mux(p)
            for p in s.encode():
                c.mux(p)
        return buf.getvalue()
    except Exception as e:  # 코덱이 없는 환경이면 이미지로 대체
        print(f"[알림] 인식 화면 영상 인코딩 실패, 이미지로 대신 표시합니다: {e}")
        return None


def overlay_html(frames: list[np.ndarray]) -> str:
    data = encode_webm(frames)
    if data:
        src = "data:video/webm;base64," + base64.b64encode(data).decode()
        media = f'<video src="{src}" autoplay loop muted playsinline controls></video>'
    else:  # 대체: 대표 프레임 4장
        idx = np.linspace(0, len(frames) - 1, 4).astype(int)
        tiles = np.hstack([cv2.resize(frames[i], (240, int(240 * frames[i].shape[0] / frames[i].shape[1]))) for i in idx])
        ok, png = cv2.imencode(".jpg", cv2.cvtColor(tiles, cv2.COLOR_RGB2BGR))
        media = f'<img src="data:image/jpeg;base64,{base64.b64encode(png.tobytes()).decode()}"/>'
    return (f'<div class="overlay">{media}</div>'
            '<div class="legend"><span class="dot"></span> 초록색 = 인식한 입술 · 네모 안쪽을 모델이 읽어요</div>')


# ---------------------------------------------------------------------------
# 결과 HTML
# ---------------------------------------------------------------------------
def esc(s) -> str:
    return html.escape(str(s))


def result_html(hyps, gpt, gpt_err, warn: str = "") -> str:
    if gpt:
        conf = gpt["confidence"]
        cls = {"높음": "high", "중간": "mid", "낮음": "low"}.get(conf, "mid")
        body = f"""
        <div class="label">인식한 문장</div>
        <div class="result-en">{esc(gpt['best_sentence'])}</div>
        <div class="label">한국어 번역</div>
        <div class="result-ko">{esc(gpt['korean_translation'])}</div>
        <div class="meta"><span class="chip {cls}">신뢰도 {esc(conf)}</span><span class="reason">{esc(gpt['reason'])}</span></div>
        <div class="sub">모델 1순위 원문 · <i>{esc(hyps[0][0])}</i></div>"""
    else:
        note = (f"GPT 호출 실패 · {esc(gpt_err)}" if gpt_err
                else "API 키 없이 실행 중이라 한국어 번역과 문장 보정은 표시되지 않아요.")
        body = f"""
        <div class="label">인식한 문장 (모델 1순위)</div>
        <div class="result-en">{esc(hyps[0][0])}</div>
        <div class="sub">{note}</div>"""
    if warn:
        body += f'<div class="warn">{esc(warn)}</div>'
    return f'<div class="card result">{body}</div>'


def candidates_html(hyps, ko) -> str:
    scores = np.array([s for _, s in hyps], dtype=float)
    prob = np.exp(scores - scores.max())
    prob = prob / prob.sum()
    rows = []
    for i, ((t, s), k, p) in enumerate(zip(hyps, ko, prob), 1):
        rows.append(f"""
        <tr class="{'top' if i == 1 else ''}">
          <td class="rank">{i}</td>
          <td class="en">{esc(t)}</td>
          <td class="ko">{esc(k)}</td>
          <td class="bar"><div class="track"><div class="fill" style="width:{p * 100:.0f}%"></div></div>
              <span>{p * 100:.0f}%</span></td>
        </tr>""")
    return f"""
    <div class="card">
      <div class="section-title">모델이 읽은 후보 문장</div>
      <div class="muted small">입모양만으로 만든 후보예요. 막대는 후보끼리 비교한 모델의 상대적인 확신도예요.</div>
      <table class="cands">
        <thead><tr><th>#</th><th>후보 문장</th><th>한국어 번역</th><th>확신도</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>"""


EMPTY_OVERLAY = ('<div class="placeholder">🎥<br>녹화 후 <b>입모양 읽기</b>를 누르면<br>'
                 '인식한 입술이 <span class="g">초록색</span>으로 표시돼요</div>')
EMPTY_RESULT = '<div class="card result empty">결과가 여기에 크게 표시돼요</div>'


# ---------------------------------------------------------------------------
# 이벤트 처리
# ---------------------------------------------------------------------------
def banner_html(text: str, kind: str) -> str:
    return f'<div class="banner {kind}">{text}</div>'


def handle_key(key: str, model_name: str):
    import gradio as gr

    key = (key or "").strip()
    if not key:
        return (None, gr.update(), gr.update(), "⚠️ API 키를 입력하거나 **API 없이 시작**을 눌러 주세요.", gr.update(), gr.update())
    r = Refiner(key, (model_name or "gpt-4o-mini").strip())
    try:
        info = r.check()
    except Exception as e:
        return (None, gr.update(), gr.update(), f"❌ 키 확인 실패 · `{type(e).__name__}` · 키를 다시 확인해 주세요.", gr.update(), gr.update())
    state = {"key": key, "model": r.model}
    banner = banner_html(f"🔑 API 연결됨 · <code>{esc(r.model)}</code> · 사용 가능한 모델 {info['model_count']}개 · "
                         f"인식 언어 <b>{esc(LANG_LABEL)}</b>", "ok")
    return (state, gr.update(visible=False), gr.update(visible=True), "", banner, gr.update(value=""))


def handle_skip():
    import gradio as gr

    banner = banner_html(f"⚪ API 없이 사용 중 · 영어 받아쓰기만 표시돼요 · 인식 언어 <b>{esc(LANG_LABEL)}</b>", "off")
    return ({}, gr.update(visible=False), gr.update(visible=True), "", banner, gr.update(value=""))


def handle_change_key():
    import gradio as gr

    return None, gr.update(visible=True), gr.update(visible=False)


def handle_video(video_path, state):
    import traceback

    import gradio as gr

    try:
        return _handle_video(video_path, state)
    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()  # 터미널에 자세한 원인 출력
        raise gr.Error(f"처리 중 오류: {type(e).__name__}: {e}  (자세한 내용은 터미널 확인)")


def _handle_video(video_path, state):
    import gradio as gr

    if not video_path:
        raise gr.Error("먼저 웹캠으로 녹화하거나 영상을 올려 주세요.")
    t_all = time.time()

    frames, times = decode_video(video_path)
    if len(frames) < 10:
        raise gr.Error("영상이 너무 짧아요. 1~4초 길이로 한 문장을 말해 주세요.")

    tracker = FacePointTracker()
    try:
        tracked = [tracker.process(f, ts_ms=t * 1000.0) for f, t in zip(frames, times)]  # (4점, 입술 20점)
    finally:
        tracker.close()
    found = sum(p is not None for p, _ in tracked)
    if found == 0:
        raise gr.Error("얼굴을 찾지 못했어요. 밝은 곳에서 얼굴 정면이 보이게 다시 녹화해 주세요.")

    video25, tracked25 = resample_to_fps(frames, tracked, times, 25.0)
    pts25 = [p for p, _ in tracked25]
    lips25 = [l for _, l in tracked25]

    t0 = time.time()
    crops = MODEL.mouth_crops(video25, pts25)
    if crops is None:
        raise gr.Error("입 영역을 자르지 못했어요. 다시 녹화해 주세요.")
    hyps = MODEL.transcribe(crops, NBEST)
    infer_sec = time.time() - t0
    if not hyps:
        raise gr.Error("모델이 문장을 만들지 못했어요. 다시 시도해 주세요.")

    gpt, gpt_err, api_meta = None, None, None
    if state and state.get("key"):
        try:
            r = Refiner(state["key"], state.get("model", "gpt-4o-mini"))
            gpt = r.refine(LANG, hyps)
            api_meta = r.last_meta
        except Exception as e:
            gpt_err = f"{type(e).__name__}: {e}"

    warn = "녹화 중 얼굴을 놓친 구간이 많아요. 정면을 보고 다시 해 보면 더 정확해져요." if found < len(frames) * 0.6 else ""
    ko = gpt["nbest_korean"] if gpt else ["–"] * len(hyps)
    overlay = overlay_html([annotate(f, l) for f, l in zip(video25, lips25)])
    strip = cv2.cvtColor(crop_strip(crops, n=8, size=110), cv2.COLOR_BGR2RGB)
    info = {
        "처리": {
            "입력 프레임": len(frames),
            "얼굴 찾은 프레임": found,
            "모델 입력 프레임(25fps)": len(crops),
            "립리딩 추론 시간(초)": round(infer_sec, 2),
            "전체 처리 시간(초)": round(time.time() - t_all, 2),
        },
        "OpenAI API": api_meta if api_meta else ("사용 안 함" if not gpt_err else gpt_err),
    }
    return overlay, result_html(hyps, gpt, gpt_err, warn), candidates_html(hyps, ko), strip, info


# ---------------------------------------------------------------------------
# 화면
# ---------------------------------------------------------------------------
CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
footer { display: none !important; }
#hero { text-align: center; padding: 34px 0 10px; }
#hero .badge { display:inline-block; font-size:.8rem; font-weight:600; letter-spacing:.04em; color:#6366f1;
  background: rgba(99,102,241,.10); padding: 5px 12px; border-radius: 999px; }
#hero h1 { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; margin: 14px 0 6px;
  background: linear-gradient(90deg,#4f46e5,#a855f7 55%,#ec4899); -webkit-background-clip: text; background-clip:text; color: transparent; }
#hero p { font-size: 1.05rem; color: var(--body-text-color-subdued); margin: 0; }

.card { border: 1px solid var(--border-color-primary); border-radius: 18px; padding: 22px 24px;
  background: var(--block-background-fill); box-shadow: 0 1px 2px rgba(0,0,0,.04), 0 8px 24px rgba(15,23,42,.04); }
#keycard { max-width: 560px; margin: 18px auto 0; }
.step-title { font-size: 1.15rem; font-weight: 700; margin-bottom: 4px; }
.muted { color: var(--body-text-color-subdued); }
.small { font-size: .88rem; }

.banner { border-radius: 12px; padding: 10px 14px; font-size: .92rem; }
.banner.ok { background: rgba(34,197,94,.10); border: 1px solid rgba(34,197,94,.30); }
.banner.off { background: rgba(148,163,184,.12); border: 1px solid rgba(148,163,184,.35); }
.banner code { font-size: .85rem; }

.panel-title { display:flex; align-items:center; gap:10px; font-weight:700; font-size:1.05rem; margin-bottom: 6px; }
.panel-title .num { width:26px; height:26px; border-radius:999px; display:inline-flex; align-items:center; justify-content:center;
  background:#4f46e5; color:#fff; font-size:.85rem; }
.panel-title .hint { font-weight:400; font-size:.85rem; color: var(--body-text-color-subdued); }

.overlay video, .overlay img { width: 100%; border-radius: 14px; display:block; background:#000; }
.legend { margin-top: 8px; font-size: .88rem; color: var(--body-text-color-subdued); }
.legend .dot { display:inline-block; width:10px; height:10px; border-radius:999px; background:#22c55e; margin-right:4px; }
.placeholder { min-height: 300px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center;
  border: 2px dashed var(--border-color-primary); border-radius: 14px; color: var(--body-text-color-subdued); line-height:1.7; font-size: 1rem; }
.placeholder .g { color:#16a34a; font-weight:700; }

.result { padding: 28px 30px; }
.result.empty { text-align:center; color: var(--body-text-color-subdued); padding: 34px; font-size: 1.05rem; }
.result .label { font-size: .8rem; font-weight: 700; letter-spacing: .06em; color: var(--body-text-color-subdued); margin-top: 6px; }
.result-en { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.25; margin: 4px 0 14px; }
.result-ko { font-size: 1.8rem; font-weight: 700; color: #4f46e5; line-height: 1.3; margin: 4px 0 14px; }
.dark .result-ko { color: #a5b4fc; }
.result .meta { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-top: 4px; }
.chip { font-size: .85rem; font-weight: 700; padding: 4px 12px; border-radius: 999px; }
.chip.high { background: rgba(34,197,94,.15); color:#15803d; }
.chip.mid  { background: rgba(234,179,8,.18); color:#a16207; }
.chip.low  { background: rgba(239,68,68,.15); color:#b91c1c; }
.reason { color: var(--body-text-color-subdued); font-size: .95rem; }
.sub { margin-top: 12px; font-size: .9rem; color: var(--body-text-color-subdued); }
.warn { margin-top: 12px; padding: 10px 12px; border-radius: 10px; background: rgba(234,179,8,.12); font-size: .9rem; }

.section-title { font-size: 1.2rem; font-weight: 800; margin-bottom: 2px; }
table.cands { width: 100%; border-collapse: collapse; margin-top: 14px; border: none !important; }
table.cands th, table.cands td { border-left: none !important; border-right: none !important; border-top: none !important; }
table.cands tr, table.cands thead, table.cands tbody { border: none !important; }
table.cands th { text-align: left; font-size: .82rem; font-weight: 700; color: var(--body-text-color-subdued);
  padding: 10px 14px; border-bottom: 2px solid var(--border-color-primary); }
table.cands td { padding: 16px 14px; border-bottom: 1px solid var(--border-color-primary); vertical-align: middle; }
table.cands td.rank { width: 40px; font-weight: 700; color: var(--body-text-color-subdued); }
table.cands td.en { font-size: 1.35rem; font-weight: 600; }
table.cands td.ko { font-size: 1.2rem; }
table.cands tr.top td { background: rgba(99,102,241,.07); }
table.cands tr.top td.en { color: #4f46e5; font-weight: 800; }
.dark table.cands tr.top td.en { color: #a5b4fc; }
table.cands td.bar { width: 170px; white-space: nowrap; }
.track { display:inline-block; vertical-align: middle; width: 110px; height: 8px; border-radius: 999px; background: rgba(148,163,184,.25); overflow:hidden; }
.fill { height: 100%; background: linear-gradient(90deg,#6366f1,#a855f7); border-radius: 999px; }
table.cands td.bar span { margin-left: 8px; font-size: .9rem; color: var(--body-text-color-subdued); }

#run-btn { font-size: 1.12rem !important; padding: 14px !important; border-radius: 12px !important; border: none !important;
  background: linear-gradient(90deg,#4f46e5,#7c3aed) !important; color: #fff !important; box-shadow: 0 6px 18px rgba(79,70,229,.25) !important; }
#run-btn:hover { filter: brightness(1.06); }
#change-btn { align-self: center; min-width: 120px !important; }
.privacy { font-size: .88rem; color: var(--body-text-color-subdued); }
.privacy table { width:100%; font-size:.86rem; }
"""

HERO = """<div id="hero">
  <span class="badge">VISUAL SPEECH RECOGNITION</span>
  <h1>입모양으로 받아쓰기</h1>
  <p>소리 없이 입모양만 녹화하면 AI가 영어 문장으로 읽고, 한국어로 번역해 드려요</p>
</div>"""

PRIVACY = """<div class="privacy"><details><summary>🔒 데이터는 어디로 가나요?</summary>

| 데이터 | 어디서 처리 | 보관 |
|---|---|---|
| 웹캠 영상 | 이 사이트를 운영 중인 컴퓨터(서버) | 처리 후 몇 분 안에 임시 파일 자동 삭제 |
| 입모양 인식 | 서버의 사전학습 모델 | 저장 안 함 |
| 후보 문장 **텍스트** | OpenAI (보정·번역) | 영상은 보내지 않음 |
| API 키 | 브라우저 세션 동안 서버 메모리 | 저장·기록 안 함, 탭을 닫으면 사라짐 |

요금은 입력한 키의 주인에게 청구됩니다. 한 번 번역에 수백 토큰 정도예요.
</details></div>"""

TIPS = ("얼굴 정면 · 밝은 조명 · 입술이 크게 보이게 · 1~4초 길이의 짧은 영어 문장을 또박또박 "
        "(예: <i>Thank you very much</i> · <i>How are you today</i> · <i>See you tomorrow</i>)")


def build_ui():
    import gradio as gr

    theme = gr.themes.Soft(
        primary_hue="indigo", secondary_hue="violet", neutral_hue="slate", radius_size="lg",
        font=[gr.themes.GoogleFont("Noto Sans KR"), "ui-sans-serif", "system-ui", "sans-serif"],
    )
    with gr.Blocks(title="입모양으로 받아쓰기 · AI 립리딩", theme=theme, css=CSS, delete_cache=(300, 300)) as demo:
        state = gr.State(None)
        gr.HTML(HERO)

        # ---- 1) 접속하면 먼저 뜨는 API 키 입력 화면 ----
        with gr.Column(visible=True, elem_id="keycard", elem_classes="card") as key_panel:
            gr.HTML('<div class="step-title">🔑 OpenAI API 키를 입력해 주세요</div>'
                    '<div class="muted small">문장 보정과 한국어 번역에 사용돼요. 키는 저장되지 않고 탭을 닫으면 사라져요.</div>')
            key_box = gr.Textbox(show_label=False, type="password", placeholder="sk-...", container=False)
            with gr.Accordion("모델 선택 (선택 사항)", open=False):
                model_box = gr.Textbox(label="OpenAI 모델", value="gpt-4o-mini")
            with gr.Row():
                start_btn = gr.Button("키 확인하고 시작", variant="primary")
                skip_btn = gr.Button("API 없이 시작", variant="secondary")
            key_msg = gr.Markdown("")
            gr.Markdown(PRIVACY)

        # ---- 2) 메인 화면 ----
        with gr.Column(visible=False) as main_panel:
            with gr.Row(equal_height=True):
                banner = gr.HTML("")
                change_btn = gr.Button("API 키 바꾸기", size="sm", scale=0, variant="secondary", elem_id="change-btn")

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, elem_classes="card"):
                    gr.HTML('<div class="panel-title"><span class="num">1</span>녹화하기'
                            '<span class="hint">● 시작 → 말하기 → ■ 정지</span></div>')
                    video = make_video_input(
                        show_label=False, sources=["webcam", "upload"], include_audio=False, height=360,
                        # 모델은 입 주변 96x96만 쓰므로 640x480이면 충분. 해상도를 낮춰 업로드 용량을 줄인다.
                        webcam_options=gr.WebcamOptions(mirror=True, constraints={
                            "video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                                      "frameRate": {"ideal": 25, "max": 30}},
                            "audio": False,
                        }),
                    )
                    run_btn = gr.Button("👄  입모양 읽기", variant="primary", elem_id="run-btn")
                    gr.HTML(f'<div class="muted small">💡 {TIPS}</div>')
                with gr.Column(scale=1, elem_classes="card"):
                    gr.HTML('<div class="panel-title"><span class="num">2</span>입모양 인식 화면'
                            '<span class="hint">AI가 본 입술</span></div>')
                    overlay = gr.HTML(EMPTY_OVERLAY)

            result = gr.HTML(EMPTY_RESULT)
            cands = gr.HTML("")
            with gr.Accordion("👁️ 모델이 실제로 본 입 영역 (시간 순서)", open=False):
                strip = gr.Image(show_label=False, interactive=False, show_download_button=False)
            with gr.Accordion("🔍 처리 정보 · API로 주고받은 정보", open=False):
                info = gr.JSON(show_label=False)
            gr.Markdown(PRIVACY)

        # ---- 이벤트 ----
        key_outputs = [state, key_panel, main_panel, key_msg, banner, key_box]
        start_btn.click(handle_key, [key_box, model_box], key_outputs)
        key_box.submit(handle_key, [key_box, model_box], key_outputs)
        skip_btn.click(handle_skip, None, key_outputs)
        change_btn.click(handle_change_key, None, [state, key_panel, main_panel])
        run_btn.click(handle_video, [video, state], [overlay, result, cands, strip, info], show_progress="full")
    return demo


def main():
    ensure_utf8_mode()
    ap = argparse.ArgumentParser(description="립리딩 웹사이트")
    ap.add_argument("--vsr-root", default=str(HERE / "Visual_Speech_Recognition_for_Multiple_Languages"))
    ap.add_argument("--lang", choices=list(CONFIGS), default="en")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--beam", type=int, default=None)
    ap.add_argument("--share", action="store_true", help="다른 사람도 접속 가능한 임시 공개 링크 만들기 (72시간)")
    ap.add_argument("--host", default="127.0.0.1", help="같은 네트워크의 다른 기기에서 접속하려면 0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    global MODEL, LANG, LANG_LABEL
    vsr_root = Path(args.vsr_root).resolve()
    if not (vsr_root / "pipelines").exists():
        sys.exit(f"저장소를 찾을 수 없어요: {vsr_root}")
    label, config_rel = CONFIGS[args.lang]
    missing = check_files(vsr_root, config_rel)
    if missing:
        sys.exit("모델 파일이 없어요:\n" + "\n".join(f"  - {vsr_root / m}" for m in missing) + f"\n다운로드: {MODEL_ZOO}")

    import torch
    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    beam = args.beam or (10 if device == "cpu" else None)
    print(f"{label} 모델을 불러오는 중... (장치: {device}, 빔: {beam or 40}) — 30초~1분 걸릴 수 있어요")
    MODEL = VSRModel(vsr_root, config_rel, device, beam)
    LANG, LANG_LABEL = args.lang, label.split(" (")[0]
    print("모델 준비 완료. 브라우저가 자동으로 열려요.")

    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share, inbrowser=True, show_error=True,
    )


if __name__ == "__main__":
    main()
