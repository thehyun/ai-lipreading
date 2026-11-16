"""
얼굴 기준점 4개 추출 (MediaPipe Tasks FaceLandmarker 사용)

사전학습 모델 저장소의 전처리 코드는 프레임마다
  [오른쪽 눈(화면 왼쪽), 왼쪽 눈(화면 오른쪽), 코끝, 입 중심]  4개 점을 받아
얼굴을 표준 위치로 정렬한 뒤 입 주변 96x96을 잘라낸다.
원래는 옛 MediaPipe(mp.solutions) API를 쓰는데, 최신 MediaPipe에서는 사라져서
새 Tasks API로 같은 4개 점을 만들어 넘겨준다.
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"

EYE_A = (33, 133)    # 한쪽 눈 양끝
EYE_B = (362, 263)   # 다른 쪽 눈 양끝
NOSE_TIP = 1
MOUTH = (13, 14)     # 윗입술/아랫입술 안쪽 가운데
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        print("MediaPipe 얼굴 모델을 내려받는 중... (최초 1회)")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(MODEL_PATH)
    return MODEL_PATH


def four_points(pts: np.ndarray) -> np.ndarray:
    """(478,2) 픽셀 좌표 → (4,2): 화면 왼쪽 눈, 화면 오른쪽 눈, 코끝, 입 중심"""
    e1 = pts[list(EYE_A)].mean(axis=0)
    e2 = pts[list(EYE_B)].mean(axis=0)
    left_eye, right_eye = (e1, e2) if e1[0] <= e2[0] else (e2, e1)
    nose = pts[NOSE_TIP]
    mouth = pts[list(MOUTH)].mean(axis=0)
    return np.stack([left_eye, right_eye, nose, mouth]).astype(np.float32)


class FacePointTracker:
    def __init__(self):
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=ensure_model().read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self._lm = vision.FaceLandmarker.create_from_options(options)
        self._t0 = time.perf_counter()
        self._last = -1

    def process(self, rgb: np.ndarray, ts_ms: int | None = None):
        """반환: (4점 (4,2) 또는 None, 입술 바깥 점 (20,2) 또는 None)"""
        import mediapipe as mp

        ts = int((time.perf_counter() - self._t0) * 1000) if ts_ms is None else int(ts_ms)
        if ts <= self._last:
            ts = self._last + 1
        self._last = ts
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ts)
        if not res.face_landmarks:
            return None, None
        h, w = rgb.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float32)
        return four_points(pts), pts[LIPS_OUTER]

    def close(self):
        self._lm.close()


def resample_to_fps(frames: list, points: list, times: list[float], fps: float = 25.0):
    """웹캠 실제 프레임 속도(보통 30fps 안팎)를 모델이 학습한 25fps로 맞춘다 (가장 가까운 프레임 선택)."""
    t = np.asarray(times) - times[0]
    targets = np.arange(0, t[-1] + 1e-9, 1.0 / fps)
    idx = np.clip(np.searchsorted(t, targets), 0, len(t) - 1)
    prev = np.clip(idx - 1, 0, len(t) - 1)
    idx = np.where(np.abs(t[prev] - targets) < np.abs(t[idx] - targets), prev, idx)
    return np.stack([frames[i] for i in idx]), [points[i] for i in idx]
