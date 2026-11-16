"""
OpenAI API 사용 부분 (선택 기능)

립리딩 모델이 내놓은 상위 N개 후보 문장을 GPT에게 보내
  1) 입모양이 같은 소리(p/b/m, f/v, t/d/n …) 혼동을 고려해 가장 그럴듯한 문장으로 다듬고
  2) 보정 문장과 모델 후보 문장 전부를 한국어로 번역한다.
영상·얼굴 이미지는 보내지 않고, 후보 '텍스트'만 보낸다.
API 키는 저장하지 않는다 (터미널: 실행할 때마다 입력 / 웹: 접속한 사람이 자기 키 입력).
"""
from __future__ import annotations

import getpass
import json
import time

from openai import OpenAI

LANG_NAME = {"en": "English", "es": "Spanish", "pt": "Portuguese", "fr": "French", "zh": "Mandarin Chinese"}
PREFERRED_MODELS = ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4o")

SYSTEM = """You post-process the output of a visual speech recognition (lip-reading) model.
The model only sees the mouth, so it often confuses sounds that look the same on the lips
(p/b/m, f/v, t/d/n/s/z, k/g/ng, ch/j/sh) and may drop or merge short words.
You get the N-best hypotheses with beam scores (higher = more likely).

Tasks
1. best_sentence: pick the most plausible hypothesis, or minimally edit it by swapping look-alike sounds
   or fixing word boundaries. Do NOT add new content that no hypothesis supports.
   If all hypotheses look like nonsense, keep hypothesis 1 and set confidence to '낮음'.
2. korean_translation: natural Korean translation of best_sentence.
3. nbest_korean: Korean translation of EACH hypothesis exactly as written, same order and same count
   (translate literally even if it is odd; write '(의미 불명)' if it cannot be translated).
4. reason: one short Korean sentence explaining the choice."""

SCHEMA = {
    "type": "object",
    "properties": {
        "best_sentence": {"type": "string"},
        "korean_translation": {"type": "string"},
        "nbest_korean": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["낮음", "중간", "높음"]},
        "reason": {"type": "string"},
    },
    "required": ["best_sentence", "korean_translation", "nbest_korean", "confidence", "reason"],
    "additionalProperties": False,
}


def pick_headers(headers) -> dict:
    return {k.lower(): v for k, v in headers.items()
            if k.lower().startswith(("x-ratelimit-remaining", "x-ratelimit-limit")) or k.lower() == "x-request-id"}


class Refiner:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key, timeout=60)
        self.model = model
        self.last_meta: dict = {}

    # ---------- 키 확인 (터미널·웹 공통) ----------
    def check(self) -> dict:
        """GET /v1/models — 키가 유효한지 확인하고 사용 가능한 모델 목록을 받는다."""
        raw = self.client.models.with_raw_response.list()
        ids = sorted(m.id for m in raw.parse().data)
        if self.model not in ids:
            self.model = next((m for m in PREFERRED_MODELS if m in ids), self.model)
        return {"model_count": len(ids), "model_ids": ids, "headers": pick_headers(raw.headers)}

    # ---------- 터미널용: 키 입력받기 ----------
    @classmethod
    def ask(cls) -> "Refiner | None":
        key = getpass.getpass("OpenAI API 키 (화면에 안 보여요, Enter = GPT 없이 진행): ").strip()
        if not key:
            return None
        r = cls(key)
        try:
            info = r.check()
            print(f"→ 키 확인 완료. 사용 가능한 모델 {info['model_count']}개")
            _print_headers(info["headers"])
        except Exception as e:
            print(f"→ 키 확인 실패, GPT 없이 진행합니다: {e}")
            return None
        m = input(f"사용할 모델 (Enter = {r.model}): ").strip()
        if m:
            r.model = m
        return r

    # ---------- 보정 + 번역 (POST /v1/chat/completions) ----------
    def refine(self, lang: str, nbest: list[tuple[str, float]]) -> dict:
        user = f"Language: {LANG_NAME.get(lang, lang)}\nN-best hypotheses ({len(nbest)}):\n" + "\n".join(
            f"{i}. {t}   (score {s:.2f})" for i, (t, s) in enumerate(nbest, 1)
        )
        t0 = time.time()
        raw = self.client.chat.completions.with_raw_response.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            store=False,  # 대시보드 로그에 남기지 않음
            response_format={"type": "json_schema", "json_schema": {"name": "lipread_refine", "strict": True, "schema": SCHEMA}},
        )
        resp = raw.parse()
        u = resp.usage
        self.last_meta = {
            "model_used": resp.model,
            "response_id": resp.id,
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
            "elapsed_sec": round(time.time() - t0, 2),
            "headers": pick_headers(raw.headers),
            "sent_text": user,
        }
        print(f"  [API 정보] 모델={resp.model} | 입력 {u.prompt_tokens} + 출력 {u.completion_tokens} = {u.total_tokens} 토큰 | 응답ID={resp.id}")
        _print_headers(self.last_meta["headers"])
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None):
            raise RuntimeError(msg.refusal)
        data = json.loads(msg.content)
        # 번역 개수가 후보 수와 다르면 맞춰 준다
        nk = list(data.get("nbest_korean", []))[: len(nbest)]
        data["nbest_korean"] = nk + ["-"] * (len(nbest) - len(nk))
        return data


def _print_headers(headers: dict):
    if headers:
        print("  [API 헤더] " + " | ".join(f"{k}={v}" for k, v in headers.items()))
