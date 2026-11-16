"""
사전학습 모델 자동 준비 스크립트 (처음 한 번)

  python setup_models.py

하는 일
  1) 사전학습 코드 저장소(Visual_Speech_Recognition_for_Multiple_Languages)가 없으면 받아오기
     - git이 있으면 git clone, 없으면 GitHub에서 zip으로 받기
  2) 공식 Model Zoo의 Google Drive 링크에서 모델 2개 내려받기 (약 1.1GB)
     - 영어 립리딩 모델 LRS3_V_WER19.1 (약 891MB)
     - 영어 언어모델 lm_en_subword (약 191MB)
  3) 압축을 풀어 정해진 위치에 넣고, 빠진 파일이 없는지 확인

모델 파일은 이 저장소에 포함하지 않는다 (용량 + 원저작자 라이선스: 비상업적 연구 목적).
항상 원저작자의 공식 링크에서 받는다.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from configparser import ConfigParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_NAME = "Visual_Speech_Recognition_for_Multiple_Languages"
REPO_URL = f"https://github.com/mpc001/{REPO_NAME}"
REPO_ZIP = f"{REPO_URL}/archive/refs/heads/master.zip"
MODEL_ZOO = f"{REPO_URL}#model-zoo"
CONFIG = "configs/LRS3_V_WER19.1.ini"

# 공식 Model Zoo(README)의 Google Drive 링크 (bit.ly/40EAtyX, bit.ly/3FE4XsV 가 가리키는 파일)
MODELS = [
    {"name": "영어 립리딩 모델 (LRS3_V_WER19.1, 약 891MB)",
     "drive_id": "1t8RHhzDTTvOQkLQhmK1LZGnXRRXOXGi6", "zip": "LRS3_V_WER19.1.zip",
     "target": "benchmarks/LRS3/models/LRS3_V_WER19.1", "link": "http://bit.ly/40EAtyX"},
    {"name": "영어 언어모델 (lm_en_subword, 약 191MB)",
     "drive_id": "1g31HGxJnnOwYl17b70ObFQZ1TSnPvRQv", "zip": "lm_en_subword.zip",
     "target": "benchmarks/LRS3/language_models/lm_en_subword", "link": "http://bit.ly/3FE4XsV"},
]
NEEDED = ("model.pth", "model.json")


def step(msg: str):
    print(f"\n=== {msg}")


# ---------------------------------------------------------------------------
# 1) 저장소
# ---------------------------------------------------------------------------
def ensure_repo(root: Path):
    if (root / "pipelines").exists():
        print(f"이미 있어요: {root.name}")
        return
    if shutil.which("git"):
        print("git clone 하는 중...")
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(root)])
        return
    print("git이 없어서 zip으로 받는 중...")
    with urllib.request.urlopen(REPO_ZIP, timeout=120) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z, tempfile.TemporaryDirectory() as tmp:
        z.extractall(tmp)
        inner = next(Path(tmp).iterdir())  # Visual_..._Languages-master
        shutil.move(str(inner), str(root))


# ---------------------------------------------------------------------------
# 2) 모델 내려받기 + 압축 풀기
# ---------------------------------------------------------------------------
def place_model(zip_path: Path, target: Path):
    """압축 안의 폴더 구조와 상관없이 model.pth / model.json을 찾아 target에 넣는다."""
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        for name in NEEDED:
            found = sorted(Path(tmp).rglob(name), key=lambda p: len(p.parts))
            if not found:
                raise FileNotFoundError(f"압축 파일 안에 {name}이(가) 없어요: {zip_path.name}")
            shutil.move(str(found[0]), str(target / name))


def ensure_models(root: Path, dl_dir: Path):
    try:
        import gdown
    except ImportError:
        sys.exit("gdown 패키지가 없어요. 먼저 `pip install -r requirements.txt` 를 실행하세요.")

    dl_dir.mkdir(exist_ok=True)
    failed = []
    for m in MODELS:
        target = root / m["target"]
        if all((target / n).exists() for n in NEEDED):
            print(f"이미 있어요: {m['name']}")
            continue
        zip_path = dl_dir / m["zip"]
        print(f"\n내려받는 중: {m['name']}")
        try:
            if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
                gdown.download(id=m["drive_id"], output=str(zip_path), quiet=False, resume=True)
            print("압축 푸는 중...")
            place_model(zip_path, target)
            zip_path.unlink(missing_ok=True)  # 다 넣었으면 zip은 지움
            print(f"완료 → {target.relative_to(HERE)}")
        except Exception as e:
            print(f"[실패] {type(e).__name__}: {e}")
            failed.append(m)
    return failed


# ---------------------------------------------------------------------------
# 3) 확인
# ---------------------------------------------------------------------------
def check(root: Path) -> list[str]:
    cfg = ConfigParser()
    cfg.read(root / CONFIG)
    missing = []
    for key in ("model_path", "model_conf", "rnnlm", "rnnlm_conf"):
        rel = cfg.get("model", key, fallback="")
        if rel and not (root / rel).exists():
            missing.append(rel)
    return missing


def main():
    root = HERE / REPO_NAME
    step("1/3 사전학습 코드 저장소")
    ensure_repo(root)

    step("2/3 모델 파일 (약 1.1GB, 인터넷 속도에 따라 수 분)")
    failed = ensure_models(root, HERE / "downloads")

    step("3/3 확인")
    missing = check(root)
    if not missing:
        print("모든 파일이 준비됐어요! 이제 실행하세요:\n  python web_app.py --cpu")
        return
    print("아직 없는 파일:")
    for m in missing:
        print("  -", root / m)
    if failed:
        print("\n자동 다운로드가 실패했어요. Google Drive가 '다운로드 한도 초과'를 낼 때가 있어요.")
        print("브라우저로 아래 링크에서 직접 받은 뒤, zip 파일을 downloads 폴더에 넣고 이 스크립트를 다시 실행하세요.")
        for m in failed:
            print(f"  - {m['name']}: {m['link']}  →  downloads/{m['zip']}")
        print(f"(공식 목록: {MODEL_ZOO})")
    sys.exit(1)


if __name__ == "__main__":
    main()
