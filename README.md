# 👄 AI 립리딩 — 입모양으로 영어 받아쓰기 + 한국어 번역

웹캠으로 **소리 없이 입모양만** 녹화하면, 사전학습 립리딩 모델이 영어 문장을 읽고 GPT가 입모양 혼동을 보정한 뒤 **한국어로 번역**해 주는 웹 서비스입니다.

<!-- 실행 화면을 캡처해서 docs/demo.png 로 저장한 뒤, 아래 줄 앞뒤의 주석 표시를 지우면 화면이 표시됩니다
![실행 화면](docs/demo.png)
-->

## 주요 기능
- 🎥 브라우저 웹캠으로 녹화 (소리·마이크 사용 안 함)
- 🟢 AI가 인식한 입술을 초록색으로 표시한 영상
- 🗣️ 영어 후보 문장 5개 + 모델 확신도
- 🇰🇷 GPT 보정 문장과 한국어 번역 (후보별 번역 포함)
- 🔑 접속한 사람이 **자기 OpenAI API 키**를 입력해서 사용 (키는 저장하지 않음)

## 동작 원리
```
웹캠 녹화 (브라우저)
 → MediaPipe FaceLandmarker: 양 눈·코끝·입 중심 4점 + 입술 20점
 → 25fps로 맞추기 → 얼굴 정렬 → 입 주변 96×96 흑백 → 88×88
 → 사전학습 립리딩 모델 (3D CNN+ResNet → Conformer 인코더 → Transformer 디코더 + 언어모델, 빔서치)
 → 영어 후보 문장 5개
 → GPT-4o-mini: 가장 그럴듯한 문장 선택·보정 + 한국어 번역 (텍스트만 전송)
```
| 모델 | 역할 | 출처 |
|---|---|---|
| MediaPipe FaceLandmarker | 얼굴·입술 위치 찾기 | [Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (실행 시 자동 다운로드) |
| Auto-AVSR 영상 전용 모델 `LRS3_V_WER19.1` | 입모양 → 영어 문장 (LRS3 기준 단어 오류율 19.1%) | [Visual_Speech_Recognition_for_Multiple_Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages) |
| 영어 언어모델 `lm_en_subword` | 자연스러운 문장 쪽으로 후보 정렬 | 위와 같음 |
| GPT-4o-mini | 후보 보정 + 한국어 번역 | OpenAI API (사용자 본인 키) |

립리딩 모델은 TED 강연 영상 데이터셋 [LRS3](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html)로 학습된 공개 모델이며, 이 프로젝트는 데이터셋을 직접 쓰지 않고 학습이 끝난 모델로 추론만 합니다.

## 빠른 시작 (Windows)
**필요한 것:** 웹캠, 인터넷, (선택) OpenAI API 키

### 가장 쉬운 방법: 더블클릭 2번
1. 이 저장소를 받습니다: 초록색 **Code → Download ZIP** → 압축 풀기 (또는 `git clone`)
2. **`setup.bat` 더블클릭** (처음 한 번, 10~20분)
   - Python 3.10이 없으면 자동으로 설치한 뒤 "다시 실행하라"고 안내해요 → 한 번 더 더블클릭
   - 가상환경 만들기 → 패키지 설치 → 모델 파일(약 1.1GB) 다운로드까지 자동
3. **`start_web.bat` 더블클릭** → 브라우저에 `http://127.0.0.1:7860` 이 열림
   - 다른 사람과 같이 쓰려면 **`start_share.bat`** (공개 링크 생성)

### 터미널로 직접 하는 방법
Python **3.10**이 필요해요 (최신 Python에서는 PyTorch 2.4.1을 설치할 수 없음).

```powershell
git clone https://github.com/thehyun/ai-lipreading.git
cd ai-lipreading

py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python setup_models.py      # 사전학습 코드 + 모델 파일(약 1.1GB) 자동 준비, 처음 한 번
python web_app.py --cpu     # http://127.0.0.1:7860 이 자동으로 열림
```
- Python 3.10이 없으면 `winget install -e --id Python.Python.3.10` 후 터미널을 다시 여세요.
- `activate`에서 "스크립트를 실행할 수 없으므로" 오류가 나면 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`를 한 번 실행하세요.
- NVIDIA GPU가 있으면 `pip install -r` 전에 `pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121` 을 먼저 실행하고, 실행할 때 `--cpu`를 빼세요.

### 모델을 직접 받는 경우
`setup_models.py`가 Google Drive 한도 초과 등으로 실패하면 공식 [Model Zoo](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages#model-zoo)에서 직접 받으세요.

| 파일 | 링크 | 크기 |
|---|---|---|
| `LRS3_V_WER19.1.zip` (Visual-only 19.1) | http://bit.ly/40EAtyX | 약 891MB |
| `lm_en_subword.zip` (Language model) | http://bit.ly/3FE4XsV | 약 191MB |

받은 zip 두 개를 `downloads/` 폴더에 넣고 `python setup_models.py`를 다시 실행하면 알맞은 위치에 풀어 줍니다.

## 다른 사람과 같이 쓰기
```powershell
python web_app.py --cpu --share
```
터미널에 나오는 `https://xxxx.gradio.live` 주소를 공유하면, 상대방은 **설치 없이** 브라우저로 접속해서 자기 API 키를 넣고 사용할 수 있습니다.
- 링크는 최대 72시간 유지되고, 실행 중인 PC가 켜져 있어야 합니다.
- 입모양 인식은 한 번에 한 명씩 처리합니다 (CPU 기준 한 문장 수 초~수십 초).
- 링크를 아는 사람은 누구나 접속할 수 있으니 필요한 사람에게만 공유하세요.

## 사용법
1. 접속하면 **API 키 입력** 화면이 뜹니다 → 키 입력 후 *키 확인하고 시작* (또는 *API 없이 시작*: 영어 받아쓰기만)
2. 웹캠 ● 녹화 → **1~4초 길이의 짧은 영어 문장** (소리 없이도 OK) → ■ 정지
3. **👄 입모양 읽기** → 초록색 입술 표시 영상, 보정 문장·한국어 번역, 후보 5개 표

**잘 되는 조건:** 얼굴 정면, 밝은 조명, 입술이 크게 보이게, 또박또박 (예: *Thank you very much*, *How are you today*)

## 프로젝트 구조
| 파일 | 역할 |
|---|---|
| `web_app.py` | 웹사이트 (Gradio): API 키 입력 → 녹화 → 결과 표시 |
| `vsr_live.py` | 모델 불러오기·받아쓰기 핵심 코드 + 데스크톱 창 버전 |
| `face_points.py` | 얼굴 기준점 추출, 25fps 맞추기 |
| `gpt_refine.py` | OpenAI API: 키 확인, 보정·번역 |
| `ui_text.py` | 데스크톱 창 한글 표시 |
| `setup_models.py` | 사전학습 코드·모델 자동 준비 |
| `setup.bat` | Windows 첫 설정 자동화 (Python 3.10 확인·설치, 가상환경, 패키지, 모델) |
| `start_web.bat` / `start_share.bat` | 더블클릭 실행 (내 PC만 / 공개 링크) |

데스크톱 창 버전: `python vsr_live.py --vsr-root Visual_Speech_Recognition_for_Multiple_Languages --lang en --cpu` (창을 클릭하고 SPACE → 말하기 → SPACE)

## 데이터와 개인정보
| 데이터 | 처리 위치 | 보관 |
|---|---|---|
| 웹캠 영상 | 앱을 실행한 PC (서버) | 임시 파일, 몇 분 안에 자동 삭제 |
| API 키 | 서버 메모리 (브라우저 탭별) | 저장·기록 안 함 |
| OpenAI로 보내는 것 | 후보 문장 **텍스트만** (`store=False`) | 영상·얼굴 이미지는 보내지 않음 |

| OpenAI 호출 | 보내는 것 | 받는 것 |
|---|---|---|
| `GET /v1/models` | API 키 | 사용 가능한 모델 목록 |
| `POST /v1/chat/completions` | 후보 문장 5개와 점수 | 보정 문장, 한국어 번역, 후보별 번역, 신뢰도, 이유, 토큰 사용량 |

## 한계와 향후 계획
- 입력은 **영어만** 지원 (한국어 공개 사전학습 모델이 없음). 출력은 한국어 번역.
- 일반 웹캠, 소리 없이 작게 말하기에서는 연구 수치(19.1%)보다 오류가 많음.
- 문장 단위 처리라 실시간 자막은 불가.
- 향후: 한국어 데이터셋(OLKAVS 등)으로 미세조정, 음성+입모양 융합으로 소음 환경 대응, 화상 회의 자막 오버레이.

## 문제 해결
| 증상 | 해결 |
|---|---|
| `No matching distribution found for torch==2.4.1` | Python 3.10 가상환경에서 설치하세요 |
| `UnicodeDecodeError: 'cp949'` | 최신 코드는 자동 해결 (UTF-8 모드로 재실행) |
| 웹캠 결과가 모두 "오류" | 최신 `web_app.py` 사용 (FFmpeg 없이 동작) |
| 폴더에 `web_app.pyi` 가 생김 | 실행과 무관한 파일, 지워도 됨 |
| 카메라가 안 켜짐 | 브라우저 주소창 자물쇠 → 카메라 허용 |

## 라이선스와 출처
- 사전학습 코드와 모델: [Visual_Speech_Recognition_for_Multiple_Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages) (Pingchuan Ma 외, Imperial College London) — **비상업적 연구·비교 목적**으로만 사용 가능. 이 저장소에는 포함하지 않고 공식 링크에서 받습니다.
- 논문: Auto-AVSR (Ma et al., ICASSP 2023), Visual Speech Recognition for Multiple Languages in the Wild (Ma et al., Nature Machine Intelligence 2022)
- 웹 화면: [Gradio](https://www.gradio.app/) · 얼굴 인식: [MediaPipe](https://ai.google.dev/edge/mediapipe)
