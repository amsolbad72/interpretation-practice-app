"""
통역 연습 앱 - 백엔드 서버
============================
녹음 저장 + AI 피드백 (Gemini / Claude)

실행 방법:
  cd server
  pip install -r requirements.txt
  python main.py
"""

import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path

# Windows 터미널에서 한글/이모지 출력 깨짐 방지
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# .env 파일에서 API 키 자동 로드
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    import anthropic as _anthropic_lib
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from google import genai as _genai_lib
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[WARNING] google-genai not installed. Run: pip install google-genai")


# ============================================
# 서버 설정
# ============================================

app = FastAPI(title="Interpretation Practice API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RECORDINGS_DIR = Path(__file__).parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

FRONTEND_DIR = Path(__file__).parent.parent
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")


# ============================================
# API 엔드포인트
# ============================================

@app.get("/")
async def root():
    return {"status": "ok", "message": "통역 연습 서버가 실행 중입니다!"}


@app.post("/api/recordings")
async def upload_recording(
    audio: UploadFile = File(...),
    metadata: str = Form(...),
):
    """녹음 업로드 저장"""
    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="메타데이터 JSON 형식이 잘못됐어요")

    recording_id = str(meta.get("id", int(datetime.now().timestamp() * 1000)))
    rec_dir = RECORDINGS_DIR / f"rec_{recording_id}"
    rec_dir.mkdir(exist_ok=True)

    audio_path = rec_dir / "audio.webm"
    with open(audio_path, "wb") as f:
        content = await audio.read()
        f.write(content)

    meta["savedAt"] = datetime.now().isoformat()
    meta["audioFile"] = "audio.webm"
    meta["audioSize"] = os.path.getsize(audio_path)

    meta_path = rec_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Server] 녹음 저장: rec_{recording_id} ({meta['audioSize']} bytes)")

    return {
        "success": True,
        "recordingId": recording_id,
        "audioSize": meta["audioSize"],
        "message": "녹음 저장 완료!",
    }


@app.get("/api/recordings")
async def list_recordings():
    """저장된 모든 녹음 목록 반환"""
    recordings = []
    for rec_dir in sorted(RECORDINGS_DIR.iterdir()):
        if not rec_dir.is_dir() or not rec_dir.name.startswith("rec_"):
            continue
        meta_path = rec_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        recordings.append(meta)
    return {"recordings": recordings, "count": len(recordings)}


@app.get("/api/recordings/{recording_id}")
async def get_recording(recording_id: str):
    """특정 녹음 메타데이터 반환"""
    meta_path = RECORDINGS_DIR / f"rec_{recording_id}" / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="녹음을 찾을 수 없어요")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/recordings/{recording_id}/audio")
async def get_recording_audio(recording_id: str):
    """녹음 오디오 파일 반환"""
    audio_path = RECORDINGS_DIR / f"rec_{recording_id}" / "audio.webm"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="오디오 파일을 찾을 수 없어요")
    return FileResponse(path=str(audio_path), media_type="audio/webm",
                        filename=f"recording_{recording_id}.webm")


@app.delete("/api/recordings/{recording_id}")
async def delete_recording(recording_id: str):
    """녹음 삭제"""
    rec_dir = RECORDINGS_DIR / f"rec_{recording_id}"
    if not rec_dir.exists():
        raise HTTPException(status_code=404, detail="녹음을 찾을 수 없어요")
    shutil.rmtree(rec_dir)
    print(f"[Server] 삭제: rec_{recording_id}")
    return {"success": True, "message": "녹음이 삭제되었습니다!"}


@app.get("/api/recordings/{recording_id}/report")
async def get_recording_report(recording_id: str):
    """리포트 반환 (Web Speech 전사 텍스트)"""
    meta_path = RECORDINGS_DIR / f"rec_{recording_id}" / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="녹음을 찾을 수 없어요")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "recordingId":         recording_id,
        "generatedAt":         datetime.now().isoformat(),
        "webSpeechTranscript": meta.get("transcript", ""),
        "sourceLanguage":      meta.get("sourceLanguage", ""),
        "targetLanguage":      meta.get("targetLanguage", ""),
    }


# ============================================
# AI 피드백
# ============================================

@app.post("/api/recordings/{recording_id}/ai-feedback")
async def get_ai_feedback(recording_id: str):
    """Gemini / Claude AI 통역 코칭 피드백"""
    gemini_key    = os.environ.get("GEMINI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not gemini_key and not anthropic_key:
        raise HTTPException(status_code=503,
            detail="GEMINI_API_KEY 또는 ANTHROPIC_API_KEY 환경변수를 설정해야 해!")

    use_gemini = bool(gemini_key and GEMINI_AVAILABLE)

    meta_path = RECORDINGS_DIR / f"rec_{recording_id}" / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="녹음을 찾을 수 없어요")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 캐시된 피드백이 있으면 바로 반환
    if meta.get("aiFeedbackStatus") == "done" and meta.get("aiFeedback"):
        return {"recordingId": recording_id, "feedback": meta["aiFeedback"], "cached": True}

    transcript = meta.get("transcript", "")
    if not transcript:
        raise HTTPException(status_code=400,
            detail="저장된 전사 텍스트가 없어. 녹음 후 말을 해야 피드백을 받을 수 있어!")

    practice_mode   = meta.get("practiceMode", "simultaneous")
    source_language = meta.get("sourceLanguage", "")
    target_language = meta.get("targetLanguage", "")
    duration        = float(meta.get("duration", 0))

    if practice_mode == "shadowing":
        mode_desc = "섀도잉 (원어민 발화를 그대로 따라하는 연습)"
        lang_desc = f"언어: {target_language}" if target_language else ""
    else:
        mode_desc = "동시통역"
        lang_desc = f"{source_language} → {target_language}" if source_language and target_language else ""

    prompt = f"""당신은 전문 통역 및 외국어 발화 코치입니다.
아래는 통역 연습생이 '{mode_desc}' 연습을 한 결과입니다.
{f'({lang_desc})' if lang_desc else ''}

[발화 시간]
약 {round(duration)}초

[연습생의 발화 (음성인식 결과)]
{transcript}

위 발화를 분석하여 다음 형식으로 한국어 코칭 피드백을 제공해 주세요.
(발화 내용만 보고 평가하세요. 음질이나 배경 소음은 고려하지 않아도 됩니다.)

## 🎯 전반적 평가
(2-3문장. 전체적인 발화 품질과 수준을 평가해 주세요.)

## ✅ 잘한 점
- (구체적으로 칭찬할 점 2-3가지)

## 🔧 개선이 필요한 부분
- (구체적인 피드백 2-3가지)

## 💪 이렇게 연습해 보세요
- (약점을 보완할 수 있는 실용적인 연습 방법 2가지)

핵심만 간결하게, 한국어로 작성해 주세요."""

    ai_provider = "Gemini" if use_gemini else "Claude"
    print(f"[{ai_provider}] 피드백 생성 시작: rec_{recording_id}")

    try:
        if use_gemini:
            client = _genai_lib.Client(api_key=gemini_key)
            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            feedback_text = response.text
        else:
            client = _anthropic_lib.Anthropic(api_key=anthropic_key)
            message = client.messages.create(
                model="claude-opus-4-6", max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            feedback_text = message.content[0].text

        meta["aiFeedback"] = feedback_text
        meta["aiFeedbackStatus"] = "done"
        meta["aiFeedbackProvider"] = ai_provider
        meta["aiFeedbackGeneratedAt"] = datetime.now().isoformat()

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[{ai_provider}] 피드백 완료: rec_{recording_id}")
        return {"recordingId": recording_id, "feedback": feedback_text, "cached": False, "provider": ai_provider}

    except Exception as e:
        print(f"[{ai_provider}] 오류: {e}")
        raise HTTPException(status_code=500, detail=f"{ai_provider} API 오류: {str(e)}")


# ============================================
# 서버 실행
# ============================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  통역 연습 서버")
    print("=" * 50)
    print(f"  앱 주소: http://localhost:8000/static/index.html")
    print(f"  Gemini : {'✅' if os.environ.get('GEMINI_API_KEY') else '❌ GEMINI_API_KEY 없음'}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
