"""
통역 연습 앱 - 백엔드 서버 (2단계: Whisper STT 추가)
=====================================================
1단계에서 만든 저장 기능에 Whisper STT를 추가.
오디오 파일을 받으면 백그라운드에서 Whisper가 더 정확한 텍스트로 변환해줘.

실행 방법:
  cd server
  pip install -r requirements.txt
  python main.py

주의: ffmpeg가 설치되어 있어야 해! (Whisper가 오디오 변환에 필요)
  winget install Gyan.FFmpeg  (한 번만 실행)
"""

import os
import re
import sys
import json
import shutil
import subprocess
import threading
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

# Windows 터미널에서 한글/이모지 출력 깨짐 방지
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ffmpeg 경로를 PATH에 추가 (winget 설치 후 재부팅 전에도 동작하게)
_FFMPEG_WINGET = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
for _p in _FFMPEG_WINGET.glob("Gyan.FFmpeg*/*/bin"):
    os.environ["PATH"] = str(_p) + os.pathsep + os.environ.get("PATH", "")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ============================================
# Whisper 모델 로딩
# ============================================
# Whisper 모델은 처음 사용할 때 인터넷에서 다운로드돼 (~150MB for base).
# 서버 시작할 때 백그라운드에서 미리 로드해두면, 첫 녹음 업로드 때 기다릴 필요가 없어.

try:
    import whisper as _whisper_lib
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[WARNING] openai-whisper not installed. Run: pip install openai-whisper")

whisper_model = None    # 로드된 모델 (None이면 아직 준비 안 됨)
whisper_ready = False   # 모델 준비 완료 여부
whisper_loading = False # 현재 로딩 중인지

def load_whisper_model():
    """
    Whisper 'base' 모델을 백그라운드 스레드에서 로드.
    'base' 모델은 속도/정확도 균형이 좋아서 시작하기 딱 좋아.
    더 정확하게 하고 싶으면 'small' 또는 'medium'으로 바꿔도 돼.
    """
    global whisper_model, whisper_ready, whisper_loading
    if not WHISPER_AVAILABLE:
        return

    whisper_loading = True
    print("[Whisper] 모델 로딩 중... (처음엔 다운로드 때문에 시간이 걸릴 수 있어)")
    try:
        whisper_model = _whisper_lib.load_model("base")
        whisper_ready = True
        print("[Whisper] 모델 준비 완료!")
    except Exception as e:
        print(f"[Whisper] 모델 로딩 실패: {e}")
    finally:
        whisper_loading = False

# 서버 시작 시 백그라운드에서 모델 로드 (서버 응답을 막지 않음)
if WHISPER_AVAILABLE:
    threading.Thread(target=load_whisper_model, daemon=True).start()


# ============================================
# 서버 설정
# ============================================

app = FastAPI(
    title="Interpretation Practice API",
    description="통역 연습 녹음 저장 서버 (Whisper STT 포함)",
    version="2.0.0",
)

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
# Whisper 전사 함수
# ============================================

def transcribe_with_whisper(recording_id: str, audio_path: Path):
    """
    오디오 파일을 Whisper로 전사하는 함수.
    FastAPI의 BackgroundTasks가 이 함수를 별도 스레드에서 실행해줘.
    녹음 업로드 → 즉시 응답 → 백그라운드에서 이 함수 실행 → metadata.json 업데이트
    """
    meta_path = RECORDINGS_DIR / f"rec_{recording_id}" / "metadata.json"

    if not whisper_ready or whisper_model is None:
        print(f"[Whisper] 모델 아직 준비 안 됨, rec_{recording_id} 전사 건너뜀")
        _update_whisper_status(meta_path, "skipped", error="모델 로딩 중이었음")
        return

    print(f"[Whisper] 전사 시작: rec_{recording_id}")

    try:
        # 브라우저 MediaRecorder webm은 Duration 메타데이터가 없어서
        # Whisper 내부 ffmpeg가 제대로 읽지 못하는 경우가 있음.
        # wav로 먼저 변환한 다음 Whisper에 넘기면 안정적으로 동작해.
        wav_path = audio_path.with_suffix(".wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path),
                 "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)],
                check=True, capture_output=True
            )
            transcribe_target = wav_path
            print(f"[Whisper] wav 변환 완료: {wav_path.name}")
        except Exception as conv_err:
            print(f"[Whisper] wav 변환 실패 ({conv_err}), webm 직접 사용")
            transcribe_target = audio_path

        print(f"[Whisper] 언어: 자동 감지")

        # Whisper로 전사!
        # fp16=False: CPU에서는 fp16 지원 안 해서 False로 설정해야 해
        result = whisper_model.transcribe(
            str(transcribe_target),
            fp16=False,
            condition_on_previous_text=False,
        )

        # 변환된 wav 임시 파일 정리
        if wav_path.exists():
            wav_path.unlink()

        transcript = result["text"].strip()
        detected_language = result.get("language", "unknown")

        # 세그먼트별 타임스탬프도 같이 저장
        whisper_segments = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            }
            for seg in result.get("segments", [])
        ]

        # metadata.json에 Whisper 결과 추가
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            meta["whisperTranscript"] = transcript
            meta["whisperSegments"] = whisper_segments
            meta["whisperLanguage"] = detected_language
            meta["whisperStatus"] = "done"
            meta["whisperCompletedAt"] = datetime.now().isoformat()

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[Whisper] 완료 rec_{recording_id}: {transcript[:60]}...")
        print(f"[Whisper] 감지된 언어: {detected_language}, 세그먼트 수: {len(whisper_segments)}")

    except Exception as e:
        print(f"[Whisper] 오류 rec_{recording_id}: {e}")
        _update_whisper_status(meta_path, "error", error=str(e))


def _update_whisper_status(meta_path: Path, status: str, error: str = None):
    """metadata.json의 Whisper 상태만 업데이트하는 헬퍼"""
    if not meta_path.exists():
        return
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["whisperStatus"] = status
        if error:
            meta["whisperError"] = error
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ============================================
# API 엔드포인트들
# ============================================

@app.get("/")
async def root():
    """서버 상태 + Whisper 준비 여부 확인"""
    return {
        "status": "ok",
        "message": "통역 연습 서버가 실행 중입니다!",
        "whisper": {
            "available": WHISPER_AVAILABLE,
            "ready": whisper_ready,
            "loading": whisper_loading,
        }
    }


@app.get("/api/whisper-status")
async def get_whisper_status():
    """Whisper 모델 준비 상태 확인 (프론트엔드에서 폴링용)"""
    return {
        "available": WHISPER_AVAILABLE,
        "ready": whisper_ready,
        "loading": whisper_loading,
    }


@app.post("/api/recordings")
async def upload_recording(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="녹음 오디오 파일 (.webm)"),
    metadata: str = Form(..., description="녹음 메타데이터 (JSON 문자열)"),
):
    """
    녹음 업로드 API (2단계: Whisper 전사 자동 시작).

    동작 순서:
    1. 오디오 + 메타데이터 저장 → 즉시 응답 반환
    2. 백그라운드에서 Whisper 전사 시작 (프론트는 기다릴 필요 없음!)
    3. 전사 완료 → metadata.json에 whisperTranscript 추가
    4. 프론트엔드가 폴링해서 결과 확인
    """
    # 1) 메타데이터 파싱
    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="메타데이터 JSON 형식이 잘못됐어요")

    recording_id = str(meta.get("id", int(datetime.now().timestamp() * 1000)))

    # 2) 녹음별 폴더 생성
    rec_dir = RECORDINGS_DIR / f"rec_{recording_id}"
    rec_dir.mkdir(exist_ok=True)

    # 3) 오디오 파일 저장
    audio_path = rec_dir / "audio.webm"
    with open(audio_path, "wb") as f:
        content = await audio.read()
        f.write(content)

    # 4) 메타데이터 저장 (whisperStatus를 "pending"으로 표시)
    meta["savedAt"] = datetime.now().isoformat()
    meta["audioFile"] = "audio.webm"
    meta["audioSize"] = os.path.getsize(audio_path)
    meta["whisperStatus"] = "pending" if (WHISPER_AVAILABLE and whisper_ready) else "unavailable"

    meta_path = rec_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Server] 녹음 저장: rec_{recording_id} ({meta['audioSize']} bytes)")

    # 5) Whisper 전사를 백그라운드 태스크로 등록
    #    → 이 함수는 바로 응답을 반환하고, Whisper는 뒤에서 조용히 실행돼
    if WHISPER_AVAILABLE and whisper_ready:
        background_tasks.add_task(transcribe_with_whisper, recording_id, audio_path)
        print(f"[Whisper] 전사 예약됨: rec_{recording_id}")

    return {
        "success": True,
        "recordingId": recording_id,
        "audioSize": meta["audioSize"],
        "whisperStatus": meta["whisperStatus"],
        "message": "녹음 저장됨! Whisper 전사 시작 중..." if whisper_ready else "녹음 저장됨 (Whisper 준비 중)",
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
    """특정 녹음의 메타데이터 반환 (whisperStatus 포함)"""
    meta_path = RECORDINGS_DIR / f"rec_{recording_id}" / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="녹음을 찾을 수 없어요")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return meta


@app.get("/api/recordings/{recording_id}/audio")
async def get_recording_audio(recording_id: str):
    """특정 녹음의 오디오 파일 다운로드"""
    audio_path = RECORDINGS_DIR / f"rec_{recording_id}" / "audio.webm"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="오디오 파일을 찾을 수 없어요")

    return FileResponse(
        path=str(audio_path),
        media_type="audio/webm",
        filename=f"recording_{recording_id}.webm",
    )


@app.post("/api/recordings/{recording_id}/transcribe")
async def retranscribe(recording_id: str, background_tasks: BackgroundTasks):
    """
    이미 저장된 녹음에 Whisper를 소급 적용하는 API.
    구 버전 서버에서 저장한 녹음이나, 처음 업로드 당시 Whisper가 아직 준비 안 됐던 경우에 사용.
    """
    audio_path = RECORDINGS_DIR / f"rec_{recording_id}" / "audio.webm"
    meta_path  = RECORDINGS_DIR / f"rec_{recording_id}" / "metadata.json"

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="오디오 파일을 찾을 수 없어요")

    if not whisper_ready:
        raise HTTPException(status_code=503, detail="Whisper 모델이 아직 준비 중이에요. 잠시 후 다시 시도해주세요.")

    # 상태를 pending으로 표시 후 백그라운드 실행
    _update_whisper_status(meta_path, "pending")
    background_tasks.add_task(transcribe_with_whisper, recording_id, audio_path)

    return {"success": True, "message": f"rec_{recording_id} Whisper 전사 시작!", "whisperStatus": "pending"}


@app.post("/api/transcribe-all")
async def transcribe_all(background_tasks: BackgroundTasks):
    """
    Whisper Status가 없거나 실패한 모든 녹음을 일괄 재전사.
    서버를 새로 바꿨을 때 기존 녹음들을 한 번에 처리할 때 유용해.
    """
    if not whisper_ready:
        raise HTTPException(status_code=503, detail="Whisper 모델이 아직 준비 중이에요.")

    scheduled = []
    for rec_dir in sorted(RECORDINGS_DIR.iterdir()):
        if not rec_dir.is_dir() or not rec_dir.name.startswith("rec_"):
            continue
        meta_path  = rec_dir / "metadata.json"
        audio_path = rec_dir / "audio.webm"
        if not meta_path.exists() or not audio_path.exists():
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        status = meta.get("whisperStatus", "none")
        if status not in ("done",):  # done이 아닌 건 전부 재시도
            recording_id = rec_dir.name.replace("rec_", "")
            _update_whisper_status(meta_path, "pending")
            background_tasks.add_task(transcribe_with_whisper, recording_id, audio_path)
            scheduled.append(recording_id)
            print(f"[Whisper] 예약: rec_{recording_id}")

    return {
        "success": True,
        "scheduled": len(scheduled),
        "recordingIds": scheduled,
        "message": f"{len(scheduled)}개 녹음 Whisper 전사 시작!"
    }


@app.delete("/api/recordings/{recording_id}")
async def delete_recording(recording_id: str):
    """특정 녹음 삭제"""
    rec_dir = RECORDINGS_DIR / f"rec_{recording_id}"
    if not rec_dir.exists():
        raise HTTPException(status_code=404, detail="녹음을 찾을 수 없어요")

    shutil.rmtree(rec_dir)
    print(f"[Server] 삭제: rec_{recording_id}")
    return {"success": True, "message": "녹음이 삭제되었습니다!"}


# ============================================
# Stage 3: 비교 리포트 분석 함수들
# ============================================

def count_filler_words(text: str, language: str) -> dict:
    """
    필러워드(말 버릇) 카운트.
    영어: um, uh, like, you know, i mean 등
    한국어: 어, 음, 그, 뭐, 이제 등
    """
    text_lower = text.lower()

    if "ko" in language or "ja" in language or "zh" in language:
        fillers = ["어", "음", "그", "뭐", "이제", "사실", "일단", "그래서", "근데", "아"]
        counts = {}
        for filler in fillers:
            count = text_lower.count(filler)
            if count > 0:
                counts[filler] = count
    else:
        fillers = [
            "um", "uh", "like", "you know", "i mean",
            "sort of", "kind of", "basically", "actually",
            "literally", "right", "okay so",
        ]
        counts = {}
        for filler in fillers:
            count = len(re.findall(r'\b' + re.escape(filler) + r'\b', text_lower))
            if count > 0:
                counts[filler] = count

    return counts


def generate_suggestions(stats: dict) -> list:
    """통계를 바탕으로 개선 제안 생성"""
    suggestions = []
    total_fillers = stats.get("totalFillerWords", 0)
    wpm = stats.get("wordsPerMinute", 0)
    word_count = stats.get("whisperWordCount", 0)
    duration = stats.get("durationSeconds", 0)
    similarity = stats.get("similarity", 0)

    if total_fillers > 5:
        suggestions.append(
            f"필러워드(um, uh 등)가 {total_fillers}회 발견됐어. "
            "말 버릇을 의식적으로 줄이는 연습을 해보자!"
        )
    elif total_fillers > 2:
        suggestions.append(
            f"필러워드가 {total_fillers}개 있어. 조금만 더 의식하면 금방 줄일 수 있을 거야."
        )

    if wpm > 180:
        suggestions.append(
            f"말 속도가 분당 {wpm}단어로 빠른 편이야. "
            "청중이 따라올 수 있도록 조금 천천히 말해보자."
        )
    elif 0 < wpm < 90:
        suggestions.append(
            f"말 속도가 분당 {wpm}단어로 느린 편이야. "
            "좀 더 자신감 있게, 리듬감 있게 말해보자."
        )

    if word_count < 15 and duration > 10:
        suggestions.append(
            "녹음 길이에 비해 발화량이 적어. "
            "침묵보다는 더 많이 채워서 말하려고 노력해보자!"
        )

    if 0 < similarity < 60:
        suggestions.append(
            f"음성 인식 정확도({similarity}%)가 낮아. "
            "더 또렷하게 발음하거나 마이크 위치를 조정해봐."
        )

    if not suggestions:
        suggestions.append(
            "훌륭한 통역이야! 꾸준히 연습하면 더욱 자연스러워질 거야."
        )

    return suggestions


@app.get("/api/recordings/{recording_id}/report")
async def get_recording_report(recording_id: str):
    """
    녹음 비교 리포트 생성.
    Web Speech(실시간 인식) vs Whisper(정확한 AI 전사)를 비교하고
    필러워드 수, 말 속도(WPM), 개선 제안을 반환해.
    """
    meta_path = RECORDINGS_DIR / f"rec_{recording_id}" / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="녹음을 찾을 수 없어요")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    whisper_status = meta.get("whisperStatus", "none")
    if whisper_status != "done":
        return {
            "recordingId": recording_id,
            "error": "Whisper 전사가 아직 완료되지 않았어요. 잠시 후 다시 시도해봐!",
            "whisperStatus": whisper_status,
        }

    web_speech_text = meta.get("transcript", "")
    whisper_text    = meta.get("whisperTranscript", "")
    language        = meta.get("whisperLanguage", "en")
    duration        = float(meta.get("duration", 0))

    # 단어 수
    whisper_words    = len(whisper_text.split()) if whisper_text else 0
    web_speech_words = len(web_speech_text.split()) if web_speech_text else 0

    # 분당 단어 수 (WPM)
    wpm = round(whisper_words / (duration / 60)) if duration > 0 else 0

    # 필러워드
    fillers       = count_filler_words(whisper_text, language)
    total_fillers = sum(fillers.values())

    # Web Speech ↔ Whisper 유사도
    similarity = 0.0
    if web_speech_text and whisper_text:
        matcher    = SequenceMatcher(None, web_speech_text.lower(), whisper_text.lower())
        similarity = round(matcher.ratio() * 100, 1)

    stats = {
        "whisperWordCount":  whisper_words,
        "webSpeechWordCount": web_speech_words,
        "durationSeconds":   round(duration, 1),
        "wordsPerMinute":    wpm,
        "fillerWords":       fillers,
        "totalFillerWords":  total_fillers,
        "similarity":        similarity,
    }

    suggestions = generate_suggestions(stats)

    return {
        "recordingId":       recording_id,
        "generatedAt":       datetime.now().isoformat(),
        "webSpeechTranscript": web_speech_text,
        "whisperTranscript": whisper_text,
        "language":          language,
        "targetLanguage":    meta.get("targetLanguage", ""),
        "sourceLanguage":    meta.get("sourceLanguage", ""),
        "stats":             stats,
        "suggestions":       suggestions,
    }


# ============================================
# 서버 실행
# ============================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("  [SERVER] Interpretation Practice Server v2")
    print("=" * 50)
    print(f"  Recording folder : {RECORDINGS_DIR}")
    print(f"  Server URL       : http://localhost:8000")
    print(f"  API docs         : http://localhost:8000/docs")
    print(f"  Open app         : http://localhost:8000/static/index.html")
    print(f"  Whisper          : {'available' if WHISPER_AVAILABLE else 'NOT installed'}")
    print("=" * 50)
    print("  Press Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
