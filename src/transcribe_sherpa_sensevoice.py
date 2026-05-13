# src/transcribe_sherpa_sensevoice.py

import sys
import time
import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_SUFFIX, OUTPUT_DIR, SHERPA_SENSEVOICE_MODEL_DIR

try:
    import sherpa_onnx
    import numpy as np
    import soundfile as sf
except ImportError:
    raise ImportError("請先安裝相依套件：pip install sherpa-onnx numpy soundfile")

SAMPLE_RATE   = 16000
CHUNK_SECONDS = 300  # 每段 5 分鐘，可視 RAM 調小


# ==========================================
# 輔助函數
# ==========================================
def format_time(seconds: float) -> str:
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = round((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def mp3_to_wav(mp3_path: Path) -> Path:
    """
    用 ffmpeg 將 mp3 轉成 16kHz mono wav，存在同目錄下。
    soundfile 無法直接讀 mp3，需要先轉檔。
    """
    wav_path = mp3_path.with_suffix(".wav")
    if wav_path.exists():
        print(f"[Sherpa-SenseVoice] 已有 WAV 暫存檔，略過轉檔：{wav_path.name}")
        return wav_path

    print(f"[Sherpa-SenseVoice] 轉換 MP3 → WAV（ffmpeg）...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mp3_path),
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-f", "wav",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 轉檔失敗：\n{result.stderr.decode()}"
        )
    print(f"[Sherpa-SenseVoice] WAV 暫存檔：{wav_path}")
    return wav_path


def process_chunk(
    recognizer,
    audio_chunk: np.ndarray,
) -> tuple[list[str], list[float]]:
    """對單一音訊片段辨識，回傳 (tokens, timestamps)。"""
    # sherpa-onnx 要求 float32、C-contiguous
    chunk = np.ascontiguousarray(audio_chunk, dtype=np.float32)
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, chunk)
    recognizer.decode_stream(stream)
    return stream.result.tokens, stream.result.timestamps


def save_to_srt(
    all_tokens: list[str],
    all_timestamps: list[float],
    out_path: Path,
    chars_per_line: int = 15,
) -> bool:
    if not all_tokens or not all_timestamps:
        print("[Sherpa-SenseVoice] ⚠️ 無時間戳記，無法生成 SRT。")
        return False

    srt_blocks   = []
    chunk_tokens = []
    start_time   = None
    chunk_idx    = 1

    for i, (token, t) in enumerate(zip(all_tokens, all_timestamps)):
        if "<|" in token or "|>" in token:
            continue
        if start_time is None:
            start_time = t
        chunk_tokens.append(token)

        if len(chunk_tokens) >= chars_per_line or i == len(all_tokens) - 1:
            end_time = t + 0.5
            text_str = "".join(chunk_tokens).replace(" ", "").replace("\u3000", "")
            if text_str:
                srt_blocks.append(
                    f"{chunk_idx}\n"
                    f"{format_time(start_time)} --> {format_time(end_time)}\n"
                    f"{text_str}\n"
                )
                chunk_idx += 1
            chunk_tokens = []
            start_time   = None

    if not srt_blocks:
        print("[Sherpa-SenseVoice] ⚠️ 過濾後無任何有效文字。")
        return False

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_blocks))
    return True


# ==========================================
# 主函數
# ==========================================
def run(audio_path: Path) -> Path | None:
    stem     = audio_path.stem
    out_dir  = OUTPUT_DIR / stem / "transcriptions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (stem + MODEL_SUFFIX["sensevoice"])

    model_dir   = SHERPA_SENSEVOICE_MODEL_DIR
    model_file  = model_dir / "model.int8.onnx"
    tokens_file = model_dir / "tokens.txt"

    if not model_file.exists() or not tokens_file.exists():
        print(f"[Sherpa-SenseVoice] ❌ 找不到模型檔案：{model_dir}")
        return None

    # MP3 → WAV（soundfile 不支援 mp3）
    if audio_path.suffix.lower() == ".mp3":
        wav_path = mp3_to_wav(audio_path)
    else:
        wav_path = audio_path

    print("[Sherpa-SenseVoice] 載入模型...")
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model_file),
        tokens=str(tokens_file),
        num_threads=4,
        use_itn=True,
        language="",
        debug=False,
    )

    # ------------------------------------------
    # 用 soundfile 分段讀取，穩定且格式保證正確
    # ------------------------------------------
    chunk_samples = CHUNK_SECONDS * SAMPLE_RATE
    all_tokens:     list[str]   = []
    all_timestamps: list[float] = []
    time_offset = 0.0
    chunk_idx   = 0

    print(f"[Sherpa-SenseVoice] 開始分段辨識（每段 {CHUNK_SECONDS}s）：{wav_path.name}")
    t0 = time.time()

    with sf.SoundFile(str(wav_path)) as f:
        # 若原始取樣率不是 16kHz，需要重新取樣
        orig_sr = f.samplerate
        if orig_sr != SAMPLE_RATE:
            print(f"[Sherpa-SenseVoice] 原始取樣率 {orig_sr}Hz，自動重新取樣至 {SAMPLE_RATE}Hz...")
            import librosa  # 只在需要時才 import
            resample_needed = True
        else:
            resample_needed = False

        while True:
            audio_chunk = f.read(chunk_samples, dtype="float32", always_2d=False)
            if len(audio_chunk) == 0:
                break

            if resample_needed:
                audio_chunk = librosa.resample(audio_chunk, orig_sr=orig_sr, target_sr=SAMPLE_RATE)

            chunk_duration = len(audio_chunk) / SAMPLE_RATE
            chunk_idx += 1
            print(f"  段 {chunk_idx}：{time_offset:.0f}s ~ {time_offset + chunk_duration:.0f}s")

            tokens, timestamps = process_chunk(recognizer, audio_chunk)
            all_tokens.extend(tokens)
            all_timestamps.extend([t + time_offset for t in timestamps])

            time_offset += chunk_duration
            del audio_chunk

    print(f"\n[Sherpa-SenseVoice] 辨識完成，寫入 SRT...")
    success = save_to_srt(all_tokens, all_timestamps, out_path, chars_per_line=15)
    if not success:
        return None

    # 清理暫存 WAV（若是從 MP3 轉換來的）
    if audio_path.suffix.lower() == ".mp3" and wav_path.exists():
        wav_path.unlink()
        print(f"[Sherpa-SenseVoice] 已刪除暫存 WAV：{wav_path.name}")

    print(f"[Sherpa-SenseVoice] ✅ 完成，耗時 {time.time()-t0:.1f}s，輸出：{out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python src/transcribe_sherpa_sensevoice.py <audio_path>")
        sys.exit(1)
    run(Path(sys.argv[1]))