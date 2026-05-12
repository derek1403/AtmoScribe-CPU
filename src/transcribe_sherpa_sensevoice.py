# src/transcribe_sherpa_sensevoice.py

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_SUFFIX, OUTPUT_DIR, SHERPA_SENSEVOICE_MODEL_DIR

try:
    import sherpa_onnx
    import librosa
except ImportError:
    raise ImportError("請先安裝相依套件：pip install sherpa-onnx librosa")


# ==========================================
# 輔助函數
# ==========================================
def format_time(seconds: float) -> str:
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def save_to_srt(result, out_path: Path, chars_per_line: int = 15) -> bool:
    """
    將 sherpa-onnx result 轉成 SRT 並寫入 out_path。
    回傳 True 表示成功，False 表示沒有時間戳無法生成。
    """
    tokens     = result.tokens
    timestamps = result.timestamps

    if not tokens or not timestamps:
        print("[Sherpa-SenseVoice] ⚠️ 模型未輸出時間戳記，無法生成 SRT。")
        return False

    srt_blocks   = []
    chunk_tokens = []
    start_time   = timestamps[0]
    chunk_idx    = 1

    for i, (token, t) in enumerate(zip(tokens, timestamps)):
        # 過濾語種標籤，例如 <|zh|>、<|NEUTRAL|>
        if "<|" in token or "|>" in token:
            continue

        chunk_tokens.append(token)

        if len(chunk_tokens) >= chars_per_line or i == len(tokens) - 1:
            end_time  = t + 0.5
            text_str  = "".join(chunk_tokens).replace(" ", "").replace("\u3000", "")

            if text_str:   # 過濾空行
                srt_blocks.append(
                    f"{chunk_idx}\n"
                    f"{format_time(start_time)} --> {format_time(end_time)}\n"
                    f"{text_str}\n"
                )
                chunk_idx += 1

            chunk_tokens = []
            if i + 1 < len(timestamps):
                start_time = timestamps[i + 1]

    if not srt_blocks:
        print("[Sherpa-SenseVoice] ⚠️ 過濾後無任何有效文字。")
        return False

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_blocks))

    return True


# ==========================================
# 主函數（供 main_pipeline.py 呼叫）
# ==========================================
def run(audio_path: Path) -> Path | None:
    """
    執行 sherpa-onnx SenseVoice 語音辨識，輸出 SRT。
    回傳輸出的 SRT 路徑，失敗時回傳 None。
    """
    stem    = audio_path.stem
    out_dir = OUTPUT_DIR / stem / "transcriptions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (stem + MODEL_SUFFIX["sensevoice"])

    model_dir   = SHERPA_SENSEVOICE_MODEL_DIR
    model_file  = model_dir / "model.int8.onnx"
    tokens_file = model_dir / "tokens.txt"

    if not model_file.exists() or not tokens_file.exists():
        print(f"[Sherpa-SenseVoice] ❌ 找不到模型檔案，請確認路徑：{model_dir}")
        print("  需要的檔案：model.int8.onnx、tokens.txt")
        return None

    print("[Sherpa-SenseVoice] 載入模型...")
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model_file),
        tokens=str(tokens_file),
        num_threads=4,
        use_itn=True,
        language="",   # 空字串 = 自動偵測語種
        debug=False,
    )

    print(f"[Sherpa-SenseVoice] 讀取音檔：{audio_path}")
    t0 = time.time()
    audio_data, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)

    print("[Sherpa-SenseVoice] 開始辨識...")
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio_data)
    recognizer.decode_stream(stream)

    print(f"\n[Sherpa-SenseVoice] 辨識文字：\n{stream.result.text}\n")

    success = save_to_srt(stream.result, out_path, chars_per_line=15)
    if not success:
        return None

    print(f"[Sherpa-SenseVoice] ✅ 完成，耗時 {time.time()-t0:.1f}s，輸出：{out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python src/transcribe_sherpa_sensevoice.py <audio_path>")
        sys.exit(1)
    run(Path(sys.argv[1]))