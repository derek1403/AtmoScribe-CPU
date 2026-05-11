# src/transcribe_faster_large.py

import sys
import datetime
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_SUFFIX, OUTPUT_DIR
from prompts.prompt_Voice_to_Text import DOMAIN_PROMPT_faster_large as DOMAIN_PROMPT

try:
    from faster_whisper import WhisperModel
except ImportError:
    raise ImportError("請先安裝 faster-whisper：pip install faster-whisper")


def format_time(seconds: float) -> str:
    td = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    ms = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def run(audio_path: Path) -> Path:
    stem = audio_path.stem
    out_dir = OUTPUT_DIR / stem / "transcriptions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (stem + MODEL_SUFFIX["fw_large"])

    print(f"[FW-Large] 初始化 large-v3-turbo 模型 (CPU INT8)...")
    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")

    print(f"[FW-Large] 開始解析：{audio_path}")
    t0 = time.time()

    segments, info = model.transcribe(
        str(audio_path),
        language="zh",
        initial_prompt=DOMAIN_PROMPT,
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=400),
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
    )

    print(f"[FW-Large] 偵測語言：{info.language} (信心度 {info.language_probability:.2f})")

    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start_str = format_time(seg.start)
            end_str   = format_time(seg.end)
            text = seg.text.strip()
            if not text:
                continue
            f.write(f"{i}\n{start_str} --> {end_str}\n{text}\n\n")
            print(f"  [{start_str} -> {end_str}] {text}")

    print(f"[FW-Large] ✅ 完成，耗時 {time.time()-t0:.1f}s，輸出：{out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python src/transcribe_faster_large.py <audio_path>")
        sys.exit(1)
    run(Path(sys.argv[1]))