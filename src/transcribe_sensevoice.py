# src/transcribe_sensevoice.py

import sys
import datetime
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# 讓 src/ 下的檔案可以 import 專案根目錄的模組
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_SUFFIX, OUTPUT_DIR
from prompts.prompt_Voice_to_Text import DOMAIN_PROMPT_sensevoice as DOMAIN_PROMPT # noqa: F401（SenseVoice 不使用 initial_prompt，保留備用）

try:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
except ImportError:
    raise ImportError("請先安裝 funasr：pip install funasr")


def format_time(seconds: float) -> str:
    td = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    ms = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def run(audio_path: Path) -> Path:
    """
    執行 SenseVoice 語音辨識，輸出 SRT 至 outputs/{stem}/transcriptions/
    回傳輸出的 SRT 路徑。
    """
    stem = audio_path.stem
    out_dir = OUTPUT_DIR / stem / "transcriptions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (stem + MODEL_SUFFIX["sensevoice"])

    print(f"[SenseVoice] 初始化模型 (CPU)...")
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 15000},
        device="cpu",
        disable_pbar=True,   # 繞過 FunASR 1.3.1 的 punc_res UnboundLocalError
    )

    print(f"[SenseVoice] 開始解析：{audio_path}")
    t0 = time.time()

    res = model.generate(
        input=str(audio_path),
        cache={},
        language="zh",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
        sentence_timestamp=True,
    )

    sentence_info_list = res[0].get("sentence_info", [])
    if not sentence_info_list:
        print("[SenseVoice] ⚠️ 無法獲取句子層級時間戳，請確認 FunASR 版本。")
        return None

    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(sentence_info_list, start=1):
            start_str = format_time(seg["start"] / 1000.0)
            end_str   = format_time(seg["end"]   / 1000.0)
            text = rich_transcription_postprocess(seg["text"]).strip()
            if not text:
                continue
            f.write(f"{i}\n{start_str} --> {end_str}\n{text}\n\n")
            print(f"  [{start_str} -> {end_str}] {text}")

    print(f"[SenseVoice] ✅ 完成，耗時 {time.time()-t0:.1f}s，輸出：{out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python src/transcribe_sensevoice.py <audio_path>")
        sys.exit(1)
    run(Path(sys.argv[1]))