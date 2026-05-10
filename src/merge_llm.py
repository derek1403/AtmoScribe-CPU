# src/merge_llm.py

import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODEL_PRIORITY, MODEL_SUFFIX, MERGE_CANDIDATES,
    OUTPUT_DIR, LLM_REPO_ID, LLM_FILENAME, LLM_N_CTX, LLM_TEMP
)
from prompts.prompt_merge_llm import SYSTEM_PROMPT, build_user_prompt

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError("請先安裝 llama-cpp-python：pip install llama-cpp-python")


# ==========================================
# 輔助函數
# ==========================================
def time_to_sec(t: str) -> float:
    t = t.strip()
    h, m, s_ms = t.split(':')
    s, ms = s_ms.split(',')
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0


def parse_srt(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8").strip()
    blocks = re.split(r'\n\n+', content)
    result = []
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            start_str, end_str = lines[1].split(' --> ')
            result.append({
                "index": lines[0],
                "times": lines[1],
                "start_sec": time_to_sec(start_str),
                "end_sec":   time_to_sec(end_str),
                "text": " ".join(lines[2:])
            })
    return result


def find_overlapping(ref_segs: list[dict], n_start: float, n_end: float) -> str:
    """找出 ref_segs 中與 [n_start, n_end] 有時間交集的所有句子，合併成一個字串。"""
    matched = [
        seg["text"] for seg in ref_segs
        if max(n_start, seg["start_sec"]) < min(n_end, seg["end_sec"])
    ]
    return " ".join(matched) if matched else "(此時間段無參考資料)"


# ==========================================
# 主函數
# ==========================================
def run(audio_path: Path, available_srt: dict[str, Path]) -> Path:
    """
    audio_path    : 原始音檔（用來決定輸出目錄）
    available_srt : { model_key: srt_path }，只包含實際產生的 SRT

    回傳輸出的 merged SRT 路徑。
    """
    stem = audio_path.stem
    out_dir = OUTPUT_DIR / stem / "merged_llm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_merged.srt"

    # ------------------------------------------
    # 1. 決定哪些模型進入 merge
    # ------------------------------------------
    # 只考慮 MERGE_CANDIDATES 且實際有產生 SRT 的模型
    candidates = [k for k in MERGE_CANDIDATES if k in available_srt]

    if len(candidates) == 0:
        print("[Merge] ❌ 沒有可用的 SRT，無法執行 merge。")
        return None
    if len(candidates) == 1:
        print(f"[Merge] ⚠️ 只有 1 份 SRT（{candidates[0]}），直接複製為最終結果。")
        import shutil
        shutil.copy(available_srt[candidates[0]], out_path)
        return out_path

    # ------------------------------------------
    # 2. 依優先級排序，最高優先的作為時間軸基準
    # ------------------------------------------
    candidates.sort(key=lambda k: MODEL_PRIORITY.get(k, 99))
    base_key  = candidates[0]
    ref_keys  = candidates[1:]

    print(f"[Merge] 時間軸基準：{base_key}")
    print(f"[Merge] 輔助參考：{ref_keys}")

    base_segs = parse_srt(available_srt[base_key])
    ref_segs_map = {k: parse_srt(available_srt[k]) for k in ref_keys}

    # ------------------------------------------
    # 3. 載入 LLM
    # ------------------------------------------
    print("[Merge] 載入 Qwen 7B...")
    llm = Llama.from_pretrained(
        repo_id=LLM_REPO_ID,
        filename=LLM_FILENAME,
        n_ctx=LLM_N_CTX,
        verbose=False,
    )

    # ------------------------------------------
    # 4. 逐句 merge
    # ------------------------------------------
    total = len(base_segs)
    print(f"\n[Merge] 開始融合 {total} 句...\n")
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(base_segs):
            n_start, n_end = seg["start_sec"], seg["end_sec"]

            ref_texts = [
                find_overlapping(ref_segs_map[k], n_start, n_end)
                for k in ref_keys
            ]

            user_prompt = build_user_prompt(seg["text"], ref_texts)

            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=LLM_TEMP,
            )
            merged_text = resp["choices"][0]["message"]["content"].strip()

            f.write(f"{seg['index']}\n{seg['times']}\n{merged_text}\n\n")

            print(f"進度 [{i+1}/{total}] {seg['times']}")
            for k, t in zip(ref_keys, ref_texts):
                print(f"  {k}: {t}")
            print(f"  基準 ({base_key}): {seg['text']}")
            print(f"  ✨ 結果: {merged_text}")
            print("-" * 50)

    print(f"\n[Merge] 🎉 完成！輸出：{out_path}")
    print(f"[Merge] 總耗時：{(time.time()-t0)/60:.2f} 分鐘")
    return out_path


if __name__ == "__main__":
    # 單獨執行示範：python src/merge_llm.py <audio_path>
    # 會自動從 outputs/{stem}/transcriptions/ 尋找可用 SRT
    if len(sys.argv) < 2:
        print("用法：python src/merge_llm.py <audio_path>")
        sys.exit(1)

    audio = Path(sys.argv[1])
    stem  = audio.stem
    trans_dir = OUTPUT_DIR / stem / "transcriptions"

    found = {}
    for model_key, suffix in MODEL_SUFFIX.items():
        p = trans_dir / (stem + suffix)
        if p.exists():
            found[model_key] = p

    if not found:
        print(f"[Merge] 找不到任何 SRT，請先執行語音辨識。")
        sys.exit(1)

    run(audio, found)