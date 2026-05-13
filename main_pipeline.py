# main_pipeline.py

"""
用法：
  # 使用 config.py 預設開關，處理單一音檔
  python main_pipeline.py TC0408-01.mp3

  # 覆蓋開關：只跑 sensevoice 和 fw_large，不跑其他模型
  python main_pipeline.py TC0408-01.mp3 --models sensevoice fw_large

  # 只重跑 merge（3 份 SRT 已存在）
  python main_pipeline.py TC0408-01.mp3 --models merge
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ── 路徑設定 ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg

# ── Logger ────────────────────────────────────────────
cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(cfg.LOG_DIR / "pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── 模型對應表 ─────────────────────────────────────────
# key → (config 開關屬性名稱, transcribe 函數 lazy import)
MODEL_MAP = {
    "whisper":    ("ENABLE_WHISPER",       "src.transcribe_whisper",               "run"),
    "fw_medium":  ("ENABLE_FASTER_MEDIUM", "src.transcribe_faster_medium",         "run"),
    "fw_large":   ("ENABLE_FASTER_LARGE",  "src.transcribe_faster_large",          "run"),
    "sensevoice": ("ENABLE_SENSEVOICE",    "src.transcribe_sherpa_sensevoice",     "run"),
}


def resolve_enabled_models(cli_models: list[str] | None) -> dict[str, bool]:
    """
    決定這次執行要開啟哪些模型。
    cli_models 為 None 時使用 config.py 預設值；
    否則以 CLI 傳入的清單為準（空 list = 全部關閉）。
    """
    if cli_models is None:
        return {k: getattr(cfg, flag) for k, (flag, *_) in MODEL_MAP.items()}

    enabled = {k: False for k in MODEL_MAP}
    for m in cli_models:
        if m in enabled:
            enabled[m] = True
    return enabled


def main():
    parser = argparse.ArgumentParser(description="Voice transcription & merge pipeline")
    parser.add_argument("audio", help="音檔路徑（絕對或相對皆可）")
    parser.add_argument(
        "--models", nargs="*",
        help="指定要執行的模型，例如：--models sensevoice fw_large merge。"
             "不指定則使用 config.py 預設值。"
             "加入 'merge' 代表執行 LLM 融合；不加代表依 config.ENABLE_MERGE_LLM。"
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_absolute():
        audio_path = cfg.DATA_DIR / audio_path
    if not audio_path.exists():
        log.error(f"找不到音檔：{audio_path}")
        sys.exit(1)

    # ── 解析要執行的模型 ───────────────────────────────
    cli_models = args.models  # None 或 list
    run_merge  = cfg.ENABLE_MERGE_LLM  # 預設

    if cli_models is not None:
        run_merge  = "merge" in cli_models
        cli_models = [m for m in cli_models if m != "merge"]
        # cli_models 可能在移除 "merge" 後變成空 list（例如 --models merge）
        # 此時代表「不跑任何語音模型，只跑 merge」，需明確傳入空 list 而非 None
        enabled = resolve_enabled_models(cli_models)
    else:
        enabled = resolve_enabled_models(None)

    log.info(f"=== Pipeline 開始：{audio_path.name} ===")
    log.info(f"啟用模型：{[k for k,v in enabled.items() if v]}")
    log.info(f"執行 LLM merge：{run_merge}")

    t_start = time.time()
    produced_srt: dict[str, Path] = {}  # { model_key: srt_path }

    # ── 依序執行語音辨識 ───────────────────────────────
    for model_key, (_, module_path, func_name) in MODEL_MAP.items():
        if not enabled.get(model_key, False):
            log.info(f"[{model_key}] 跳過")
            continue
        try:
            import importlib
            mod = importlib.import_module(module_path)
            run_fn = getattr(mod, func_name)
            srt_path = run_fn(audio_path)
            if srt_path and srt_path.exists():
                produced_srt[model_key] = srt_path
        except Exception as e:
            log.error(f"[{model_key}] 執行失敗：{e}", exc_info=True)

    # ── 補上已存在（但這次沒跑）的 SRT，供 merge 使用 ──
    stem = audio_path.stem
    trans_dir = cfg.OUTPUT_DIR / stem / "transcriptions"
    for model_key, suffix in cfg.MODEL_SUFFIX.items():
        if model_key not in produced_srt:
            p = trans_dir / (stem + suffix)
            if p.exists():
                produced_srt[model_key] = p
                log.info(f"[{model_key}] 使用既有 SRT：{p.name}")

    # ── LLM Merge ─────────────────────────────────────
    if run_merge:
        try:
            from src.merge_llm import run as merge_run
            merge_run(audio_path, produced_srt)
        except Exception as e:
            log.error(f"[merge] 執行失敗：{e}", exc_info=True)

    log.info(f"=== Pipeline 完成，總耗時 {(time.time()-t_start)/60:.2f} 分鐘 ===")


if __name__ == "__main__":
    main()