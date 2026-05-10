# config.py

from pathlib import Path

# ==========================================
# 路徑設定
# ==========================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data" / "audio_inputs"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
LOG_DIR      = PROJECT_ROOT / "logs"

# ==========================================
# 模型開關（True = 執行，False = 跳過）
# 可在 main_pipeline.py 或 CLI 參數覆蓋
# ==========================================
ENABLE_WHISPER         = True
ENABLE_FASTER_MEDIUM   = False   # 預設關閉，不進 LLM merge
ENABLE_FASTER_LARGE    = True
ENABLE_SENSEVOICE      = True
ENABLE_MERGE_LLM       = True

# ==========================================
# 模型優先級（數字越小越高）
# 決定 LLM merge 時誰的時間軸作為基準
# ==========================================
MODEL_PRIORITY = {
    "sensevoice": 1,
    "fw_large":   2,
    "fw_medium":  3,
    "whisper":    4,
}

# 預設進入 LLM merge 的模型（可在此調整）
MERGE_CANDIDATES = ["sensevoice", "fw_large", "whisper"]

# ==========================================
# SRT 輸出後綴（對應 MODEL_PRIORITY 的 key）
# ==========================================
MODEL_SUFFIX = {
    "whisper":    "_whisper.srt",
    "fw_medium":  "_fw_medium.srt",
    "fw_large":   "_fw_large.srt",
    "sensevoice": "_sensevoice.srt",
}

# ==========================================
# LLM 模型設定（Qwen 7B）
# ==========================================
LLM_REPO_ID  = "Qwen/Qwen2.5-7B-Instruct-GGUF"
LLM_FILENAME = "qwen2.5-7b-instruct-q3_k_m.gguf"
LLM_N_CTX    = 2048
LLM_TEMP     = 0.1