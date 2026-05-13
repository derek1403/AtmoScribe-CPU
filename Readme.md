# Voice Transcription & Merge Pipeline

大氣科學學術錄音的自動語音辨識與 LLM 校稿系統。
將單一錄音檔透過多個語音模型平行轉錄，再由 Qwen 7B 融合校正專有名詞，輸出一份高品質 SRT 字幕。

```
1 個音檔 → 3 個語音模型依序處理 → 3 份 SRT → Qwen 7B 融合 → 1 份最終 SRT
```

---

## 專案結構

```
voice_project/
├── data/
│   └── audio_inputs/           # 放原始 .mp3 / .m4a 錄音檔
├── models/
│   └── sherpa_sensevoice/      # sherpa-onnx SenseVoice 模型檔案
│       ├── model.int8.onnx
│       └── tokens.txt
├── src/
│   ├── transcribe_whisper.py
│   ├── transcribe_faster_medium.py
│   ├── transcribe_faster_large.py
│   ├── transcribe_sherpa_sensevoice.py
│   └── merge_llm.py
├── prompts/
│   ├── prompt_Voice_to_Text.py
│   └── prompt_merge_llm.py
├── logs/                       # pipeline.log 自動生成於此
├── outputs/
│   └── {audio_stem}/
│       ├── transcriptions/     # 各語音模型輸出的 SRT
│       └── merged_llm/         # LLM 融合後的最終 SRT
├── generate_task_runner_sh_script.py # 生成自動化腳本
├── config.py                   # 模型開關、優先級、路徑設定
├── main_pipeline.py            # 主流程控制
└── README.md
```

---

## 環境需求

- Python 3.10+
- conda 或 venv 虛擬環境（建議）
- CPU 執行環境（無需 GPU）

---

## 安裝

**1. 建立虛擬環境**

```bash
conda create -n voice python=3.11
conda activate voice
```

**2. 安裝各語音模型套件**

```bash
# OpenAI Whisper
pip install openai-whisper

# Faster-Whisper（medium 與 large-v3-turbo 共用）
pip install faster-whisper

# sherpa-onnx SenseVoice
pip install sherpa-onnx soundfile numpy

# LLM merge（Qwen 7B GGUF）
pip install llama-cpp-python
```

**3. 安裝系統套件**

```bash
# ffmpeg（sherpa-onnx SenseVoice 將 MP3 轉 WAV 時需要）
sudo apt install ffmpeg      # Ubuntu / Debian
# 或
conda install -c conda-forge ffmpeg
```

確認安裝成功：
```bash
ffmpeg -version
```

**4. 放置 sherpa-onnx SenseVoice 模型檔案**

將以下兩個檔案放入 `models/sherpa_sensevoice/`：
- `model.int8.onnx`
- `tokens.txt`

可從 [sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17](https://github.com/k2-fsa/sherpa-onnx/releases) 下載。

> **注意：** 第一次執行各模型時，會自動從 Hugging Face 下載模型權重，需要一段時間與足夠的磁碟空間。
> - `large-v3-turbo`（INT8）：約 1.6 GB
> - `sherpa-onnx SenseVoice`（INT8）：約 400 MB（需手動放置）
> - Qwen 2.5-7B Q3：約 3.5 GB

---

## 快速開始

將錄音檔放入 `data/audio_inputs/`，然後執行：

```bash
python main_pipeline.py TC0408-01.mp3
```

輸出結果會在：
```
outputs/TC0408-01/
├── transcriptions/
│   ├── TC0408-01_whisper.srt
│   ├── TC0408-01_fw_large.srt
│   └── TC0408-01_sensevoice.srt
└── merged_llm/
    └── TC0408-01_merged.srt    ← 最終結果
```

---

## 進階用法

### 指定要執行的模型

```bash
# 只跑 sensevoice 和 fw_large（不跑 whisper，不跑 merge）
python main_pipeline.py TC0408-01.mp3 --models sensevoice fw_large

# 只重跑 LLM merge（3 份 SRT 已存在，不重新語音辨識）
python main_pipeline.py TC0408-01.mp3 --models merge

# 只跑 sensevoice，然後直接 merge
python main_pipeline.py TC0408-01.mp3 --models sensevoice merge
```

> `--models` 未指定時，使用 `config.py` 的預設開關。

### 單獨執行某個語音模型

```bash
python src/transcribe_sensevoice.py data/audio_inputs/TC0408-01.mp3
python src/transcribe_faster_large.py data/audio_inputs/TC0408-01.mp3
python src/transcribe_whisper.py data/audio_inputs/TC0408-01.mp3
```

### 單獨執行 LLM merge

```bash
python src/merge_llm.py data/audio_inputs/TC0408-01.mp3
```

會自動從 `outputs/{stem}/transcriptions/` 尋找所有可用的 SRT。

---

## 模型說明與優先級

| 優先級 | 模型 key | 說明 | 預設啟用 | 進入 LLM merge |
|:---:|---|---|:---:|:---:|
| 1 | `sensevoice` | SenseVoiceSmall，時間軸最準，作為 merge 基準 | ✅ | ✅ |
| 2 | `fw_large` | Faster-Whisper large-v3-turbo，辨識率高 | ✅ | ✅ |
| 3 | `fw_medium` | Faster-Whisper medium，速度較快 | ❌ | ❌ |
| 4 | `whisper` | OpenAI Whisper small，輔助參考 | ✅ | ✅ |

優先級決定 LLM merge 時誰的**時間軸作為骨幹**，數字越小越優先。

---

## 調整設定

所有關鍵設定集中在 `config.py`：

```python
# 模型開關
ENABLE_WHISPER       = True
ENABLE_FASTER_MEDIUM = False   # 預設關閉
ENABLE_FASTER_LARGE  = True
ENABLE_SENSEVOICE    = True
ENABLE_MERGE_LLM     = True

# 預設進入 LLM merge 的模型
MERGE_CANDIDATES = ["sensevoice", "fw_large", "whisper"]
```

### 修改提示詞

每個模型有獨立的提示詞檔案，位於 `prompts/`：

- 語音辨識提示詞（`DOMAIN_PROMPT`）：調整大氣科學專有名詞清單
- LLM merge 提示詞（`SYSTEM_PROMPT`）：調整校稿規則與風格

---

## 常見問題

**Q：想換成不同的 Qwen GGUF 量化版本**

修改 `config.py` 中的 `LLM_FILENAME`，例如：
```python
LLM_FILENAME = "qwen2.5-7b-instruct-q4_k_m.gguf"  # 更高品質，但更慢
```
---

## 輸出檔案命名規則

| 模型 | 輸出後綴 |
|---|---|
| Whisper | `_whisper.srt` |
| Faster-Whisper medium | `_fw_medium.srt` |
| Faster-Whisper large-v3-turbo | `_fw_large.srt` |
| SenseVoice | `_sensevoice.srt` |
| LLM 融合結果 | `_merged.srt` |

---

## 致謝、共同編程
Gemini🥇
Claude🥈
PC(我)🥉