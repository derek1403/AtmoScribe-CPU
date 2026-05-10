# Voice Project 重構架構

```
voice_project/
├── data/
│   └── audio_inputs/               # 放原始 .mp3 / .m4a 錄音檔
│
├── models/                         # 保留，未來放本機 GGUF 等檔案
│
├── src/
│   ├── transcribe_whisper.py           # 原 whisper 版本
│   ├── transcribe_faster_medium.py     # 原 faster-whisper medium
│   ├── transcribe_faster_large.py      # 原 faster-whisper large-v3-turbo
│   ├── transcribe_sensevoice.py        # 原 sensevoice
│   └── merge_llm.py                    # 原 merge_srt_7B.py
│
├── prompts/
│   ├── prompt_whisper.py
│   ├── prompt_faster_medium.py
│   ├── prompt_faster_large.py
│   ├── prompt_sensevoice.py
│   └── prompt_merge_llm.py
│
├── logs/
│
├── outputs/
│   └── {audio_stem}/               # 例如 TC0408-01/
│       ├── transcriptions/         # 4 個語音模型輸出的 SRT
│       └── merged_llm/             # LLM 融合後的最終 SRT
│
├── config.py                       # 模型開關、優先級、路徑設定
├── main_pipeline.py                # 主流程控制
└── README.md
```

---

## 檔案命名規則（SRT 輸出）

| 模型 | 輸出檔名 |
|---|---|
| Whisper (openai) | `{stem}_whisper.srt` |
| Faster-Whisper medium | `{stem}_fw_medium.srt` |
| Faster-Whisper large-v3-turbo | `{stem}_fw_large.srt` |
| SenseVoice | `{stem}_sensevoice.srt` |
| LLM 融合結果 | `{stem}_merged.srt` |

---

## 模型優先級（決定 merge 時間軸基準）

```
1. sensevoice      ← 最高，時間軸基準優先
2. fw_large        ← large-v3-turbo
3. fw_medium       ← medium（預設不進 LLM merge）
4. whisper         ← 最低
```

預設進入 LLM merge 的 3 個模型：`sensevoice`、`fw_large`、`whisper`

---

## 流程說明

```
main_pipeline.py
│
├─ 讀取 config.py → 決定這次跑哪些模型
│
├─ 依序執行被啟用的 transcribe_*.py
│   └─ 輸出至 outputs/{stem}/transcriptions/
│
└─ 執行 merge_llm.py
    ├─ 根據優先級，找出可用 SRT 中最高優先的作為時間軸基準
    ├─ 其餘 SRT 作為輔助參考
    └─ 輸出至 outputs/{stem}/merged_llm/
```