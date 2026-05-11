import os
import stat
from pathlib import Path

def generate_sh_script():
    # 1. 定義路徑（使用 pathlib.Path 確保跨平台相容性）
    input_dir = Path("./data/audio_inputs")
    output_dir = Path("./outputs")
    sh_filename = "run_tasks.sh"

    # 確保輸出目錄存在，若不存在則建立（避免比對時報錯）
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # 2. 獲取所有 mp3 檔案的名稱 (去掉 .mp3)
    # A 集合：input 內的檔案名稱（不含副檔名）
    # 我們只抓 .mp3 結尾的，避免抓到隱藏檔
    all_mp3_files = {f.name for f in input_dir.glob("*.mp3")}
    all_stems = {f.stem for f in input_dir.glob("*.mp3")}

    # 3. 獲取已經處理過的資料夾名稱
    # B 集合：output 內已存在的子目錄
    processed_folders = {d.name for d in output_dir.iterdir() if d.is_dir()}

    # 4. 計算差集 (A - B): 找出還沒被處理的 stems
    to_process_stems = all_stems - processed_folders

    if not to_process_stems:
        print("🎉 所有音檔皆已處理完畢，不需要產生腳本。")
        return

    # 5. 產生 sh 檔案內容
    print(f"🔎 發現 {len(to_process_stems)} 個新檔案，準備寫入 {sh_filename}...")
    
    with open(sh_filename, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# 自動產生的任務清單\n")
        
        # 按照字母順序排序，讓執行順序比較好預測
        for stem in sorted(to_process_stems):
            mp3_name = f"{stem}.mp3"
            log_name = f"{stem}.log"
            # 依照你之前的格式：python3 task.py > log 2>&1
            f.write(f"python3 main_pipeline.py {mp3_name} > {log_name} 2>&1\n")

    # 6. 自動 chmod +x (賦予執行權限)
    # 這裡解釋一下：os.stat(path).st_mode 取得目前權限
    # stat.S_IXUSR 代表 Owner Execute 權限 (0o100)
    current_mode = os.stat(sh_filename).st_mode
    os.chmod(sh_filename, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"✅ {sh_filename} 已生成並賦予執行權限。")
    print(f"👉 請輸入: nohup ./{sh_filename} &")

if __name__ == "__main__":
    generate_sh_script()