# prompts/prompt_Voice_to_Text.py



# transcriber 的 initial_prompt，用於引導模型更好地辨識專業術語。
# 其他 3 個語音模型的 prompt_*.py 格式相同，只需複製此檔並修改內容即可

DOMAIN_PROMPT_whisper = (
    "專業大氣科學與中文天氣預報研討會，包含許多專有名詞，例如："
    "位渦、梯度風平衡、正壓不穩定、沉降逆溫、多邊形眼牆、條理分明的結構、"
    "渦度梯度、變分學、環流守恆定理、尺度匹配、Noether's Theorem, AI"
)

DOMAIN_PROMPT_faster_medium = (
    "專業大氣科學與中文天氣預報研討會，包含許多專有名詞，例如："
    "位渦、梯度風平衡、正壓不穩定、沉降逆溫、多邊形眼牆、條理分明的結構、"
    "渦度梯度、變分學、環流守恆定理、尺度匹配、Noether's Theorem, AI"
)

DOMAIN_PROMPT_faster_large = (
    "專業大氣科學與中文天氣預報研討會，包含許多專有名詞，例如："
    "位渦、梯度風平衡、正壓不穩定、沉降逆溫、多邊形眼牆、條理分明的結構、"
    "渦度梯度、變分學、環流守恆定理、尺度匹配、Noether's Theorem, AI"
)

DOMAIN_PROMPT_sensevoice = (
    "專業大氣科學與中文天氣預報研討會，包含許多專有名詞，例如："
    "位渦、梯度風平衡、正壓不穩定、沉降逆溫、多邊形眼牆、條理分明的結構、"
    "渦度梯度、變分學、環流守恆定理、尺度匹配、Noether's Theorem, AI"
)

