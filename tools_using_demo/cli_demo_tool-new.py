import os
import platform
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_PATH = os.environ.get('MODEL_PATH', '../chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH, trust_remote_code=True)
if 'cuda' in DEVICE:  # AMD, NVIDIA GPU can use Half Precision
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True).to(DEVICE).eval()
else:  # CPU, Intel GPU and other GPU can use Float16 Precision Only
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True).float().to(DEVICE).eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM3-6B：{response}"
    return prompt


tools = [{"name": "get_activity_diagram_description", "description": "Get the activity diagram description in a given name", "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "The activity diagram name is military control system"}, }, "required": ["name"]}}, {'name': 'track', 'description': '追踪指定股票的实时价格', 'parameters': {'type': 'object', 'properties': {'symbol': {'description': '需要追踪的股票代码'}}, 'required': []}}, {'name': '/text-to-speech', 'description': '将文本转换为语音', 'parameters': {'type': 'object', 'properties': {'text': {'description': '需要转换成语音的文本'}, 'voice': {'description': '要使用的语音类型（男声、女声等）'}, 'speed': {'description': '语音的速度（快、中等、慢等）'}}, 'required': []}},
         {'name': '/image_resizer', 'description': '调整图片的大小和尺寸', 'parameters': {'type': 'object', 'properties': {'image_file': {'description': '需要调整大小的图片文件'}, 'width': {'description': '需要调整的宽度值'}, 'height': {'description': '需要调整的高度值'}}, 'required': []}}, {'name': '/foodimg', 'description': '通过给定的食品名称生成该食品的图片', 'parameters': {'type': 'object', 'properties': {'food_name': {'description': '需要生成图片的食品名称'}}, 'required': []}}]
system_item = {"role": "system",
               "content": "Answer the following questions as best as you can. You have access to the following tools:",
               "tools": tools}


def get_activity_diagram_description(name) -> str:
    if name == "military control system":
        text = """武控系统接收遥测频点命令。
                 武控系统设置遥测频点。
                 以上工作完成后，
                 武控系统反馈遥测频点。
                 1) 满足条件“else”时，武控系统处理遥测频点不一致故障。
                 2) 满足条件“Success”时，活动图结束。"""
    else:
        text = """并行开展以下工作：
                1.1 
                武器系统接收目标指示命令。
                1.2 
                武器系统接收目标数据。
                武器系统存储目标数据。
                武器系统处理目标数据。
                以上所有工作完成后，
                1.3 
                武器系统压点。
                以上任一工作完成后，
                武器系统目标库更新。
                武器系统航迹管理。
                武器系统主航迹表更新。
                武器系统显示目标信息。"""
    return text


def main():
    past_key_values, history = None, [system_item]
    role = "user"
    global stop_stream
    print("欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：") if role == "user" else input("\n结果：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None,  [system_item]
            role = "user"
            os.system(clear_command)
            print("欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        print("\nChatGLM：", end="")
        response, history = model.chat(
            tokenizer, query, history=history, role=role)
        print(response, end="", flush=True)
        print("")
        role = "user"
        if isinstance(response, dict):
            role = "observation"
            if response["name"] == "get_activity_diagram_description":
                result = get_activity_diagram_description(
                    response["parameters"]["name"])
                print(result)
                query = "请把下面的内容，按照你的理解，组织成一段语义通顺，让人容易理解的文字，以写文档的语气写，注意：不改变原文意思，不添加解释性的文字，不删减内容，请用中文回复提问。"
                query += result
                response, history = model.chat(
                    tokenizer, query, history=history, role=role)
                print(response, end="", flush=True)
                role = "user"


if __name__ == "__main__":
    main()
