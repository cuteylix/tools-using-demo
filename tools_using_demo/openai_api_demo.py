import json
import os

from openai import OpenAI
from openai import AzureOpenAI
from colorama import init, Fore
from loguru import logger

from tool_register import get_tools, dispatch_tool

init(autoreset=True)
client = OpenAI(
  base_url = "https://chatapi.a3e.top/v1",
  api_key = "test"
)

def get_activity_diagram_description(name) -> str:
    if name == 'military control system':
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

def run_conversation2():
    messages = [{
        "role": "user",
        "content": "What's the activity diagram description in military control system"
    }]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_activity_diagram_description",
                "description": "Get the activity diagram description in a given name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The activity diagram name is military control system"
                        },
                    },
                    "required": ["name"]
                }
            }
        }
    ]
	
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    if tool_calls:
        available_functions = {
            "get_activity_diagram_description": get_activity_diagram_description,
        }  
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = globals().get(function_name)
            
            if function_to_call:
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                
                function_response = function_to_call(
                name=function_args.get("name"),
            )
                messages.append({
                      "role": "assistant",
                      "content": str(response),
                })
                messages.append({
                      "tool_call_id": tool_call.id,
                      "role": "function",
                      "name": function_name,
                      "content": f'请把下面的内容，按照你的理解，组织成一段语义通顺，让人容易理解的文字，以写文档的语气写，注意：不改变原文意思，不添加解释性的文字，不删减内容，请用中文回复提问。' + function_response,
                })
            else:
                print(f"No function found for the name {function_name}")
    else:
        print("No tool calls made by the model.")
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="none"
    )
    messages.append({
        "role": "assistant",
        "content": str(response),
    })
    
    print(messages)
    return json.dumps(messages, indent=2)

'''
tools = get_tools()
'''

def run_conversation(query: str, stream=False, tools=None, max_retry=5):
    params = dict(model="gpt-3.5-turbo-0613", messages=[{"role": "user", "content": query}], stream=stream)
    if tools:
        params["tools"] = tools
    response = client.chat.completions.create(**params)

    for _ in range(max_retry):
        if not stream:
            if response.choices[0].message.function_call:
                function_call = response.choices[0].message.function_call
                logger.info(f"Function Call Response: {function_call.model_dump()}")

                function_args = json.loads(function_call.arguments)
                tool_response = dispatch_tool(function_call.name, function_args)
                logger.info(f"Tool Call Response: {tool_response}")

                params["messages"].append(response.choices[0].message)
                params["messages"].append(
                    {
                        "role": "function",
                        "name": function_call.name,
                        "content": tool_response,  # 调用函数返回结果
                    }
                )
            else:
                reply = response.choices[0].message.content
                logger.info(f"Final Reply: \n{reply}")
                return

        else:
            output = ""
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(Fore.BLUE + content, end="", flush=True)
                output += content

                if chunk.choices[0].finish_reason == "stop":
                    return

                elif chunk.choices[0].finish_reason == "function_call":
                    print("\n")

                    function_call = chunk.choices[0].delta.function_call
                    logger.info(f"Function Call Response: {function_call.model_dump()}")

                    function_args = json.loads(function_call.arguments)
                    tool_response = dispatch_tool(function_call.name, function_args)
                    logger.info(f"Tool Call Response: {tool_response}")

                    params["messages"].append(
                        {
                            "role": "assistant",
                            "content": output
                        }
                    )
                    params["messages"].append(
                        {
                            "role": "function",
                            "name": function_call.name,
                            "content": tool_response,
                        }
                    )

                    break

        response = client.chat.completions.create(**params)



if __name__ == "__main__": 
    run_conversation2()
