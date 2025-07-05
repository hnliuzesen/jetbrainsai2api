import json
import time
import uuid
import threading
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Configuration
DEFAULT_REQUEST_TIMEOUT = 30.0

# Global variables
VALID_CLIENT_KEYS: set = set()
JETBRAINS_ACCOUNTS: list = []
current_account_index: int = 0
account_rotation_lock = threading.Lock()
models_data: Dict[str, Any] = {}
http_client: Optional[httpx.AsyncClient] = None


# Pydantic Models
class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    tools: Optional[List[Dict[str, Any]]] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class ChatCompletionChoice(BaseModel):
    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )


class StreamChoice(BaseModel):
    delta: Dict[str, Any] = Field(default_factory=dict)
    index: int = 0
    finish_reason: Optional[str] = None


class StreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]


# FastAPI App
app = FastAPI(title="JetBrains AI OpenAI Compatible API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

security = HTTPBearer(auto_error=False)


# Helper functions
def load_models():
    """加载模型配置"""
    try:
        with open("models.json", "r", encoding="utf-8") as f:
            model_ids = json.load(f)

        processed_models = []
        if isinstance(model_ids, list):
            for model_id in model_ids:
                if isinstance(model_id, str):
                    processed_models.append(
                        {
                            "id": model_id,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "jetbrains-ai",
                        }
                    )

        return {"data": processed_models}
    except Exception as e:
        print(f"加载 models.json 时出错: {e}")
        return {"data": []}


def load_client_api_keys():
    """加载客户端 API 密钥"""
    global VALID_CLIENT_KEYS
    try:
        with open("client_api_keys.json", "r", encoding="utf-8") as f:
            keys = json.load(f)
            if not isinstance(keys, list):
                print("警告: client_api_keys.json 应包含密钥列表")
                VALID_CLIENT_KEYS = set()
                return
            VALID_CLIENT_KEYS = set(keys)
            if not VALID_CLIENT_KEYS:
                print("警告: client_api_keys.json 为空")
            else:
                print(f"成功加载 {len(VALID_CLIENT_KEYS)} 个客户端 API 密钥")
    except FileNotFoundError:
        print("错误: 未找到 client_api_keys.json")
        VALID_CLIENT_KEYS = set()
    except Exception as e:
        print(f"加载 client_api_keys.json 时出错: {e}")
        VALID_CLIENT_KEYS = set()


def load_jetbrains_accounts():
    """加载 JetBrains AI 认证信息"""
    global JETBRAINS_ACCOUNTS
    try:
        with open("jetbrainsai.json", "r", encoding="utf-8") as f:
            accounts_data = json.load(f)

        if not isinstance(accounts_data, list):
            print("警告: jetbrainsai.json 格式不正确，应为对象列表")
            JETBRAINS_ACCOUNTS = []
            return

        processed_accounts = []
        for account in accounts_data:
            if "licenseId" in account and "authorization" in account:
                processed_accounts.append(
                    {
                        "licenseId": account["licenseId"],
                        "authorization": account["authorization"],
                        "jwt": None,
                        "last_updated": 0,
                    }
                )
            elif "jwt" in account:
                processed_accounts.append(
                    {
                        "licenseId": None,
                        "authorization": None,
                        "jwt": account["jwt"],
                        "last_updated": 0,
                    }
                )

        JETBRAINS_ACCOUNTS = processed_accounts
        if not JETBRAINS_ACCOUNTS:
            print("警告: jetbrainsai.json 中未找到有效的认证信息")
        else:
            print(f"成功加载 {len(JETBRAINS_ACCOUNTS)} 个 JetBrains AI 账户")

    except FileNotFoundError:
        print("错误: 未找到 jetbrainsai.json 文件")
        JETBRAINS_ACCOUNTS = []
    except Exception as e:
        print(f"加载 jetbrainsai.json 时出错: {e}")
        JETBRAINS_ACCOUNTS = []


def get_model_item(model_id: str) -> Optional[Dict]:
    """根据模型ID获取模型配置"""
    for model in models_data.get("data", []):
        if model.get("id") == model_id:
            return model
    return None


async def authenticate_client(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """客户端认证"""
    if not VALID_CLIENT_KEYS:
        raise HTTPException(status_code=503, detail="服务不可用: 未配置客户端 API 密钥")

    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=401,
            detail="需要在 Authorization header 中提供 API 密钥",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if auth.credentials not in VALID_CLIENT_KEYS:
        raise HTTPException(status_code=403, detail="无效的客户端 API 密钥")


async def _refresh_jetbrains_jwt(account: dict):
    """使用 licenseId 和 authorization 刷新 JWT"""
    if not http_client:
        raise HTTPException(status_code=500, detail="HTTP 客户端未初始化")

    print(f"正在为 licenseId {account['licenseId']} 刷新 JWT...")
    try:
        headers = {
            "User-Agent": "ktor-client",
            "Content-Type": "application/json",
            "Accept-Charset": "UTF-8",
            "authorization": f"Bearer {account['authorization']}",
        }
        payload = {"licenseId": account["licenseId"]}

        response = await http_client.post(
            "https://api.jetbrains.ai/auth/jetbrains-jwt/provide-access/license/v2",
            json=payload,
            headers=headers,
            timeout=DEFAULT_REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json()
        if data.get("state") == "PAID" and "token" in data:
            account["jwt"] = data["token"]
            account["last_updated"] = time.time()
            print(f"成功刷新 licenseId {account['licenseId']} 的 JWT")
        else:
            print(f"刷新 JWT 失败: 无效的响应状态 {data.get('state')}")
            raise HTTPException(status_code=500, detail=f"刷新 JWT 失败: {data}")

    except httpx.HTTPStatusError as e:
        print(f"刷新 JWT 时 HTTP 错误: {e.response.status_code} {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"刷新 JWT 失败: {e.response.text}",
        )
    except Exception as e:
        print(f"刷新 JWT 时发生未知错误: {e}")
        raise HTTPException(status_code=500, detail=f"刷新 JWT 时发生未知错误: {e}")


async def get_next_jetbrains_jwt() -> str:
    """轮询获取下一个 JetBrains JWT，必要时刷新"""
    global current_account_index

    if not JETBRAINS_ACCOUNTS:
        raise HTTPException(status_code=503, detail="服务不可用: 未配置 JetBrains 账户")

    with account_rotation_lock:
        account = JETBRAINS_ACCOUNTS[current_account_index]
        current_account_index = (current_account_index + 1) % len(JETBRAINS_ACCOUNTS)

    # 如果是基于许可证的账户，检查是否需要刷新
    if account.get("licenseId"):
        is_stale = (
            time.time() - account.get("last_updated", 0) > 12 * 3600
        )  # 12 小时有效期
        if not account.get("jwt") or is_stale:
            await _refresh_jetbrains_jwt(account)

    if not account.get("jwt"):
        raise HTTPException(
            status_code=503,
            detail=f"无法为 licenseId {account.get('licenseId')} 获取有效的 JWT",
        )

    return account["jwt"]


# FastAPI 生命周期事件
@app.on_event("startup")
async def startup():
    global models_data, http_client
    models_data = load_models()
    load_client_api_keys()
    load_jetbrains_accounts()
    http_client = httpx.AsyncClient(timeout=None)
    print("JetBrains AI OpenAI Compatible API 服务器已启动")


@app.on_event("shutdown")
async def shutdown():
    global http_client
    if http_client:
        await http_client.aclose()


# API 端点
@app.get("/v1/models", response_model=ModelList)
async def list_models(_: None = Depends(authenticate_client)):
    """列出可用模型"""
    model_list = [
        ModelInfo(
            id=model.get("id", ""),
            created=model.get("created", int(time.time())),
            owned_by=model.get("owned_by", "jetbrains-ai"),
        )
        for model in models_data.get("data", [])
    ]
    return ModelList(data=model_list)


async def openai_stream_adapter(
    api_stream_generator: AsyncGenerator[str, None],
    model_name: str,
    tools: Optional[List[Dict[str, Any]]],
) -> AsyncGenerator[str, None]:
    """将 JetBrains API 的流转换为 OpenAI 格式的 SSE"""
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    first_chunk_sent = False
    tool_id = 0

    try:
        async for line in api_stream_generator:
            if not line or line == "data: end":
                continue

            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    event_type = data.get("type")

                    if event_type == "Content":
                        content = data.get("content", "")
                        if not content:
                            continue

                        delta_payload = {}
                        if not first_chunk_sent:
                            delta_payload = {"role": "assistant", "content": content}
                            first_chunk_sent = True
                        else:
                            delta_payload = {"content": content}

                        stream_resp = StreamResponse(
                            id=stream_id,
                            model=model_name,
                            choices=[StreamChoice(delta=delta_payload)],
                        )
                        yield f"data: {stream_resp.json()}\n\n"

                    elif event_type == "FunctionCall":
                        func_name = data.get("name", None)
                        func_argu = data.get("content", None)
                        if func_name and tools:
                            for tool_id, tool in enumerate(tools):
                                if tool["name"] == func_name:
                                    break

                        delta_payload = {
                            "tool_calls": [
                                {
                                    "index": tool_id,
                                    "id": f"call_{uuid.uuid4().hex}",
                                    "function": {
                                        "arguments": func_argu,
                                        "name": func_name,
                                    },
                                    "type": "function" if func_name else None,
                                }
                            ]
                        }
                        stream_resp = StreamResponse(
                            id=stream_id,
                            model=model_name,
                            choices=[StreamChoice(delta=delta_payload)],
                        )
                        yield f"data: {stream_resp.json()}\n\n"

                    elif event_type == "FinishMetadata":
                        final_resp = StreamResponse(
                            id=stream_id,
                            model=model_name,
                            choices=[StreamChoice(delta={}, finish_reason="stop")],
                        )
                        yield f"data: {final_resp.json()}\n\n"
                        break
                except json.JSONDecodeError:
                    print(f"警告: 无法解析的 JSON 行: {line}")
                    continue

        yield "data: [DONE]\n\n"

    except Exception as e:
        print(f"流式适配器错误: {e}")
        error_resp = StreamResponse(
            id=stream_id,
            model=model_name,
            choices=[
                StreamChoice(
                    delta={"role": "assistant", "content": f"内部错误: {str(e)}"},
                    index=0,
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {error_resp.json()}\n\n"
        yield "data: [DONE]\n\n"


async def aggregate_stream_for_non_stream_response(
    openai_sse_stream: AsyncGenerator[str, None], model_name: str
) -> ChatCompletionResponse:
    """聚合流式响应为完整响应"""
    content_parts = []
    tool_calls_map = {}
    final_finish_reason = "stop"

    async for sse_line in openai_sse_stream:
        if sse_line.startswith("data: ") and sse_line.strip() != "data: [DONE]":
            try:
                data = json.loads(sse_line[6:].strip())
                if not data.get("choices"):
                    continue

                choice = data["choices"][0]
                delta = choice.get("delta", {})

                if choice.get("finish_reason"):
                    final_finish_reason = choice.get("finish_reason")

                if delta.get("content"):
                    content_parts.append(delta["content"])

                if "tool_calls" in delta:
                    for tc_chunk in delta["tool_calls"]:
                        idx = tc_chunk["index"]
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }

                        if tc_chunk.get("id"):
                            tool_calls_map[idx]["id"] = tc_chunk["id"]

                        func_chunk = tc_chunk.get("function", {})
                        if func_chunk.get("name"):
                            tool_calls_map[idx]["function"]["name"] = func_chunk["name"]
                        if func_chunk.get("arguments"):
                            tool_calls_map[idx]["function"]["arguments"] += func_chunk[
                                "arguments"
                            ]
            except json.JSONDecodeError:
                print(f"警告: 聚合时无法解析的 JSON 行: {sse_line}")

    final_tool_calls = []
    for k, v in sorted(tool_calls_map.items()):
        if "id" not in v:
            v["id"] = f"call_{uuid.uuid4().hex}"
        final_tool_calls.append(v)

    full_content = "".join(content_parts) or None

    if final_tool_calls:
        message = ChatMessage(
            role="assistant", content=full_content, tool_calls=final_tool_calls
        )
        final_finish_reason = "tool_calls"
    else:
        message = ChatMessage(role="assistant", content=full_content)

    return ChatCompletionResponse(
        model=model_name,
        choices=[
            ChatCompletionChoice(
                message=message,
                finish_reason=final_finish_reason,
            )
        ],
    )


def extract_text_content(content: Optional[Union[str, List[Dict[str, Any]]]]) -> str:
    """从消息内容中提取文本内容"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # 处理多模态消息格式，提取所有文本内容
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return " ".join(text_parts)
    return ""


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, _: None = Depends(authenticate_client)
):
    """创建聊天完成"""
    model_config = get_model_item(request.model)
    if not model_config:
        raise HTTPException(status_code=404, detail=f"模型 {request.model} 未找到")

    auth_token = await get_next_jetbrains_jwt()

    # 从历史消息中创建 tool_call_id 到 function_name 的映射
    tool_id_to_func_name_map = {}
    for m in request.messages:
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                if tc.get("id") and tc.get("function", {}).get("name"):
                    tool_id_to_func_name_map[tc["id"]] = tc["function"]["name"]

    # 将 OpenAI 格式的消息转换为 JetBrains 格式
    jetbrains_messages = []
    for msg in request.messages:
        # 提取文本内容，处理多模态消息格式
        text_content = extract_text_content(msg.content)

        if msg.role in ["user", "system"]:
            jetbrains_messages.append(
                {"type": f"{msg.role}_message", "content": text_content}
            )

        elif msg.role == "assistant":
            if msg.tool_calls:
                jetbrains_messages.append(
                    {
                        "type": "assistant_message",
                        "content": text_content,
                        "functionCall": {
                            "functionName": msg.tool_calls[0]["function"]["name"],
                            "content": msg.tool_calls[0]["function"]["arguments"],
                        },
                    }
                )
            else:
                jetbrains_messages.append(
                    {"type": "assistant_message", "content": text_content}
                )

        elif msg.role == "tool":
            function_name = tool_id_to_func_name_map.get(msg.tool_call_id)
            if function_name:
                jetbrains_messages.append(
                    {
                        "type": "function_message",
                        "content": text_content,
                        "functionName": function_name,
                    }
                )
            else:
                print(
                    f"警告: 无法为 tool_call_id {msg.tool_call_id} 找到对应的函数调用"
                )
        else:
            jetbrains_messages.append({"type": "user_message", "content": text_content})

    data = []
    tools = None
    if request.tools:
        data.append({"type": "json", "fqdn": "llm.parameters.functions"})
        tools = []
        for t in request.tools:
            tools.append(t["function"])
        data.append({"type": "json", "value": json.dumps(tools)})

    # 创建 API 请求的 payload
    payload = {
        "prompt": "ij.chat.request.new-chat-on-start",  # or other relevant prompt
        "profile": request.model,
        "chat": {"messages": jetbrains_messages},
        "parameters": {"data": data},
    }

    headers = {
        "User-Agent": "ktor-client",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "Accept-Charset": "UTF-8",
        "Cache-Control": "no-cache",
        "grazie-agent": '{"name":"aia:pycharm","version":"251.26094.80.13:251.26094.141"}',  # 可根据需要更新
        "grazie-authenticate-jwt": auth_token,
    }

    async def api_stream_generator():
        """一个包装 httpx 请求的异步生成器"""
        async with http_client.stream(
            "POST",
            "https://api.jetbrains.ai/user/v5/llm/chat/stream/v7",
            json=payload,
            headers=headers,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                yield line

    # 创建 OpenAI 格式的流
    openai_sse_stream = openai_stream_adapter(
        api_stream_generator(), request.model, tools or []
    )

    # 返回流式或非流式响应
    if request.stream:
        return StreamingResponse(openai_sse_stream, media_type="text/event-stream")
    else:
        return await aggregate_stream_for_non_stream_response(
            openai_sse_stream, request.model
        )


# 主程序入口
if __name__ == "__main__":
    import os

    # 创建示例配置文件（如果不存在）
    if not os.path.exists("client_api_keys.json"):
        with open("client_api_keys.json", "w", encoding="utf-8") as f:
            json.dump(["sk-your-custom-key-here"], f, indent=2)
        print("已创建示例 client_api_keys.json 文件")

    if not os.path.exists("jetbrainsai.json"):
        with open("jetbrainsai.json", "w", encoding="utf-8") as f:
            json.dump([{"jwt": "your-jwt-here"}], f, indent=2)
        print("已创建示例 jetbrainsai.json 文件")

    if not os.path.exists("models.json"):
        with open("models.json", "w", encoding="utf-8") as f:
            json.dump(["anthropic-claude-3.5-sonnet"], f, indent=2)
        print("已创建示例 models.json 文件")

    print("正在启动 JetBrains AI OpenAI Compatible API 服务器...")
    print("端点:")
    print("  GET  /v1/models")
    print("  POST /v1/chat/completions")
    print("\n在 Authorization header 中使用客户端 API 密钥 (Bearer sk-xxx)")

    uvicorn.run(app, host="0.0.0.0", port=8000)
