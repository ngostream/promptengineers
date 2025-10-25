import asyncio, os, json, random
from typing import List, Dict, Optional, Any, Type
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import AsyncAzureOpenAI, pydantic_function_tool
from openai.types.chat import ChatCompletion
from enum import Enum

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
subscription_key = os.getenv("SUBSCRIPTION_KEY")

class GPT5Deployment(str, Enum):
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5 = "gpt-5"

class ReasoningEffort(str, Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

_openai_semaphore = asyncio.Semaphore(20)
_client: Optional[AsyncAzureOpenAI] = None

def get_client() -> AsyncAzureOpenAI:
    global _client
    if _client is None:
        _client = AsyncAzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=endpoint,
            api_key=subscription_key,
            timeout=60,
        )
    return _client

async def _with_retries(coro_factory, *, max_attempts=4, base_delay=0.6):
    attempt = 0
    while True:
        try:
            return await coro_factory()
        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                raise
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)) + random.random() * 0.2)

async def create_openai_completion(
    messages: List[Dict[str, Any]],
    model: GPT5Deployment = GPT5Deployment.GPT_5_MINI,
    reasoning_effort: ReasoningEffort = ReasoningEffort.LOW,
    max_completion_tokens: int = 2048,
    tools: Optional[List[Type[BaseModel]]] = None,
    tool_choice: Optional[str | Dict[str, Any]] = None,
) -> ChatCompletion:
    async with _openai_semaphore:
        client = get_client()
        openai_tools = [pydantic_function_tool(t) for t in tools] if tools else None
        params: Dict[str, Any] = {
            "messages": messages,
            "model": model,
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": reasoning_effort,
        }
        if openai_tools:
            params["tools"] = openai_tools
            if tool_choice is not None:
                params["tool_choice"] = tool_choice
        return await _with_retries(lambda: client.chat.completions.create(**params))

async def create_embeddings(inputs: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    client = get_client()
    out: List[List[float]] = []
    for i in range(0, len(inputs), 100):
        chunk = inputs[i:i+100]
        resp = await _with_retries(lambda: client.embeddings.create(model=model, input=chunk))
        out.extend([d.embedding for d in resp.data])
    return out

def execute_tool_call(tool_call, available_tools: Dict[str, type[BaseModel]]) -> Dict[str, Any]:
    name = tool_call.function.name
    if name not in available_tools:
        return {"ok": False, "error": f"Tool {name} not found"}
    try:
        args = json.loads(tool_call.function.arguments or "{}")
        tool_instance = available_tools[name](**args)
        if hasattr(tool_instance, "execute"):
            data = tool_instance.execute()
            return {"ok": True, "tool": name, "args": args, "data": data}
        return {"ok": False, "error": f"Tool {name} missing execute()"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
