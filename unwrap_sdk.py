import asyncio, os, json, random
from typing import List, Dict, Optional, Any, Type
from pydantic import BaseModel
from openai import AsyncAzureOpenAI, pydantic_function_tool
from openai.types.chat import ChatCompletion
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

endpoint = "https://unwrap-hackathon-oct-20-resource.cognitiveservices.azure.com/"
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

# --- Concurrency + client caching ---
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

# --- Retry wrapper ---
async def _with_retries(funkshun, *, max_attempts=4, base_delay=0.6):
    attempt = 0
    while True:
        try:
            return await funkshun()
        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                raise
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)) + random.random() * 0.2)

# --- Chat completions ---
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
            "model": model.value,
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": reasoning_effort.value if isinstance(reasoning_effort, ReasoningEffort) else reasoning_effort,
        }
        if openai_tools:
            params["tools"] = openai_tools
            if tool_choice is not None:
                params["tool_choice"] = tool_choice
        return await _with_retries(lambda: client.chat.completions.create(**params))

# --- Embeddings ---
async def create_embeddings(inputs: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    # UNCOMMENT ONCE EMBEDDINGS ARE FIGURED OUT
    # client = get_client()
    # out: List[List[float]] = []
    # for i in range(0, len(inputs), 100):
    #     chunk = inputs[i:i+100]
    #     resp = await _with_retries(lambda: client.embeddings.create(model=model, input=chunk))
    #     out.extend([d.embedding for d in resp.data])
    # return out
    print("[DEBUG] Mocking embeddings because deployment is missing")
    return [[0.1 * (i+1) for _ in range(5)] for i in range(len(inputs))]
    client = get_client()
    batch_size = 100
    max_concurrency = 5
    sem = asyncio.Semaphore(max_concurrency)

    chunks = [inputs[i:i+batch_size] for i in range(0, len(inputs), batch_size)]

    async def concChunk(chunk: List[str]):
        async with sem: 
            return await _with_retries(lambda: client.embeddings.create(model=model, input=chunk))

    tasks = [
        #lambad chunk=chunk:
        concChunk(chunk)
        for chunk in chunks
    ]

    responses = await asyncio.gather(*tasks, return_exceptions= True)

    failures= [i for i,r in enumerate(responses) if isinstance(r, Exception)]
    
    for failIndex in failures:
        try:
            responses[failIndex] = await concChunk(chunks[failIndex])
        except Exception as e:
            print(f"Chunk {failIndex} failed, with {e}")
    
    out: List[List[float]] = []
    for resp in responses:
        if not isinstance(resp, Exception):
            out.extend([d.embedding for d in resp.data])
    return out

# --- Tool execution ---
def execute_tool_call(tool_call, available_tools: Dict[str, Type[BaseModel]]) -> Dict[str, Any]:
    name = tool_call.function.name
    if name not in available_tools:
        return {"ok": False, "error": f"Tool {name} not found"}
    try:
        args = json.loads(tool_call.function.arguments or "{}")
        inst = available_tools[name](**args)
        if hasattr(inst, "execute"):
            return {"ok": True, "tool": name, "args": args, "data": inst.execute()}
        return {"ok": False, "error": f"Tool {name} missing execute()"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

# --- Example Pydantic tool ---
class GetWeatherTool(BaseModel):
    location: str
    unit: str = "celsius"

    def execute(self) -> Dict[str, Any]:
        return {
            "location": self.location,
            "temperature": "22°C" if self.unit == "celsius" else "72°F",
            "condition": "sunny",
            "unit": self.unit,
        }

# --- Example usage ---
async def main():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Paris?"},
    ]
    # Chat
    chat_resp = await create_openai_completion(messages, tools=[GetWeatherTool], tool_choice="auto")
    print("Chat response:", chat_resp.choices[0].message.content)

    # Tool execution
    if chat_resp.choices[0].message.tool_calls:
        result = execute_tool_call(chat_resp.choices[0].message.tool_calls[0], {"GetWeatherTool": GetWeatherTool})
        print("Tool result:", result)

    # Embeddings
    embeddings = await create_embeddings(["Hello world", "Azure OpenAI is awesome"])
    print("Embeddings shape:", len(embeddings), "x", len(embeddings[0]))

if __name__ == "__main__":
    asyncio.run(main())
