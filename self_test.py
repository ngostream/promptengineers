import asyncio
from unwrap_sdk import create_openai_completion, create_embeddings, GPT5Deployment

async def main():
    r = await create_openai_completion([
        {"role":"user","content":"Say ok"}
    ], model=GPT5Deployment.GPT_5_MINI)
    print("Chat OK:", r.choices[0].message.content)

    e = await create_embeddings(["hello world"]) 
    print("Embeddings OK: dim=", len(e[0]))

if __name__ == "__main__":
    asyncio.run(main())
