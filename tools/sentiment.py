from typing import List, Dict, ClassVar, Set
from pydantic import BaseModel, Field
from unwrap_sdk import create_openai_completion, GPT5Deployment

def parse_output(tag: str, content: str) -> str:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start_idx = content.find(start_tag)
    end_idx = content.find(end_tag)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return content[start_idx + len(start_tag):end_idx].strip()
    return None

class Cluster_Summarize_and_Score(BaseModel):
    """LLM."""
    texts: List[str]
    original_prompt: str

    async def execute(self) -> Dict:
        texts_str = ''
        for t in self.texts:
            texts_str += '- ' + t + '\n'
        messages = [
            {'role': "system", "content": """You analyze analyze clusters of text segments which were gathered in response to a prompt and have similar embeddings. You will be provided a list of texts which belong to a single cluster, and you will score their relevancy to a given prompt, their sentiment, and give an overall summary of their content.
Rate their relevance score between 0 and 1, where 0 means completely irrelevant, 0.5 means moderately relevant, and 1 means completely relevant to the original prompt. Output just this number enclosed in <relevancy> and </relevancy> tags.
Rate their sentiment score between -1 and 1, where -1.0 is completely negative, 0.0 is completely neutral, and 1.0 is completely positive. Output just this number enclosed in <sentiment> and </sentiment> tags.
Finally, provide a concise summary of the main themes present in the texts. Output the summary between <summary> and </summary> tags.
For example, your output should look like this for a relevant and slightly positive cluster:
<relevancy>0.75</relevancy>
<sentiment>0.2</sentiment>
<summary>This cluster discusses...</summary>
"""},
            {'role': "user", "content": f"Prompt:\n{self.original_prompt}\n\nTexts: \n{texts_str}"}
        ]
        resp = await create_openai_completion(messages, model=GPT5Deployment.GPT_5_MINI)
        print(resp.choices[0].message.content)
        relevancy = parse_output("relevancy", resp.choices[0].message.content)
        if relevancy is None:
            relevancy = 0
        else:
            relevancy = float(relevancy)
        sentiment = parse_output("sentiment", resp.choices[0].message.content)
        if sentiment is None:
            sentiment = 0
        else:
            sentiment = float(sentiment)
        summary = parse_output("summary", resp.choices[0].message.content)
        if summary is None:
            summary = "Summary unavailable."

        return {"relevancy": relevancy, "sentiment": sentiment, "summary": summary}
