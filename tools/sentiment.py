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
            {'role': "system", "content": """
You are an analytical assistant that evaluates clusters of text segments gathered in response to a research prompt. Each cluster contains texts with similar embeddings.

For each cluster, your task is to:
1. Assess **relevance and informativeness** to the original prompt.
2. Assess **overall sentiment**.
3. Provide a concise **summary** of the cluster’s content.
4. Generate a short **topic phrase** (a few words) naming what the cluster is mainly about.

---

### Relevance Score  
Rate between 0 and 1, where:
- 0.0 → completely irrelevant to the prompt.  
- 0.5 → somewhat related but adds little insight or substance.  
- 1.0 → highly relevant and provides informative, insightful, or unique content about the prompt.  
Output this number inside `<relevancy>` and `</relevancy>` tags.

### Sentiment Score  
Rate between -1 and 1, where:
- -1.0 → completely negative  
- 0.0 → neutral  
- 1.0 → completely positive  
Output this number inside `<sentiment>` and `</sentiment>` tags.


### Summary  
Write a short paragraph summarizing the main ideas, themes, or perspectives expressed in the cluster.  
Output inside `<summary>` and `</summary>` tags.

### Topic Phrase  
Output a concise phrase (3–8 words) that best describes the cluster’s main subject or theme.  
Output inside `<topic>` and `</topic>` tags.

---

Your entire output must follow this structure exactly:
<relevancy>...</relevancy>
<sentiment>...</sentiment>
<summary>...</summary>
<topic>...</topic>
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
        topic = parse_output("topic", resp.choices[0].message.content)
        if summary is None:
            topic = "Topic preview unavailable."

        return {"relevancy": relevancy, "sentiment": sentiment, "summary": summary, "topic": topic}
