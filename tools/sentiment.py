from typing import List, Dict, ClassVar, Set
from pydantic import BaseModel, Field
from unwrap_sdk import create_openai_completion, GPT5Deployment


class SimpleLexSentimentTool(BaseModel):
    """Sentiment analysis of a chunk of text from a cluster by LLM."""
    reasoning: str = Field(..., description="Reasoning process before calling tool")
    texts: List[str]

    async def execute(self) -> Dict:
        messages = [
            {'role': "system", "content": "You analyze sentiment. The following text belongs to a single cluster, analyze the sentiment and return a numerical value, where:\
             -1.0 is completely negative\
             0.0 is completely neutral\
             1.0 is completely positive. "},
            {'role': "user", "content": f"Summarize the sentiment of these sentences: \n{'\n'.join(self.texts)}."}
        ]
        resp = await create_openai_completion(messages, model=GPT5Deployment.GPT_5_MINI)
        try:
            score = float(resp.choices[0].message.content.strip())
        except Exception:
            score = 0.0
        return {"scores": [score]}

    # def execute(self) -> Dict:
    #     scores = []
    #     for t in self.texts:
    #         tl = (t or "").lower()
    #         s = sum(w in tl for w in self.POS) - sum(w in tl for w in self.NEG)
    #         scores.append(s)
    #     return {"scores": scores}