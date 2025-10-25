from typing import List, Dict, ClassVar, Set
from pydantic import BaseModel

class SimpleLexSentimentTool(BaseModel):
    """Very simple sentiment using a tiny lexicon (hackathon-fast)."""
    texts: List[str]

    POS: ClassVar[Set[str]] = {"great","good","love","amazing","win","fast","improve","growth","rising","boost"}
    NEG: ClassVar[Set[str]] = {"bad","worse","hate","terrible","issue","bug","scam","falling","drop","risk"}

    def execute(self) -> Dict:
        scores = []
        for t in self.texts:
            tl = (t or "").lower()
            s = sum(w in tl for w in self.POS) - sum(w in tl for w in self.NEG)
            scores.append(s)
        return {"scores": scores}
