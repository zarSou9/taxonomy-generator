from typing import Literal

from pydantic import BaseModel


class Summary(BaseModel):
    text: str
    type: Literal["abstract", "ai_summary"] = "abstract"


class Paper(BaseModel):
    id: str
    title: str
    published: str
    summary: Summary
    source: Literal["arxiv", "alignmentforum", "lesswrong"] = "arxiv"
