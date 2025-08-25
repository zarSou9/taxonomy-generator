from pydantic import BaseModel

from taxonomy_generator.models.corpus import Paper


class Link(BaseModel):
    id: str


class Topic(BaseModel):
    title: str
    description: str
    papers: list[Paper] = []
    topics: list["Topic"] = []
    links: list[Link] = []
    scores: list[float] = []
    final_score: float | None = None


class TopicsFeedback(BaseModel):
    score: int
    feedback: str
    system: str | None


class EvalScores(BaseModel):
    feedback_score: float
    topics_overview_score: float | None
    not_placed_score: float
    deviation_score: float
    single_score: float


class EvalResult(BaseModel):
    all_scores: EvalScores
    overall_score: float
    topics_feedbacks: list[TopicsFeedback]
    topic_papers: dict[str, list[Paper]]
    overlap_topics_papers: dict[frozenset[str], list[Paper]]
    not_placed: list[Paper]
    single_papers: list[Paper]
    overlap_papers: list[Paper]
    sample_len: int
    overview_papers: dict[str, list[Paper] | None]
    invalid: bool = False
    invalid_reason: str = ""
