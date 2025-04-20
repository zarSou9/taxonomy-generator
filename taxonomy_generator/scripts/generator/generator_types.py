from pydantic import BaseModel

from taxonomy_generator.corpus.corpus_types import Paper


class SortTopic(BaseModel):
    title: str
    papers: list[str]


class TopicPaper(BaseModel):
    title: str
    arx: str
    published: str
    abstract: str


class Link(BaseModel):
    id: str


class Topic(BaseModel):
    title: str
    description: str
    papers: list[TopicPaper] = []
    topics: list["Topic"] = []
    links: list[Link] = []


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
    topic_papers: dict[str, list[TopicPaper]]
    overlap_topics_papers: dict[frozenset[str], list[TopicPaper]]
    not_placed: list[TopicPaper]
    single_papers: list[TopicPaper]
    overlap_papers: list[TopicPaper]
    sample_len: int
    overview_papers: dict[str, list[Paper]]
    invalid: bool = False
    invalid_reason: str = ""
