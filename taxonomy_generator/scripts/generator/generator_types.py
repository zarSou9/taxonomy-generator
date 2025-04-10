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


class Topic(BaseModel):
    title: str
    description: str
    papers: list[TopicPaper] = []
    topics: list["Topic"] = []


class TopicsFeedback(BaseModel):
    score: int
    feedback: str
    system: str | None


class EvalResult(BaseModel):
    overall_score: float
    topics_feedbacks: list[TopicsFeedback]
    topic_papers: dict[str, list[TopicPaper]]
    overlap_topics_papers: dict[frozenset[str], list[TopicPaper]]
    not_placed: list[TopicPaper]
    single_papers: list[TopicPaper]
    overlap_papers: list[TopicPaper]
    sample_len: int
    overview_papers: dict[str, list[Paper]]
