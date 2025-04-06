import json
import random
from pathlib import Path

from pydantic import BaseModel

from taxonomy_generator.corpus.reader import AICorpus, Paper
from taxonomy_generator.utils.llm import Chat, run_in_parallel
from taxonomy_generator.utils.parse_llm import parse_response_json
from taxonomy_generator.utils.prompting import fps

INIT_GET_TOPICS = """
Your task is to develop a taxonomy for categorizing a corpus of {field} related research papers. The full corpus has {corpus_len} papers, but to give some context, here are {sample_len} randomly chosen papers from the corpus:

---
{sample}
---

Specifically, your job is to develop a list of sub-topics under {field} to effectively categorize all the papers in this corpus. Your breakdown may have anywhere from 2 to 8 topics with each topic defined by a title and brief description.

After providing your breakdown, it will automatically be evaluated using LLMs and various metrics so that it can be iterated upon.

You are ultimately striving for the following attributes:
- Aim for MECE: Mutually Exclusive, Collectively Exaustive.
    - Mutually Exclusive: An LLM will categorize each paper, and for each paper it finds fitting multiple topics, the lower your score will be.
    - Collectively Exaustive: All papers should fit into at least one topic.
    - Optimize for these as best you can, but don't strive for perfection. Mutually exclusivity is likely impossible, and there are probably a few papers which shouldn't even be in the corpus.
- Use semantically meaningful categories. E.g., don't categorize by non-content attributes like publication date.
- Your breakdown should provide a clear mental model of {field} that is valuable to both newcomers and experienced researchers.
- Strive for topics that likely have existing survey or literature review papers. The evaluation system will reward topics for which it could find at least one associated overview/survey paper.

Please present your topics as a JSON array without any other text or explanation. Example format:

```json
[
    {{
        "title": "Clear and concise title",  # Make shorter I think
        "description": "~2 sentence description of the topic"
    }}
]
```
"""

GET_TOPICS = """
The evaluation ran successfully on your proposed breakdown. Here are the results:

# Sorting
A sample of {sort_sample_len} papers from the corpus were asked to be categorized by an LLM.

Of these, {easy_num} ({easy_perc}%) papers were reportedly easy to categorize.

For {no_place_num} ({no_place_perc}%) papers, the LLM couldn't find a suitable category. Examples:

---
{no_place_examples}
---

The LLM found overlap in {overlap_num} ({overlap_perc}%) papers (those which it decided had multiple applicable categories). Examples:

---
{overlap_examples}
---

Here are how many papers were sorted into each of your categories

---
{numbers_breakdown}
---

It is generally good to try and even out how many papers are in each category, but be cautious as this could just indicate there is less work in the topic but it is still important to have as a seperate category.

For additional context, here are {paper_topic_examples_num_per_topic} randomly selected papers for each topic that were categorized by the LLM:

---
{paper_topic_examples}
---

# Overview Papers
For each topic, we attempted to find at least one associated overview or literature review paper.

{overview_paper_results}

# Helpfulness Scores
In an attempt to guage *helpfulness*, 4 LLM were asked to generate a helpfullness score and provide feedback on your proposed breakdown. Here are the results:

---
{helpfulness_scores}
---

As this is LLM generated feedback, take it with disgression, and only incorporate feedback that makes sense.

# Final Score
From these metrics, here is the overall score that was calculated: {overall_score}

After analyzing and incorporating this feedback, please present your updated set of topics as a JSON array like before.
"""

SORT_PAPER = """
You are categorizing a paper for an {field} taxonomy.

PAPER:
---
{paper}
---

AVAILABLE CATEGORIES:
```json
{topics}
```

TASK: Identify which category/categories this paper belongs to. Respond with a JSON array of strings containing the title(s) of matching categories. If none fit, return an empty array. Add no other text or explanation.

Example responses:
["Category A"]
["Category A", "Category B"]
[]
"""

TOPICS_FEEDBACK = """
Given the following breakdown for the field of {field}

```json
{topics}
```

Please rate how helpful this breakdown is for you (1-5)
- Does it make sense
- Does it feel like a satisfying encapsulation of the
- Does it help provide a good


Please respond with XML in the following format:
<score>~#</score> <feedback>...</feedback>
"""

TOPICS_FEEDBACK_SYSTEM_PROMPTS = [
    None,
    "You are an enthusiast of {}",
    "You are an experienced and prolific {} researcher who is well renown in the field",
    "You are a newcomer to the field of {} and want to learn more",
]

fps(globals())

TREE_PATH = Path("data/tree.json")

FIELD = "AI safety"

corpus = AICorpus()


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


def save_tree(root: Topic):
    TREE_PATH.write_text(json.dumps(root.model_dump(), ensure_ascii=False))


def resolve_topic_papers(papers: list[Paper]):
    return [
        TopicPaper(
            title=p.title,
            arx=p.arxiv_id.split("v")[0],
            published=p.published,
            abstract=p.abstract,
        )
        for p in papers
    ]


def resolve_topics(response: str):
    return [Topic(**t) for t in parse_response_json(response, [])]


def format_topics_feedbacks(topics_feedbacks: list[TopicsFeedback]):
    return "\n\n".join(
        [
            f"System Prompt: {tf.system or 'You are a helpful assistant'}\nScore: {tf.score}\nFeedback: {tf.feedback}"
            for tf in topics_feedbacks
        ]
    )


def topics_json_str(topics: list[Topic]):
    return json.dumps(
        [{"title": t.title, "description": t.description} for t in topics],
        indent=2,
        ensure_ascii=False,
    )


def evaluate_topics(topics: list[Topic], sample_len: int, all_papers: list[TopicPaper]):
    # Sorting

    random.seed(1)
    sample = random.sample(all_papers, min(sample_len, len(all_papers)))
    topics_str = topics_json_str(topics)

    results = run_in_parallel(
        [
            SORT_PAPER.format(
                paper=corpus.get_pretty_paper(p), topics=topics_str, field=FIELD
            )
            for p in sample
        ],
        model="gemini-2.0-flash",
    )
    results = [
        {
            "paper": p,
            "topics": next(
                [t for t in topics if t.title in parse_response_json(r, [])]
            ),
        }
        for r, p in zip(results, sample)
    ]

    # Overview Papers

    # Helpfulness Scores

    # Final Score

    return EvalResult()


def main(
    init_sample_len: int = 150,
    sort_sample_len: int = 300,
    num_iterations: int = 5,
):
    topic = Topic(
        title=FIELD,
        description="...",
        papers=resolve_topic_papers(corpus.papers),
    )

    best: tuple[list[Topic], int] = (None, 0)
    chat = Chat()
    eval_result: EvalResult | None = None

    for _ in range(num_iterations):
        if eval_result:
            prompt = GET_TOPICS.format(
                field=FIELD,
                overall_score=eval_result.overall_score,
                helpfulness_scores=format_topics_feedbacks(
                    eval_result.topics_feedbacks
                ),
            )
        else:
            prompt = INIT_GET_TOPICS.format(
                field=FIELD,
                sample_len=f"{init_sample_len:,}",
                corpus_len=f"{len(topic.papers):,}",
                sample=corpus.get_pretty_sample(init_sample_len, seed=1),
            )

        topics = resolve_topics(chat.ask(prompt, use_thinking=True, verbose=True))

        eval_result = evaluate_topics(topics, sort_sample_len, topic.papers)

        if eval_result.overall_score > best[1]:
            best = (topics, eval_result.overall_score)

    topic.topics = best[0]

    save_tree(topic)


if __name__ == "__main__":
    main()
