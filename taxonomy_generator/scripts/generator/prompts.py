from tabulate import tabulate

from taxonomy_generator.corpus.reader import AICorpus
from taxonomy_generator.scripts.generator.generator_types import (
    EvalResult,
    Topic,
    TopicPaper,
    TopicsFeedback,
)
from taxonomy_generator.utils.prompting import fps, prompt
from taxonomy_generator.utils.utils import format_perc, random_sample

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
        "title": "Clear and concise title",  #! Make shorter I think
        "description": "~2 sentence description of the topic"
    }}
]
```
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
<score>#</score> <feedback>...</feedback>
"""

TOPICS_FEEDBACK_SYSTEM_PROMPTS = [
    None,
    "You are an enthusiast of {}",
    "You are an experienced and prolific {} researcher who is well renown in the field",
    "You are a newcomer to the field of {} and want to learn more",
]

fps(globals())

corpus = AICorpus(papers_override=[])


def format_topics_feedbacks(topics_feedbacks: list[TopicsFeedback]) -> str:
    return "\n\n".join(
        [
            f"System Prompt: {tf.system or 'You are a helpful assistant'}\nScore: {tf.score}\nFeedback: {tf.feedback}"
            for tf in topics_feedbacks
        ]
    )


def get_not_placed_str(eval_result: EvalResult) -> str:
    not_placed_num = len(eval_result.not_placed)
    not_placed_examples = corpus.get_pretty_sample(eval_result.not_placed)

    if not_placed_num == 0:
        return "The sorting LLM marked all papers as fitting into at least one category - nice!"

    if not_placed_num == 1:
        return f"""
The LLM marked only one paper as not fitting into any of the topics - nice! This was that paper:

<paper_not_placed>
{not_placed_examples}
</paper_not_placed>
"""

    return f"""
For {not_placed_num} ({format_perc(not_placed_num / eval_result.sample_len)}) papers, the LLM couldn't find a suitable category. Here are a couple of those papers:

<example_papers_not_placed>
{not_placed_examples}
</example_papers_not_placed>
"""


def get_overlap_str(eval_result: EvalResult, duplicate_arxs: set[str]) -> str:
    overlap_num = len(duplicate_arxs)
    overlap_stats: list[tuple[str, int, str]] = []
    examples: list[str] = []

    overlap_sorted = sorted(
        eval_result.overlap_papers.items(), key=lambda tp: len(tp[1]), reverse=True
    )

    for topic_titles, papers in overlap_sorted:
        title = " | ".join(topic_titles)

        overlap_stats.append(
            (
                title,
                len(papers),
                format_perc(len(papers) / eval_result.sample_len, fill=True),
            )
        )

        sample_str = corpus.get_pretty_sample(random_sample(papers, 3, seed=1))
        examples.append(f"## {title}\n\n{sample_str}")

    examples_str = "\n\n".join(examples)

    if overlap_num == 0:
        return """
The LLM didn't classify any papers as fitting into more than one category. Your topics are very well-separated.
"""

    if overlap_num == 1:
        return """
The LLM only found one paper that was categorized into more than one category. Your topics are very well-separated.
"""

    return f"""
The LLM found overlap in {overlap_num} ({format_perc(overlap_num / eval_result.sample_len)}) papers (those which it decided had multiple applicable categories).

Here's a table showing the topic combinations these papers were sorted into, ordered by frequency (highest to lowest).

{tabulate(overlap_stats, headers=["Topics", "Num Papers", "Percent of Sample"], colalign=["left", "right", "right"])}

And here are some examples of papers sorted into these combinations:

<overlap_examples_by_topic_combinations>
{examples_str}
</overlap_examples_by_topic_combinations>
"""


def get_topic_paper_table(eval_result: EvalResult) -> str:
    topic_papers_sorted = sorted(
        eval_result.topic_papers.items(), key=lambda tp: len(tp[1]), reverse=True
    )

    numbers_breakdown = []
    for topic_title, papers in topic_papers_sorted:
        numbers_breakdown.append(
            (
                topic_title,
                len(papers),
                format_perc(len(papers) / eval_result.sample_len, fill=True),
            )
        )

    return tabulate(
        numbers_breakdown,
        headers=["Topic", "Num Papers", "Percent of Sample"],
        colalign=["left", "right", "right"],
    )


def get_topic_paper_examples(topic_papers: dict[str, list[TopicPaper]]) -> str:
    topic_paper_examples = []
    for topic_title, papers in topic_papers.items():
        sample = random_sample(papers, 3, seed=1)
        pretty_sample = (
            corpus.get_pretty_sample(sample)
            if sample
            else "No papers were categorized into this topic"
        )
        topic_paper_examples.append(f"## {topic_title}\n\n{pretty_sample}")
    return "\n\n".join(topic_paper_examples)


def get_single_arxs(eval_result: EvalResult) -> list[str]:
    single_arxs: set[str] = set()
    duplicate_arxs: set[str] = set()
    for _, papers in eval_result.topic_papers.items():
        for paper in papers:
            if paper.arx in single_arxs:
                duplicate_arxs.add(paper.arx)
            else:
                single_arxs.add(paper.arx)

    for dup_arx in duplicate_arxs:
        single_arxs.remove(dup_arx)

    return single_arxs, duplicate_arxs


@prompt
def resolve_get_topics_prompt(eval_result: EvalResult, topics: list[Topic]) -> str:
    single_arxs, duplicate_arxs = get_single_arxs(eval_result)

    return f"""
The evaluation script ran successfully on your proposed breakdown. Here are the results:

#! Sorting
A random sample of {eval_result.sample_len} papers from the corpus were asked to be categorized by an LLM.

Of these, {len(single_arxs)} ({format_perc(len(single_arxs) / eval_result.sample_len)}) papers were cleanly categorized into one category.
{get_not_placed_str(eval_result)}{get_overlap_str(eval_result, duplicate_arxs)}
This table shows many papers were sorted into each topic, ordered by frequency:

{get_topic_paper_table(eval_result)}

It is generally good to try and even out how many papers are in each category, but be cautious as this could just indicate there is less work in the topic but it is still important to have as a seperate category.

For additional context, here are a some randomly chosen papers from the sample, organized by the topic they were sorted into:

<example_papers_by_topic>
{get_topic_paper_examples(eval_result.topic_papers)}
</example_papers_by_topic>

#! Overview Papers
For each topic, we attempted to find at least one associated overview or literature review paper.

{{overview_paper_results}}

#! Helpfulness Scores
In an attempt to gauge *helpfulness*, 4 LLMs were asked to provide feedback on how helpful they found the taxonomy. Each was given a different system prompt (to emulate different user groups), and were asked to provide both open ended feedback, and an objective score from 1-5 (where 5 is best). Here are the results:

<helpfulness_feedback>
{format_topics_feedbacks(eval_result.topics_feedbacks)}
</helpfulness_feedback>

As this is LLM-generated feedback, use your discretion and only incorporate suggestions or consider feedback that is reasonable and relevant.

#! Final Score
From these metrics, here is the overall score that was calculated: {eval_result.overall_score}

Depending on the results of this evaluation, you may want to combine, split, update, or add topics.

After developing your improved taxonomy, please present your new set of topics in the same format as before: as a JSON array of objects with "title" and "description" keys.
"""
