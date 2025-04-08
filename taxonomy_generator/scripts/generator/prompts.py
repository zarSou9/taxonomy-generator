from tabulate import tabulate

from taxonomy_generator.corpus.reader import AICorpus
from taxonomy_generator.scripts.generator.generator_types import (
    EvalResult,
    Topic,
    TopicsFeedback,
)
from taxonomy_generator.utils.prompting import fps, prompt
from taxonomy_generator.utils.utils import format_perc, random_sample

INIT_GET_TOPICS = """
Your task is to develop a taxonomy for categorizing a corpus of {field} related research papers. The full corpus has a total of {corpus_len} papers, but to give some context, here are {sample_len} randomly chosen papers from the corpus:

---
{sample}
---

Specifically, your job is to develop a list of sub-topics under {field} to effectively categorize all the papers in this corpus. Your breakdown may have anywhere from 2 to 8 topics with each topic defined by a title and brief description.

After providing your breakdown, it will automatically be evaluated using LLMs and various metrics so that it can be iterated upon.

You are ultimately striving for the following attributes:
- Aim for MECE: Mutually Exclusive, Collectively Exaustive.
    - Mutually Exclusive: An LLM will used to categorize each paper. Minimize the chance of it identifying more than one suitable topic for a given paper.
    - Collectively Exaustive: All papers should fit into at least one topic.
    - Optimize for these as best you can, but don't strive for perfection. Mutually exclusivity is likely impossible, and there are probably a few papers which shouldn't even be in the corpus.
- Use semantically meaningful categories. E.g., don't categorize by non-content attributes like publication date.
- Your breakdown should provide a clear mental model of {field} that is valuable to both newcomers and experienced researchers.
- Strive for topics that likely have existing survey or literature review papers. The evaluation system will reward topics for which it could find at least one associated overview/survey paper.

Please present your topics as a JSON array without any other text or explanation. Example format:

```json
[
    {{
        "title": "Clear and concise title",
        "description": "~2 sentence description of the topic"
    }}
]
```
"""

SORT_PAPER = """
You are categorizing a paper into a taxonomy for {field} related papers.

PAPER:
---
{paper}
---

AVAILABLE CATEGORIES:
```json
{topics}
```

Please identify which category/categories this paper belongs to. Respond with a JSON array of strings containing the title(s) of matching categories. If none fit, return an empty array. Add no other text or explanation.
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


@prompt
def get_not_placed_str(eval_result: EvalResult) -> str:
    not_placed_num = len(eval_result.not_placed)
    not_placed_examples = corpus.get_pretty_sample(
        random_sample(eval_result.not_placed, 4, 1)
    )
    perc = not_placed_num / eval_result.sample_len

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
For{" only" if perc < 0.02 else ""} {not_placed_num} ({format_perc(perc)}) papers, the LLM couldn't find a suitable category. Here are a couple of those papers for context:

<papers_not_placed>
{not_placed_examples}
</papers_not_placed>
"""


@prompt
def get_overlap_str(eval_result: EvalResult, duplicate_arxs: set[str]) -> str:
    overlap_num = len(duplicate_arxs)
    overlap_sorted = sorted(
        eval_result.overlap_papers.items(), key=lambda tp: len(tp[1]), reverse=True
    )

    stats: list[tuple[str, int, str]] = []
    examples: list[str] = []
    for topic_titles, papers in overlap_sorted:
        title = " | ".join(topic_titles)

        stats.append(
            (
                title,
                len(papers),
                format_perc(len(papers) / eval_result.sample_len, fill=True),
            )
        )

        if len(papers) / overlap_num > 0.09:
            sample_str = corpus.get_pretty_sample(random_sample(papers, 3, seed=1))
            examples.append(f"## {title}\n\n{sample_str}")

    examples_str = "\n\n".join(examples)

    if overlap_num == 0:
        return """
The LLM didn't classify any papers as fitting into more than one category. Your topics are very well-separated (maybe too separated?).
"""

    if overlap_num == 1:
        return """
The LLM only found one paper that was categorized into more than one category. Your topics are very well-separated (maybe even too separated?).
"""

    return f"""
The LLM found overlap in {overlap_num} ({format_perc(overlap_num / eval_result.sample_len)}) papers (those which it decided had multiple applicable categories).

Here's a table showing the combinations of topics these papers were sorted into, ordered by frequency (highest to lowest).

{tabulate(stats, headers=["Topics", "Num Papers", "Percent of Sample"], colalign=["left", "right", "right"])}

And here are a few examples from the top combinations:

<overlap_examples>
{examples_str}
</overlap_examples>

Remember, while reducing overlap is a core goal here, it's important that you don't sacrifice the quality of your taxonomy in sole pursuit of mutual exclusivity. For example, you might be tempted to create a new category that represents one of these topic combinations. Don't do this unless you have a very good reason to, as it would likely confuse readers and disrupt the consistent structure of your taxonomy.
"""


@prompt
def get_topic_papers_str(eval_result: EvalResult, single_arxs: set[str]) -> str:
    topic_papers_sorted = sorted(
        eval_result.topic_papers.items(), key=lambda tp: len(tp[1]), reverse=True
    )

    stats: list[tuple[str, int, str]] = []
    examples: list[str] = []
    for topic_title, papers in topic_papers_sorted:
        stats.append(
            (
                topic_title,
                len(papers),
                format_perc(len(papers) / eval_result.sample_len, fill=True),
            )
        )

        clean_papers = [p for p in papers if p.arx in single_arxs]
        papers = clean_papers if len(clean_papers) >= 3 else papers
        sample = random_sample(papers, 3, seed=1)
        pretty_sample = (
            corpus.get_pretty_sample(sample)
            if sample
            else "No papers were categorized into this topic"
        )
        examples.append(f"## {topic_title}\n\n{pretty_sample}")

    table_str = tabulate(
        stats,
        headers=["Topic", "Num Papers", "Percent of Sample"],
        colalign=["left", "right", "right"],
    )
    examples_str = "\n\n".join(examples)

    return f"""
This table shows how many papers were sorted into each topic, ordered by frequency:

{table_str}

For additional context, here are example papers from each topic:

<example_papers_by_topic>
{examples_str}
</example_papers_by_topic>
"""


def get_single_arxs(eval_result: EvalResult) -> tuple[set[str], set[str]]:
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

A random sample of {eval_result.sample_len} papers from the corpus were categorized by an LLM.

Of these, {len(single_arxs)} ({format_perc(len(single_arxs) / eval_result.sample_len)}) papers were cleanly categorized into one category.

{get_not_placed_str(eval_result)}

{get_overlap_str(eval_result, duplicate_arxs)}

{get_topic_papers_str(eval_result, single_arxs)}

For each topic, we attempted to find at least one associated overview or literature review paper. Here are the results:

{{overview_paper_results}}

In an attempt to gauge *helpfulness*, 4 LLMs were asked to provide feedback on how helpful they found the taxonomy. Each was given a different system prompt (to emulate different user groups), and asked to provide both open ended feedback, and an objective score from 1-5 (where 5 is best). Here are the results:

<helpfulness_feedback>
{format_topics_feedbacks(eval_result.topics_feedbacks)}
</helpfulness_feedback>

As this is LLM-generated feedback, use your discretion and only incorporate suggestions or consider feedback that is reasonable and relevant.

Depending on the results of this evaluation, you may want to combine, split, update, or add topics.

After developing your improved taxonomy, please present your new set of topics in the same format as before: as a JSON array of objects with "title" and "description" keys.
"""
