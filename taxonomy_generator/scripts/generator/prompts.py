from tabulate import tabulate

from taxonomy_generator.corpus.ai_corpus import AICorpus
from taxonomy_generator.scripts.generator.generator_types import (
    EvalResult,
    TopicsFeedback,
)
from taxonomy_generator.utils.prompting import fps, prompt
from taxonomy_generator.utils.utils import format_perc, random_sample

INIT_TOPICS = """
Your task is to develop a taxonomy for categorizing a corpus of {field} related research papers. The full corpus has a total of {corpus_len} papers, but to give some context, here are {sample_len} randomly chosen papers from the corpus:

---
{sample}
---

Specifically, your job is to develop a list of sub-topics under {field} to effectively categorize all the papers in this corpus. Your breakdown may have anywhere from 2 to 8 topics with each topic defined by a title and brief description.

Note: By default, there will already be a special category for papers that provide a broad overview or literature review of {field} as a whole (i.e., for papers like "{field_cap}: A Comprehensive Overview"). You don't need to consider these general overview papers for your taxonomy.

After providing your breakdown, it will automatically be evaluated using LLMs and various metrics so that it can be iterated upon.

You should strive for the following attributes:
- Aim for MECE: Mutually Exclusive, Collectively Exaustive.
    - Mutually Exclusive: An LLM will be used to categorize each paper. Minimize the chance of it identifying more than one suitable topic for a given paper.
    - Collectively Exaustive: All papers should fit into at least one topic.
    - Optimize for these as best you can, but don't strive for perfection. Mutual exclusivity is likely impossible, and there are probably a few papers which shouldn't even be in the corpus.
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
You are categorizing a paper into a taxonomy for {field} related research papers.

PAPER:
---
{paper}
---

AVAILABLE CATEGORIES:
```json
{topics}
```

If, however, this paper is a broad overview or survey of {field} as a whole, categorize it as "{field_cap} Overview/Survey" instead of the categories above.

Please identify which category/categories this paper belongs to. Respond with a JSON array of strings containing the title(s) of matching categories. If none fit, return an empty array. Add no other text or explanation.
"""

SORT_PAPER_SINGLE = """
You are categorizing a paper into a taxonomy for {field} related research papers.

PAPER:
---
{paper}
---

AVAILABLE CATEGORIES:
```json
{topics}
```

If, however, this paper is a broad overview or survey of {field} as a whole, categorize it as "{field_cap} Overview/Survey" instead of the categories above.

Please identify which single category this paper belongs to. Respond with only the title of the best matching category. If this is an overview or survey paper of {field} as a whole, respond with "{field_cap} Overview/Survey". If no categories fit, respond with "NONE APPLICABLE". Add no other text or explanation.
"""


TOPICS_FEEDBACK = """
Given the following topic breakdown for organizing {field} research papers:

```json
{topics}
```

Please provide your feedback on:
- Do you feel you understand each of the topics? Are the descriptions clear?
- Does it make sense conceptually? Is anything contradictory?
- Does it satisfyingly capture the breadth of {field}?
- How useful would you find this breakdown for navigating the research area?
- Do the categories seem sufficiently distinct to minimize overlap between topics?

Instead of offering suggestions for improvement, focus more on your experience: what you found helpful or unhelpful, clear or unclear, and why.

After providing feedback, please rate the overall usefulness of this taxonomy from 1-5 (where 5 is excellent).

Please provide your feedback and score in `<feedback>` and `<score>` XML tags. Example response format:
<feedback>Your feedback here.</feedback> <score>#</score>
"""


TOPICS_FEEDBACK_SYSTEM_PROMPTS: list[str | None] = [
    None,
    "You are an enthusiast of {} research",
    "You are an experienced and prolific {} researcher who is well renown in the field",
    "You are a newcomer to the field of {} and want to learn more",
]

fps(globals())

corpus = AICorpus(papers_override=[])


def format_topics_feedbacks(topics_feedbacks: list[TopicsFeedback]) -> str:
    return "\n\n".join(
        [
            f"System Prompt: {tf.system or 'No system prompt'}\nScore: {tf.score}\nFeedback:\n---\n{tf.feedback}\n---"
            for tf in topics_feedbacks
        ]
    )


@prompt
def get_not_placed_str(eval_result: EvalResult) -> str:
    not_placed_num = len(eval_result.not_placed)
    not_placed_examples = corpus.get_pretty_sample(
        random_sample(eval_result.not_placed, 4, seed=1)
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
def get_overlap_str(eval_result: EvalResult, first: bool) -> str:
    overlap_num = len(eval_result.overlap_papers)
    overlap_sorted = sorted(
        eval_result.overlap_topics_papers.items(),
        key=lambda tp: len(tp[1]),
        reverse=True,
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

    examples_str = "\n\n".join(examples[:4])

    if overlap_num == 0:
        return """
The LLM didn't classify any papers as fitting into more than one category. Your topics are very well-separated (maybe too separated?).
"""

    if overlap_num == 1:
        return """
The LLM only found one paper that was categorized into more than one category. Your topics are very well-separated (maybe even too separated?).
"""

    return f"""
The LLM found overlap in {overlap_num} ({format_perc(overlap_num / eval_result.sample_len)}) papers{" (those which it decided had multiple applicable categories)" if first else ""}.

{"Here's a table showing" if first else "Here are"} the combinations of topics these papers were sorted into{", ordered by frequency (highest to lowest)." if first else ":"}

{tabulate(stats, headers=["Topics", "Num Papers", "Percent of Sample"], colalign=["left", "right", "right"])}

And here are a few examples from the top combinations:

<overlap_examples>
{examples_str}
</overlap_examples>
"""


@prompt
def get_overview_results_str(eval_result: EvalResult, first: bool):
    if all(eval_result.overview_papers.values()) and not first:
        return "At least one overview paper was found for all topics."

    return f"""
{"For each topic, we attempted to find at least one associated overview or literature review paper. Here are the results:" if first else "Here are the overview paper results:"}

{tabulate(((title, "YES" if papers else "NO") for title, papers in eval_result.overview_papers.items()), headers=["Topic", "Found Overview Paper"])}
"""


@prompt
def get_topic_papers_str(eval_result: EvalResult, first: bool) -> str:
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

        clean_papers = [
            p for p in papers if p.arx in (p.arx for p in eval_result.single_papers)
        ]
        papers = clean_papers if len(clean_papers) >= 3 else papers
        sample = random_sample(papers, 2, seed=1)
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
    examples_message = f"""
For additional context, here are a couple example papers from each topic:

<example_papers_by_topic>
{examples_str}
</example_papers_by_topic>
"""

    return f"""
{"This table shows how many papers were sorted into each topic, ordered by frequency" if first else "Here are how many papers were sorted into each topic"}:

{table_str}

{examples_message if first else ""}
"""


@prompt
def get_iter_topics_prompt(eval_result: EvalResult, first: bool) -> str:
    if first:
        iterative_message = "Depending on the results of this evaluation, you may decide to combine, split, update, or add topics. As this is an iterative process, you are encouraged to experiment with different approaches - try taxonomies of different sizes (smaller with 2-3 topics, medium with 4-6 topics, or larger with 7-8 topics) or alternative ways of conceptualizing the field."
    else:
        iterative_message = "This is an iterative process and you have many attempts to test out different taxonomies. Take advantage of this. Experiment with different sized taxonomies or different ways of breaking down the field."

    return f"""
The evaluation script ran successfully on your {"proposed breakdown" if first else "latest taxonomy"}. Here are the results:

A random sample of {eval_result.sample_len} papers from the corpus were categorized by an LLM.

Of these, {len(eval_result.single_papers)} ({format_perc(len(eval_result.single_papers) / eval_result.sample_len)}) papers were cleanly categorized into one category.

{get_not_placed_str(eval_result)}

{get_overlap_str(eval_result, first)}

{get_topic_papers_str(eval_result, first)}

{get_overview_results_str(eval_result, first)}

{f"In an attempt to gauge *helpfulness*, {len(eval_result.topics_feedbacks)} LLMs were asked to provide feedback on how helpful or useful they found the taxonomy. Each was given a different system prompt (to emulate different user groups), and asked to provide both open ended feedback, and an objective score from 1-5 (where 5 is excellent). Here are the results:" if first else "Here are the LLM-generated feedback results:"}

<helpfulness_feedback>
{format_topics_feedbacks(eval_result.topics_feedbacks)}
</helpfulness_feedback>

{"As this is LLM-generated feedback," if first else "Remember to"} use your discretion and only incorporate suggestions or consider feedback that is reasonable and relevant.

{f"These metrics have been combined to produce an overall score of {eval_result.overall_score:.2f} for this taxonomy." if first else f"The overall score for this taxonomy comes out to {eval_result.overall_score:.2f}"}

{iterative_message}

{"Please present" if first else "Output"} your new set of topics in the same format as before: as a JSON array of objects with "title" and "description" keys. The titles should be clear and concise, and the descriptions around 2 sentences.
"""
