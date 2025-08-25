from tabulate import tabulate

from taxonomy_generator.corpus.corpus_instance import corpus
from taxonomy_generator.models.corpus import Paper
from taxonomy_generator.models.generator import (
    EvalResult,
    Topic,
    TopicsFeedback,
)
from taxonomy_generator.scripts.generator.utils import (
    list_titles,
    topic_breadcrumbs,
    topics_to_json,
)
from taxonomy_generator.utils.prompting import prompt
from taxonomy_generator.utils.utils import format_perc, random_sample, safe_lower


@prompt
def get_init_topics_prompt(
    topic: Topic,
    sample_len: int,
    sample_seed: int | None,
    overrview_checks: bool,
    topics_len_bounds: tuple[int, int],
    parents: list[Topic] = [],
) -> str:
    if not parents:
        field = safe_lower(topic.title)
        title = field

        start = f"""
Your task is to develop a taxonomy for categorizing a corpus of {field} related research papers. The full corpus has a total of {len(topic.papers):,} papers, but to give some context, here are {sample_len:,} randomly chosen papers from the corpus:

<corpus_sample>
{corpus.get_pretty_sample(random_sample(topic.papers, sample_len, sample_seed))}
</corpus_sample>

Specifically, your job is to develop a list of sub-topics under {field} to effectively categorize all the papers in this corpus."""
    else:
        field = safe_lower(parents[0].title)
        title = topic.title
        use_sample = sample_len / len(topic.papers) < 0.8

        parents_str = ""
        if len(parents) > 1:
            additional_breakdowns = ""
            if len(parents) > 2:
                next_breakdown_strs = [
                    "And then {} has been broken down as follows:",
                    *(
                        ["Now zooming in on {}, here is its breakdown:"]
                        if len(parents[2:]) >= 3
                        else []
                    ),
                    "And here's the breakdown for {}:",
                ]
                for i, parent in enumerate(parents[2:]):
                    additional_breakdowns += f"""
{(next_breakdown_strs[i] if i < len(next_breakdown_strs) else next_breakdown_strs[-1]).format(parent.title)}

{list_titles(parent.topics)}
"""

            parents_str = f"""at {topic_breadcrumbs(topic, parents[1:])}. So for additional context, here are all the categories under {parents[1].title}:

{list_titles(parents[1].topics)}

{additional_breakdowns}

Now, your category of focus, {topic.title}, is specifically defined as "{topic.description}" The category"""

        start = f"""
You are developing a hierarchical taxonomy for organizing a corpus of {field} related research papers. You've already developed the root breakdown (of {field}). Here's the set of categories comprising this breakdown (titles only):

{list_titles(parents[0].topics)}

The category you're currently focused on breaking down further is {parents_str if parents_str else f'{topic.title}, which is defined as "{topic.description}" This category'} currently has {len(topic.papers):,} papers sorted into it. {f"The following is a sample of {sample_len:,} papers from the full list:" if use_sample else "Here are those papers:"}

<papers{"_sample" if use_sample else ""}>
{corpus.get_pretty_sample(random_sample(topic.papers, sample_len, sample_seed) if use_sample else topic.papers)}
</papers{"_sample" if use_sample else ""}>

Your task is to develop a list of sub-categories/sub-topics to effectively categorize all papers in the {title} category."""

    return f"""
{start} Your breakdown may have anywhere from {topics_len_bounds[0]} to {topics_len_bounds[1]} topics with each topic defined by a title and brief description.

Note: By default, there will already be a special category for papers that provide a broad overview or literature review of {title} as a whole. Thus, you don't need to consider these general overview papers for your taxonomy.

After providing your breakdown, it will automatically be evaluated using LLMs and various metrics so that it can be iterated upon.

You should strive for the following attributes:
- Aim for MECE: Mutually Exclusive, Collectively Exaustive.
    - Mutually Exclusive: An LLM will be used to categorize each paper. Minimize the chance of it identifying more than one suitable topic for a given paper.
    - Collectively Exaustive: All papers should fit into at least one topic.
    - Optimize for these as best you can, but don't strive for perfection. Mutual exclusivity is likely impossible, and there {f"might be a couple papers which shouldn't even be under {title}" if parents else "are probably a few papers which shouldn't even be in the corpus"}.
- Use semantically meaningful categories. E.g., don't categorize by non-content attributes like publication date.
- Your breakdown should provide a clear mental model of {title} that is valuable to both newcomers and experienced researchers.
{f"- Strive for topics that likely have existing overview or literature review papers{' (if possible)' if parents else ''}. The evaluation system will reward topics for which it could find at least one associated overview/survey paper." if overrview_checks else ""}

Please present your topics as a JSON array of objects with "title" and "description" keys, without any other text or explanation. The descriptions should provide complete clarity on what the topic encompasses while remaining concise.

Example response format:

```json
[
    {{
        "title": "Clear and concise title",
        "description": "~2 sentence description of the topic"
    }}
]
```
"""


@prompt
def get_sort_prompt(
    topic: Topic,
    paper: Paper,
    topics: list[Topic],
    parents: list[Topic] = [],
    multiple: bool = False,
) -> str:
    root_topic = parents[0] if parents else topic

    field = safe_lower(root_topic.title)
    title = topic.title if parents else field

    start = f"""
You are categorizing a paper into a{" hierarchical" if parents else ""} taxonomy for{" organizing" if parents else ""} {field} related research papers.

{f'The paper is specifically under the topic {topic.title}{f", which can be found at {topic_breadcrumbs(topic, parents)}" if len(parents) > 1 else ""}. {topic.title} is defined as "{topic.description}" This topic has been broken down into the included list of categories, which you will use to categorize the paper further.' if parents else ""}

PAPER:
---
{corpus.get_pretty_paper(paper)}
---

AVAILABLE CATEGORIES:
```json
{topics_to_json(topics)}
```

If, however, this paper is {"an" if parents else "a broad"} overview or survey of{" specifically" if parents else ""} {title} as a whole, categorize it as "{topic.title} Overview/Survey" instead of the categories above.
"""

    if multiple:
        return f"""
{start}

Please identify which category/categories this paper belongs to. Respond with a JSON array of strings containing the title(s) of the matching category or categories. If none fit, return an empty array. Add no other text or explanation.
"""
    return f"""
{start}

Please identify which single category this paper belongs to. Respond with only the title of the best matching category. If no categories fit, respond with "NONE APPLICABLE". Add no other text or explanation.
"""


@prompt
def get_topics_feedback_prompt(topics: list[Topic], parents: list[Topic] = []) -> str:
    field = safe_lower(parents[0].title)

    output_format = """
Please provide your feedback and score in `<feedback>` and `<score>` XML tags. Example response format:
<feedback>Your feedback here.</feedback> <score>#</score>
"""

    if len(parents) == 1:
        return f"""
Given the following topic breakdown for organizing {field} research papers:

```json
{topics_to_json(topics)}
```

Please provide your feedback on:
- Do you feel you understand each of the topics? Are the descriptions clear?
- Does it make sense conceptually? Is anything contradictory?
- Does it satisfyingly capture the breadth of {field}?
- How useful would you find this breakdown for navigating the research area?
- Do the categories seem sufficiently distinct to minimize overlap between topics?

Instead of offering suggestions for improvement, focus more on your experience: what you found helpful or unhelpful, clear or unclear, and why.

After providing feedback, please rate the overall usefulness of this taxonomy from 1-5 (where 5 is excellent).

{output_format}
"""
    return f"""
You are reviewing part of a hierarchical taxonomy for organizing research papers related to {field}. Specifically you're reviewing a proposed breakdown of {f"{parents[-1].title}, which is directly under {field}" if len(parents) == 2 else f"the {parents[-1].title} topic, which can be found at {topic_breadcrumbs(parents[-1], parents[:-1])}"} in the taxonomy. Here's the proposed breakdown:

```json
{topics_to_json(topics)}
```

Please provide your feedback on:
- Do you feel you understand each of the categories? Are the descriptions clear?
- Does it make sense conceptually? Is anything contradictory?
- Does it satisfyingly capture the kinds of papers you'd expect to find under {parents[-1].title}?
- Do the categories seem sufficiently distinct with minimal overlap?

Instead of offering suggestions for improvement, focus more on your experience: what you found helpful or unhelpful, clear or unclear, and why.

After providing feedback, please rate the overall usefulness of this breakdown from 1-5 (where 5 is excellent).

{output_format}
"""


def get_topics_feedback_system_prompts(field: str) -> list[str | None]:
    return [
        None,
        f"You are an enthusiast of {field} research",
        f"You are an experienced and prolific {field} researcher who is well renown in the field",
        f"You are a newcomer to the field of {field} and want to learn more",
    ]


def format_topics_feedbacks(topics_feedbacks: list[TopicsFeedback]) -> str:
    return "\n\n".join(
        [
            f"System Prompt: {tf.system}\nScore: {tf.score}\nFeedback:\n---\n{tf.feedback}\n---"
            for tf in topics_feedbacks
            if tf.system
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
        return f"""
The LLM only found one paper that was categorized into more than one category. Your topics are very well-separated{" (maybe even too separated?)" if overlap_num / eval_result.sample_len < 0.015 else ""}.
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

{tabulate(((title, "SEARCH ERROR" if papers is None else "YES" if papers else "NO") for title, papers in eval_result.overview_papers.items()), headers=["Topic", "Found Overview Paper"])}
"""


@prompt
def get_topic_papers_str(eval_result: EvalResult, first: bool) -> str:
    topic_papers_sorted = sorted(
        eval_result.topic_papers.items(),
        key=lambda tp: len(tp[1]),
        reverse=True,
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
            p for p in papers if p.id in (p.id for p in eval_result.single_papers)
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
def get_iter_topics_prompt(
    eval_result: EvalResult,
    first: bool,
    topic: Topic,
    depth: int,
    topics_len_bounds: tuple[int, int],
    no_overviews: bool = False,
) -> str:
    title = "the field" if depth == 0 else topic.title
    if first:
        iterative_message = f"Depending on the results of this evaluation, you may decide to combine, split, update, or add topics. As this is an iterative process, you are encouraged to experiment with different approaches - try taxonomies of different sizes (anywhere from {topics_len_bounds[0]} to {topics_len_bounds[1]} topics) or alternative ways of conceptualizing {title}."
    else:
        iterative_message = f"This is an iterative process and you have many attempts to test out different taxonomies. Take advantage of this. Experiment with different sized taxonomies or different ways of breaking down {title}."

    return f"""
The evaluation script ran successfully on your {"proposed breakdown" if first else "latest taxonomy"}. Here are the results:

{f"A random sample of {eval_result.sample_len:,} papers from the {'corpus' if depth == 0 else 'full list'}" if eval_result.sample_len < len(topic.papers) else f"All {eval_result.sample_len:,} papers"} were categorized by an LLM.

Of these, {len(eval_result.single_papers)} ({format_perc(len(eval_result.single_papers) / eval_result.sample_len)}) papers were cleanly categorized into one category.

{get_not_placed_str(eval_result)}

{get_overlap_str(eval_result, first)}

{get_topic_papers_str(eval_result, first)}

{"" if no_overviews else get_overview_results_str(eval_result, first)}

{f"In an attempt to gauge *helpfulness*, {len([tf for tf in eval_result.topics_feedbacks if tf.system])} LLMs were asked to provide feedback on how helpful or useful they found the taxonomy. Each was given a different system prompt (to emulate different user groups), and asked to provide both open ended feedback, and an objective score from 1-5 (where 5 is excellent). Here are the results:" if first else "Here are the LLM-generated feedback results:"}

<helpfulness_feedback>
{format_topics_feedbacks(eval_result.topics_feedbacks)}
</helpfulness_feedback>

{"As this is LLM-generated feedback," if first else "Remember to"} use your discretion and only incorporate suggestions or consider feedback that is reasonable and relevant.

{f"These metrics have been combined to produce an overall score of {eval_result.overall_score:.2f} for this taxonomy." if first else f"The overall score for this taxonomy comes out to {eval_result.overall_score:.2f}"}

{iterative_message}

{"Please present" if first else "Output"} your new set of topics in the same format as before: as a JSON array of objects with "title" and "description" keys. The titles should be clear and concise, and the descriptions around 2 sentences.
"""


@prompt
def get_order_papers_prompt(
    topic: Topic,
    papers: list[Paper],
    root: Topic,
    parents: list[Topic] = [],
) -> str:
    """Generate a prompt to order all papers in a topic by relevance."""
    field = safe_lower(root.title)

    # Determine if this is a dead-end topic (no subtopics)
    is_dead_end = len(topic.topics) == 0

    # Build context based on the depth of the taxonomy
    if not parents:
        # Root level
        taxonomy_context = f"""
You are organizing papers in a corpus of {field} research. You need to order a list of papers by their relevance or centrality to the field of {field} as a whole.
"""
    else:
        # Non-root level
        taxonomy_context = f"""
You are organizing papers in a hierarchical taxonomy of {field} research. You need to order a list of papers by their relevance or centrality to a specific topic in this taxonomy.

The papers are categorized under the topic {topic.title}, which can be found at {topic_breadcrumbs(topic, parents)}. This topic is defined as: "{topic.description}"
"""

    # Add special instruction for non-dead-end topics (topics with subtopics)
    overview_context = ""
    if not is_dead_end:
        overview_context = f"""
Papers that provide broad overviews, surveys, or literature reviews of {topic.title} as a whole are particularly valuable and should be considered highly relevant to this topic.
"""

    # Full formatted prompt
    return f"""
{taxonomy_context}
{overview_context}
I'll provide you with a list of {len(papers)} papers that have been categorized under {topic.title}. Your task is to order these papers from most to least relevant to this topic.

PAPERS:
----
{corpus.get_pretty_sample(papers)}
----

Please respond with a JSON array of paper titles, ordered from most to least relevant to the topic {topic.title}. The array should include ALL paper titles from the list, with no omissions.

Example response format:
```json
[
  "Most relevant paper title",
  "Second most relevant paper title",
  "Third most relevant paper title",
  ...
]
```

Respond with ONLY the JSON array, no other text.
"""
