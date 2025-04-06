from taxonomy_generator.scripts.generator.types import Topic, TopicsFeedback
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


def resolve_topic(title: str, topics: list[Topic]):
    return next((t for t in topics if t.title.lower() == title.lower()), None)


def format_topics_feedbacks(topics_feedbacks: list[TopicsFeedback]):
    return "\n\n".join(
        [
            f"System Prompt: {tf.system or 'You are a helpful assistant'}\nScore: {tf.score}\nFeedback: {tf.feedback}"
            for tf in topics_feedbacks
        ]
    )


def resolve_get_topics_prompt(
    field: str,
):
    return """
The evaluation script ran successfully on your proposed breakdown. Here are the results:

# Sorting
A sample of {sort_sample_len} papers from the corpus were asked to be categorized by an LLM.

Of these, {easy_num} ({easy_perc}%) papers were categorized cleanly into one category.

For {no_place_num} ({no_place_perc}%) papers, the LLM couldn't find a suitable category. [[xamples:

---
{no_place_examples}
---]]

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
