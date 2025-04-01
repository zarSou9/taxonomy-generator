from pydantic import BaseModel

from taxonomy_generator.corpus.reader import AICorpus
from taxonomy_generator.scripts.format_prompts import fps
from taxonomy_generator.utils.llm import Chat
from taxonomy_generator.utils.parse_llm import get_xml_content, parse_response_json

BREAKDOWN = """
Please develop a 1D taxonomy of the field of AI safety

Expected output:

<analysis>
...
</analysis>

<topics>
[
    {{
        "title": "Clear and concise title"
        "description": "~2 sentence description of topic"
    }}
]
</topics>
"""

BREAKDOWN_ITERATE = """
{num_papers} from the corpus were asked to be categorized by an LLM.

Of these, {easy_num} papers were reportedly easy to categorize

For {no_place_num} papers, the LLM reported that it didn't fit into any of the categories. Here are some examples of those:

{no_place_papers}

For {multiple_num} papers, the LLM reported the paper fit into multiple of the presented categories and couldn't uniquely categorize it. Here are some examples of those:

{multiple_papers}  # With reasonings from LLM

For the papers that were succesfully categorized, here is how many were in assigned to each topic:

{numbers_breakdown}

# Explain it is generally good to try and even out how many papers are in each category, but be cautious as this could just indicate there is less work in the topic but it is still important


Expected output:
<analysis>
...
</analysis>

<topics>
[
    {{
        "title": "Clear and concise title"
        "description": "~2 sentence description of topic"
    }}
]
</topics>
"""

fps(globals())

corpus = AICorpus()


class Topic(BaseModel):
    title: str
    description: str


class BreakdownResponse(BaseModel):
    analysis: str
    topics: list[Topic]


def resolve_breakdown_response(res: str):
    return BreakdownResponse(
        analysis=get_xml_content(res, "analysis"),
        topics=parse_response_json(get_xml_content(res, "analysis"), []),
    )


def main():
    chat = Chat()

    while True:
        b_response = resolve_breakdown_response(chat.ask(BREAKDOWN))

        b_response


if __name__ == "__main__":
    main()
