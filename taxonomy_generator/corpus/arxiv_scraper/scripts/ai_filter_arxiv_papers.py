from taxonomy_generator.config import (
    ARXIV_AI_FILTERED_PAPERS_FORMAT,
    ARXIV_CATEGORY_METADATA,
    ARXIV_FILTERED_PAPERS_FORMAT,
    CATEGORY,
)
from taxonomy_generator.corpus.corpus import read_papers_jsonl, write_papers_jsonl
from taxonomy_generator.corpus.corpus_types import Paper
from taxonomy_generator.corpus.utils import get_pretty_paper
from taxonomy_generator.utils.llm import run_in_parallel
from taxonomy_generator.utils.prompting import prompt


@prompt
def get_filter_arxiv_paper_prompt(paper: Paper) -> str:
    return f"""
Review this paper to determine if it meets quality standards and belongs in the specified arXiv category.

<arxiv_category>
Parent Category: {ARXIV_CATEGORY_METADATA.category_group}
Code: {ARXIV_CATEGORY_METADATA.code}
Title: {ARXIV_CATEGORY_METADATA.name}
Description: {ARXIV_CATEGORY_METADATA.description}
</arxiv_category>

<paper>
{get_pretty_paper(paper)}
</paper>

## Quality Criteria
Assess if the paper meets basic academic standards. Mark as LOW quality (N) if it:
- Contains obvious factual errors or unsupported claims
- Has poor writing quality (incoherent structure, excessive grammatical errors)
- Appears to be machine-generated or spam
- Would not be useful to researchers (no clear contribution or findings)

Mark as SUFFICIENT quality (Y) if the paper appears to be a legitimate research contribution, even if preliminary or incremental.

## Relevance Criteria
Assess if the paper's primary focus aligns with the category. Mark as IRRELEVANT (N) if:
- The main topic clearly belongs to a different field
- The category is mentioned only tangentially or in passing
- The paper was miscategorized (e.g., a biology paper in a physics category)

Mark as RELEVANT (Y) if the paper's core content directly addresses topics within the category description above.

## Response Format
Provide your assessment in exactly this format:

QUALITY: Y/N
RELEVANCE: Y/N

Use Y for sufficient quality/relevance, N for insufficient
"""


def ai_filter_arxiv_papers():
    papers = read_papers_jsonl(ARXIV_FILTERED_PAPERS_FORMAT.format(CATEGORY))

    # Generate prompts for all papers
    prompts = [get_filter_arxiv_paper_prompt(paper) for paper in papers]

    # Process all papers in parallel
    responses = run_in_parallel(prompts, model="gemini-2.5-flash", max_workers=40)

    # Process responses and filter papers
    filtered_papers: list[Paper] = []
    for paper, response in zip(papers, responses, strict=True):
        lines = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in response.strip().split("\n")
            if ":" in line
        }
        quality_response = lines.get("QUALITY")
        relevance_response = lines.get("RELEVANCE")

        if quality_response is None or relevance_response is None:
            print("Invalid reponse from LLM")
            print(f"PROMPT:\n---\n{prompts[papers.index(paper)]}\n---\n")
            print(f"RESPONSE:\n---\n{response}\n---\n")
            continue

        if not quality_response == "Y" or not relevance_response == "Y":
            print("Skipping paper:")
            print("-" * 20)
            print(get_pretty_paper(paper))
            print("-" * 20)
            print(response)
            print("-" * 20)
            continue

        filtered_papers.append(paper)

    write_papers_jsonl(
        ARXIV_AI_FILTERED_PAPERS_FORMAT.format(CATEGORY), filtered_papers
    )


if __name__ == "__main__":
    ai_filter_arxiv_papers()
