from taxonomy_generator.corpus.reader import AICorpus
from taxonomy_generator.scripts.format_prompts import fps
from taxonomy_generator.utils.exa import fetch_arxiv_urls

FIELDS = [
    "AI safety",
    "AI alignment",
    "AI governance",
    "AI mechanistic interpretability",
    "formal software verification",
]

EXA_SURVEY_PAPER = """
This survey paper provides a comprehensive overview of {} research
"""

EXA_MECHANISTIC_INTERPRETABILITY = """
This paper presents a novel approach to mechanistic interpretability of AI models
"""

EXA_FOUNDATIONAL_PAPER = """
This paper is considered a foundational work in the field of AI alignment, and has significantly influenced subsequent research
"""

EXA_ROBUSTNESS_EVALUATION = """
This paper presents a novel approach to evaluating the robustness of AI systems against adversarial attacks and distribution shifts
"""

EXA_GOVERNANCE_PAPER_POLICY = """
This paper presents an innovative approach to AI governance proposing a novel policy framework
"""

EXA_GOVERNANCE_PAPER = """
This is a key paper in the field of AI governance
"""

fps(globals())

corpus = AICorpus()


def main():
    results = fetch_arxiv_urls(EXA_SURVEY_PAPER.format(FIELDS[0]))
    papers = corpus.add_papers(results)
    print(corpus.get_pretty_sample(papers, ["title", "url", "published", "abstract"]))


if __name__ == "__main__":
    main()
