from taxonomy_generator.corpus.reader import AICorpus
from taxonomy_generator.scripts.format_prompts import fps
from taxonomy_generator.utils.exa import search_arx_urls

"""
Fields:
- AI safety
- AI alignment
- AI governance
- AI mechanistic interpretability
- formal software verification

Approaches:
- Value learning in the context of AI safety
- Value learning in the context of AI safety, and addresses
"""

EXA_SURVEY = """
This survey paper provides a comprehensive overview of {} research
"""

EXA_KEY = """
This is a key paper in the field of {}
"""

EXA_FOUNDATIONAL = """
This paper is considered a foundational work in the field of {}, and has significantly influenced subsequent research
"""

EXA_NOVEL_APPROACH = """
This paper presents a novel approach to {}
"""

fps(globals())

corpus = AICorpus()


def main():
    corpus.add_papers(
        search_arx_urls(
            EXA_NOVEL_APPROACH.format(),
            num_results=25,
        ),
        verbose=2,
    )


if __name__ == "__main__":
    main()
