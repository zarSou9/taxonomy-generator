from taxonomy_generator.corpus.corpus_instance import corpus
from taxonomy_generator.utils.exa import search_arxs
from taxonomy_generator.utils.prompting import fps

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


def main():
    corpus.add_papers(
        search_arxs(
            EXA_NOVEL_APPROACH.format(),
            num_results=25,
        ),
        verbose=2,
    )


if __name__ == "__main__":
    main()
