# Hierarchical Taxonomy Generator for AI Safety Research Papers

## Automated Generation

- To start, give an LLM (claude-sonnet-3-7) a batch of 80 papers (only title and abstracts) randomly chosen from the corpus, and ask it to create a 1D taxonomy of sub-topics to catagorize all of the papers into. Explain that the 80 papers is just a subset of a larger corpus, and that it is opimizing for (1)  and (2) Helpfulness:

- Then, go through the full corpus (or just a large, random, subset), and for each paper, ask an LLM (gemini-flash-1.5) to catagorize it into one of the sub-topics. If the paper is
