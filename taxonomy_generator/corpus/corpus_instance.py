from taxonomy_generator.corpus.ai_corpus import AICorpus

corpus = AICorpus()


if __name__ == "__main__":
    print()
    print(corpus.get_pretty_sample(1))
    print()
    print("Total papers:", len(corpus.papers))
    print()
