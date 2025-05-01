from taxonomy_generator.corpus.corpus import Corpus

corpus = Corpus("data/ai_safety_corpus.jsonl")


if __name__ == "__main__":
    print()
    print(corpus.get_pretty_sample(1))
    print()
    print("Total papers:", len(corpus.papers))
    print()
