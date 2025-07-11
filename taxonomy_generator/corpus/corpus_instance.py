from taxonomy_generator.config import CORPUS_PATH
from taxonomy_generator.corpus.corpus import Corpus

corpus = Corpus(CORPUS_PATH)


if __name__ == "__main__":
    print()
    print(corpus.get_pretty_sample(1))
    print()
    print("Total papers:", len(corpus.papers))
    print()
