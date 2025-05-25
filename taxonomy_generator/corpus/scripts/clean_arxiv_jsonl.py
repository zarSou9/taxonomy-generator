from pathlib import Path

import jsonlines

from taxonomy_generator.corpus.arxiv_helper import get_arxiv_id_from_url
from taxonomy_generator.corpus.corpus import Corpus

good_path = Path("data/classifier/ai_safety/good")

input_path = good_path / "arxiv.jsonl"
output_path = good_path / "arxiv_clean.jsonl"

with jsonlines.open(input_path.resolve().as_posix(), mode="r") as reader:  # pyright: ignore[reportUnknownMemberType]
    arx_ids = [get_arxiv_id_from_url(p["url"]) for p in reader]
    arx_ids = [arx for arx in arx_ids if arx]

corpus = Corpus(corpus_path=output_path)

corpus.add_papers(arx_ids)
