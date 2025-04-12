import json
from pathlib import Path

from taxonomy_generator.scripts.generator.generator import calculate_overall_score
from taxonomy_generator.scripts.generator.generator_types import EvalScores
from taxonomy_generator.utils.utils import plot_list

all_results = json.loads(Path("data/breakdown_results/all_results.json").read_text())


all_results = sorted(
    all_results,
    key=lambda x: calculate_overall_score(EvalScores.model_validate(x["scores"])),
    reverse=True,
)

for result in all_results[-1:]:
    print("Topics:")
    print(json.dumps(result["topics"], indent=2))
    print("Scores:")
    print(json.dumps(result["scores"], indent=2))
    print("Overall Score:")
    print(calculate_overall_score(EvalScores.model_validate(result["scores"])))
    print("--------------------------------")

plot_list(
    [
        calculate_overall_score(EvalScores.model_validate(r["scores"]))
        for r in all_results
    ]
)
