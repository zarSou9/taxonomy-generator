from taxonomy_generator.corpus.corpus_instance import corpus
from taxonomy_generator.scripts.generator.generator import (
    evaluate_topics,
    resolve_topic_papers,
)
from taxonomy_generator.scripts.generator.generator_types import Topic
from taxonomy_generator.scripts.generator.prompts import get_iter_topics_prompt

TOPICS = [
    {
        "title": "Technical Robustness and Safety Verification",
        "description": "Research on methods to make AI systems reliable, resistant to attacks, and mathematically verified to meet safety specifications. This includes adversarial defenses, formal verification, testing frameworks, uncertainty quantification, and techniques to ensure systems behave as expected under various conditions.",
    },
    {
        "title": "Safe Decision-Making and Learning",
        "description": "Work focused on ensuring AI systems make safe decisions and learn safely, especially when interacting with environments or humans. This includes safe reinforcement learning, constrained optimization, safe exploration strategies, and methods for reliable decision-making under uncertainty.",
    },
    {
        "title": "AI Alignment and Human Values",
        "description": "Research on aligning AI systems with human values, preferences, intentions, and ethical considerations. This includes human feedback techniques, preference optimization, reward modeling, and methods to ensure AI systems understand and respect human objectives, including work on interpretability and explainability to make AI reasoning transparent.",
    },
    {
        "title": "AI Governance and Societal Impact",
        "description": "Studies addressing the governance, regulation, and broader social implications of AI systems. This includes policy frameworks, regulatory approaches, fairness and bias mitigation, and analyses of how AI systems affect different populations, institutional structures, and society as a whole.",
    },
]


def main():
    papers = resolve_topic_papers(corpus.papers)
    topics = [Topic.model_validate(t) for t in TOPICS]

    eval_result = evaluate_topics(
        topics, 400, papers, sample_seed=122, no_overviews=True
    )

    print("--------------------------------")
    print(f"All Scores:\n{eval_result.all_scores.model_dump_json(indent=2)}")
    print(f"Overall Score: {eval_result.overall_score}")
    print("--------------------------------")

    print(get_iter_topics_prompt(eval_result, True))


if __name__ == "__main__":
    main()
