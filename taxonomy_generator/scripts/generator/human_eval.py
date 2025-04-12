from taxonomy_generator.corpus.corpus_instance import corpus
from taxonomy_generator.scripts.generator.generator import (
    evaluate_topics,
    resolve_topic_papers,
)
from taxonomy_generator.scripts.generator.generator_types import Topic
from taxonomy_generator.scripts.generator.prompts import get_iter_topics_prompt

TOPICS = [
    {
        "title": "Present-Day AI Safety",
        "description": "Research on making current AI systems robust, reliable, and aligned with human values. This includes adversarial defenses, formal verification, transparency methods, safety testing protocols, fairness evaluation, and techniques to ensure existing AI technologies operate safely and predictably.",
    },
    {
        "title": "Emerging AI Safety Challenges",
        "description": "Research addressing the evolving safety landscape as AI capabilities grow in sophistication and deployment scope. This includes scalable oversight mechanisms, value alignment frameworks, safety standards development, governance structures for increasingly powerful systems, and methods to handle increasingly complex AI behaviors.",
    },
    {
        "title": "Long-Term AI Safety",
        "description": "Research on foundational safety approaches for potentially transformative future AI capabilities. This includes theoretical frameworks for advanced AI alignment, power-seeking prevention, deceptive alignment, emergent capabilities, existential risk reduction, and ensuring safe development paths toward more capable AI systems.",
    },
]


def main():
    papers = resolve_topic_papers(corpus.papers)
    topics = [Topic.model_validate(t) for t in TOPICS]

    eval_result = evaluate_topics(
        topics, 400, papers, sample_seed=122, no_overviews=True, no_feedback=True
    )

    print("--------------------------------")
    print(f"All Scores:\n{eval_result.all_scores.model_dump_json(indent=2)}")
    print(f"Overall Score: {eval_result.overall_score}")
    print("--------------------------------")

    print(get_iter_topics_prompt(eval_result, True))


if __name__ == "__main__":
    main()
