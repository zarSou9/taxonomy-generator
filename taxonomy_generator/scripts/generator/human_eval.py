from taxonomy_generator.corpus.corpus_instance import corpus
from taxonomy_generator.scripts.generator.generator import evaluate_topics
from taxonomy_generator.scripts.generator.generator_types import Topic
from taxonomy_generator.scripts.generator.prompts import get_iter_topics_prompt
from taxonomy_generator.scripts.generator.utils import resolve_topic_papers

TOPICS = [
    {
        "title": "Robustness & Security",
        "description": "Research on making AI systems resilient against adversarial attacks, distribution shifts, and security threats in deployment environments. This includes adversarial defenses, uncertainty quantification, robust training methods, and techniques to detect or mitigate malicious attacks on AI systems.",
    },
    {
        "title": "Alignment & Value Learning",
        "description": "Research on ensuring AI systems understand, learn, and act in accordance with human values, preferences, and intentions. This includes reinforcement learning from human feedback (RLHF), preference optimization, reward modeling, value specification, and approaches to address challenges in accurate representation of complex human objectives.",
    },
    {
        "title": "Interpretability & Transparency",
        "description": "Research on making AI decision processes and internal representations understandable to humans. This includes mechanistic interpretability, feature attribution, explainable AI (XAI), neural network visualization techniques, and methods to generate human-comprehensible explanations of model behavior.",
    },
    {
        "title": "Governance & Policy",
        "description": "Research on institutional frameworks, regulatory approaches, and ethical guidelines for responsible AI development and deployment. This includes international coordination mechanisms, standards development, risk management frameworks, legal structures, auditing systems, and organizational practices that promote safe and beneficial AI.",
    },
    {
        "title": "Verification & Testing",
        "description": "Research on formal verification methods and systematic evaluation approaches to confirm AI systems meet safety specifications. This includes mathematical verification, formal guarantees, red teaming, benchmarking, safety metrics, and methodologies for comprehensive testing of AI capabilities and limitations against predefined criteria.",
    },
    {
        "title": "Fairness & Societal Impact",
        "description": "Research on ensuring AI systems produce fair, beneficial, and equitable outcomes across diverse populations and contexts. This includes algorithmic fairness, bias mitigation, privacy protection, studies of technological impact on society, and approaches to measure and address the distributional effects of AI systems.",
    },
]


def main():
    papers = resolve_topic_papers(corpus.papers)
    topics = [Topic.model_validate(t) for t in TOPICS]

    eval_result = evaluate_topics(
        topics, 400, papers, sample_seed=222, no_overviews=True
    )

    print("--------------------------------")
    print(f"All Scores:\n{eval_result.all_scores.model_dump_json(indent=2)}")
    print(f"Overall Score: {eval_result.overall_score}")
    print("--------------------------------")

    print(get_iter_topics_prompt(eval_result, True))


if __name__ == "__main__":
    main()
