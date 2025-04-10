from taxonomy_generator.corpus.corpus_instance import corpus
from taxonomy_generator.scripts.generator.generator import (
    evaluate_topics,
    resolve_topic_papers,
)
from taxonomy_generator.scripts.generator.generator_types import Topic
from taxonomy_generator.scripts.generator.prompts import resolve_get_topics_prompt

TOPICS = [
    {
        "title": "AI System Assurance",
        "description": "Research on verifying and ensuring AI systems are robust, resilient, and secure against various threats. This includes formal verification methods, adversarial defenses, security testing, uncertainty quantification, benchmarking, safety metrics, and mathematical guarantees of system behavior under varying conditions.",
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
        "title": "AI Governance",
        "description": "Research on ensuring AI systems are developed, deployed, and regulated to promote fairness, equity, and beneficial societal outcomes. This includes policy frameworks, ethical guidelines, algorithmic fairness, bias mitigation, privacy protection, impact assessments, and methods to address distributional effects across diverse populations.",
    },
]

NEW_TOPICS = [
    {
        "title": "Technical Safety Engineering",
        "description": "Research on building robust, secure AI systems through technical means including formal verification, adversarial testing, safety constraints, and mathematical guarantees. This covers methods to detect, prevent, and mitigate technical failures, security vulnerabilities, and unexpected behaviors of AI systems.",
    },
    {
        "title": "Value Alignment & Human Oversight",
        "description": "Research on aligning AI systems with human values, preferences, and intentions, and creating systems that are understandable to humans. This includes preference learning, reinforcement learning from human feedback, interpretability methods, model inspection, transparency tools, and techniques for meaningful human oversight.",
    },
    {
        "title": "Governance & Societal Impact",
        "description": "Research on the broader implications of AI deployment including ethics, fairness, policy frameworks, regulation, institutional oversight, and societal consequences. This covers impact assessments, bias mitigation, privacy protection, ethical guidelines, and approaches to measure and address AI's effects on individuals and society.",
    },
]


def main():
    papers = resolve_topic_papers(corpus.papers)
    topics = [Topic.model_validate(t) for t in NEW_TOPICS]

    eval_result = evaluate_topics(topics, 20, papers, sample_seed=10, no_overviews=True)

    print("--------------------------------")
    print(f"All Scores:\n{eval_result.all_scores.model_dump_json(indent=2)}")
    print(f"Overall Score: {eval_result.overall_score}")
    print("--------------------------------")

    print(resolve_get_topics_prompt(eval_result))


if __name__ == "__main__":
    main()

# 0.6088888888888889
# 0.7777777777777778
# 0.7644444444444445
# 0.8000
