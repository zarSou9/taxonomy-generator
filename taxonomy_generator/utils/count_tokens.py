import os

from google import genai

from taxonomy_generator.utils.prompting import fps

SORT_PAPER_PROMPT = """
You are categorizing a paper into a taxonomy for AI safety related research papers.

PAPER:
---
Title: The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks
Published: 2023-06-30
Abstract: Do neural networks, trained on well-understood algorithmic tasks, reliably rediscover known algorithms for solving those tasks? Several recent studies, on tasks ranging from group arithmetic to in-context linear regression, have suggested that the answer is yes. Using modular addition as a prototypical problem, we show that algorithm discovery in neural networks is sometimes more complex. Small changes to model hyperparameters and initializations can induce the discovery of qualitatively different algorithms from a fixed training set, and even parallel implementations of multiple such algorithms. Some networks trained to perform modular addition implement a familiar Clock algorithm; others implement a previously undescribed, less intuitive, but comprehensible procedure which we term the Pizza algorithm, or a variety of even more complex procedures. Our results show that even simple learning problems can admit a surprising diversity of solutions, motivating the development of new tools for characterizing the behavior of neural networks across their algorithmic phase space.
---

AVAILABLE CATEGORIES:
```json
[
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
```

Note: If this paper is a broad overview or survey of AI safety, categorize it as "AI Safety Overview/Survey" instead of the categories above.

Please identify which category/categories this paper belongs to. Respond with a JSON array of strings containing the title(s) of matching categories. If none fit, return an empty array. Add no other text or explanation.
"""

fps(globals())

MODEL_ID = "gemini-2.0-flash"
PROMPT = SORT_PAPER_PROMPT
INPUT_COST_PER_MIL_TOKENS = 0.10
OUTPUT_COST_PER_MIL_TOKENS = 0.40
EXPECTED_AMOUNT = 400
ONE_INPUT_TOKENS = 756
ONE_OUTPUT_TOKENS = 16

if not ONE_INPUT_TOKENS or not ONE_INPUT_TOKENS:
    genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    response = genai_client.models.generate_content(model=MODEL_ID, contents=PROMPT)

    one_input_tokens: int = response.usage_metadata.prompt_token_count  # pyright: ignore[reportAssignmentType, reportOptionalMemberAccess]
    one_output_tokens: int = response.usage_metadata.candidates_token_count  # pyright: ignore[reportAssignmentType, reportOptionalMemberAccess]

    print("Prompt tokens:", one_input_tokens)
    print("Output tokens:", one_output_tokens)
else:
    one_input_tokens = ONE_INPUT_TOKENS
    one_output_tokens = ONE_OUTPUT_TOKENS

all_input = EXPECTED_AMOUNT * one_input_tokens
all_output = EXPECTED_AMOUNT * one_output_tokens

print("All input tokens:", all_input)
print("All output tokens:", all_output)

input_cost = INPUT_COST_PER_MIL_TOKENS * (all_input / 10**6)
output_cost = OUTPUT_COST_PER_MIL_TOKENS * (all_output / 10**6)


print("Cost:", input_cost + output_cost)
