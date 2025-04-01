from taxonomy_generator.scripts.format_prompts import fps

IS_AI_SAFETY = """
Given the following paper:
---
{}
---

Please evaluate how much this paper contributes to the field of AI safety on a scale of 1-4:
1: Not relevant to AI safety
2: Minimally relevant to AI safety
3: Significantly contributes to AI safety
4: Highly valuable for the goals of AI safety

Provide your score and a one-sentence explanation only if the score is 1 or 2. Please respond with XML in the following format:

<score>#</score> <explanation>...</explanation>

Only include `<explanation>` if your score is 1 or 2
"""


fps(globals)
