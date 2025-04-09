from taxonomy_generator.utils.prompting import fps

IS_AI_SAFETY_EXPLANATION = """
Given the following paper:
---
{}
---

Please evaluate how much this paper contributes to the field of AI safety on a scale of 1-5:
1: Is counter productive or does not contribute to AI safety
2: Minimally contributes to AI safety
3: Moderately contributes to AI safety
4: Significantly contributes to AI safety
5: Highly valuable for the goals of AI safety

Provide your score and a one-sentence explanation only if the score is 1 or 2. Please respond with XML in the following format:

<score>#</score> <explanation>...</explanation>

Only include `<explanation>` if your score is 1 or 2.

Remember, this evaluation is about how much the paper contributes to AI safety goals, not just topical relevance. A paper that productively advances even a small sub-goal of AI safety should still receive a higher score.
"""

IS_AI_SAFETY = """
Given the following paper:
---
{}
---

Please evaluate how much this paper contributes to the field of AI safety on a scale of 1-5:
1: Counterproductive or does not contribute to AI safety
2: Minimally contributes to AI safety
3: Moderately contributes to AI safety
4: Significantly contributes to AI safety
5: Highly valuable for AI safety

To clarify, this evaluation is about how much the paper ultimately contributes to the goals of AI safety, not just topical relevance. A paper that productively advances even a small sub-goal under AI safety should still receive a higher score.

Please respond with only the numerical score (1-5) without any other text or explanation.
"""

fps(globals())
