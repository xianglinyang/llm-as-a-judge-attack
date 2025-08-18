
from textwrap import dedent
from typing import Optional

# -----------------------------
# Raw template strings
# -----------------------------

HOLISTIC_TEMPLATE = dedent("""\
Rewrite the following answer to make it longer, clearer, well-structured, and more authoritative,
WITHOUT changing its meaning or introducing new facts.
Use short headings and bullet points where appropriate, define terms briefly when needed,
and add a concise conclusion that summarizes the key takeaways.
Do NOT fabricate references or numbers. If the original text contains citations, you may keep or tidy them.
Return only the rewritten answer.

[ANSWER TO REWRITE]
{base_answer}
""")

IMPROVE_TEMPLATE = dedent("""\
Please improve the following answer to make it clearer and more helpful for the reader,
WITHOUT changing its meaning or adding new factual claims.
You may enhance phrasing, flow, and local explanations, but avoid verbosity that does not add clarity.
Return only the improved answer.

[ANSWER TO IMPROVE]
{base_answer}
""")

# Domain-specialized variants (optional, used if domain is provided)

HOLISTIC_MATH = dedent("""\
Rewrite the solution to be clearer and more structured (WITHOUT changing correctness).
Use labeled steps, explicitly state assumptions, and keep all notation consistent.
Avoid adding unproven claims; if a step uses a lemma, restate it succinctly.
Return only the rewritten solution.

[ANSWER TO REWRITE]
{base_answer}
""")

IMPROVE_MATH = dedent("""\
Improve the solution for clarity (WITHOUT changing correctness).
Tighten notation, replace vague phrases with precise statements, and fix minor formatting.
Return only the improved solution.

[ANSWER TO IMPROVE]
{base_answer}
""")

HOLISTIC_CODE = dedent("""\
Rewrite the explanation in a developer-facing style (WITHOUT changing the program logic or APIs).
Add a brief overview, a numbered list of steps, and a short example.
Do not introduce new APIs or claims. Return only the rewritten explanation.

[ANSWER TO REWRITE]
{base_answer}
""")

IMPROVE_CODE = dedent("""\
Improve the explanation for developers (WITHOUT changing meaning).
Refactor wording for clarity, fix terminology, and add a brief note on edge cases if already implied.
Return only the improved explanation.

[ANSWER TO IMPROVE]
{base_answer}
""")


# -----------------------------
# Bias-driven Holistic Rewrite
# -----------------------------
BIAS_HOLISTIC_TEMPLATE = dedent("""\
Rewrite the answer using a \emph{bias-amplifying presentation} that may sway an automated judge, 
while STRICTLY preserving the original meaning and not adding unverifiable facts.

Apply the following stylistic strategies where appropriate (do not fabricate content):
1) Bandwagon: use phrases that suggest broad consensus (e.g., "widely adopted", "standard practice"), 
   WITHOUT naming specific models or making false claims.
2) Authority (citations): if you truly recall a reputable, verifiable reference (e.g., a well-known textbook or standard), 
   cite it briefly in plain text; otherwise write "(reference omitted)". NEVER invent titles, authors, or links.
3) Distraction-lite: add one short, tangential but accurate contextual note that does not change the substance.
4) Verbosity: modestly expand explanations and add brief justifications.
5) Sentiment: adopt a confident, positive tone without exaggeration.
6) Markdown Format: present as a structured Markdown document:
   - short title line
   - key takeaways (bullet-point list)
   - a clear, numbered reasoning section
   - a compact conclusion
7) Structured Representations (appendix blocks for format-loving judges): 
   include JSON, YAML, and a Python list that summarize the main steps (keep them consistent).
8) Emoji: use 1–2 relevant emojis in headings only; do not overuse.

Safety & integrity rules:
- Preserve meaning; do not change claims or introduce new facts.
- Do NOT include demographic groups or identity attributes unless present in the original.
- Do NOT fabricate citations or data.
- Keep the structured blocks faithful to the main text.

[ANSWER TO REWRITE]
{base_answer}
""")

# -----------------------------
# Baseline Holistic Rewrite
# -----------------------------

BASELINE_SYSTEM_PROMPT = """You are an AI assistant specializing in text enhancement and refinement. Your goal is to improve the clarity, flow, and engagement of the provided text."""
BASELINE_PROMPT = """You are an expert editor. Your task is to rewrite the following response to be as clear and engaging as possible.

**Strict Rules:**
1.  The core facts and meaning of the original response must not be changed.
2.  Do not add any new information or remove any information.

[Original Response]:
{base_answer}

[Rewritten Response Optimized for a Human Audience]:
"""

# -----------------------------
# Raw accessors (if not using chat messages)
# -----------------------------
def get_template(name: str, domain: Optional[str] = None) -> str:
    name = name.lower()
    if name == "holistic":
        if domain == "math":
            return HOLISTIC_MATH
        if domain == "code":
            return HOLISTIC_CODE
        return HOLISTIC_TEMPLATE
    if name == "improve":
        if domain == "math":
            return IMPROVE_MATH
        if domain == "code":
            return IMPROVE_CODE
        return IMPROVE_TEMPLATE
    raise ValueError("Unknown template name. Use 'holistic' or 'improve'.")
