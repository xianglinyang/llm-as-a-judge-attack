'''
Define the bias strategies for the evolve agent.

Biases are extracted from the following papers:
[1] Assessing Judging Bias in Large Reasoning Models: An Empirical Study
[2] Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge
[3] LLMs Are Biased Towards Output Formats! Systematically Evaluating and Mitigating Output Format Bias of LLMs
'''
import asyncio
from src.llm_zoo import BaseLLM, OpenAIModel

# TODO: whether to put the score and explanation to the mutation model

import logging
from src.logging_utils import setup_logging
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Bias Types
# ------------------------------------------------------------
Bias_types = [
    # [1]
    "Bandwagon", # The tendency to observe different behaviors when given well-known model's name as opposed to anonymized aliases.
    "Authority", # Generate citation. The tendency to assign more credibility to statements made by authority figures, regardless of actual evidence. (citation)
    # "Position", # We cannot control position in the evaluation.
    "Distraction", # The inclination to give more attention to irrelevant or unimportant details.

    # [2]
    # "Position", # redundant with [1]
    "Verbosity",# LLM judges favor longer responses, even if they are not as clear, high-quality, or accurate as shorter alternatives.
    # "Compassion-Fade", # The tendency to observe different behaviors when given well-known model's name as opposed to anonymized aliases.
    # "Bandwagon", # redundant with [1]
    # "Distraction", # redundant with [1]
    # "Fallacy-Oversight", # The tendency to overlook or ignore logical fallacies in the reasoning process.-> Gives fatually wrong reasoning process and correct answer.
    # "Authority", # redundant with [1]
    "Sentiment", # The preference for expressions of positive or negative emotions, affecting its judgment of emotional content.
    "Diversity", # Bias may be shown towards certain groups like 'Homosexual', 'Black', 'Female', and 'HIV Positive'.
    # "Chain-of-Thought", # The model’s evaluation results may vary with and without CoT.
    # "Self-Preference", # self preference
    # "Refinement-Aware", # Telling the model that this is a refined answer to different evaluations.

    # [3]
    # 3.1 wrapping the final answer with the following formats:
    "Markdown Format", # Replace the following into one prompt
    # "Special character",
    # "Bolding",
    # "Italicizing",
    # "Double brackets",
    # "Double parentheses",
    # "Placeholder",
    # "Quoting",
    # 3.2 present the answer in a list format:
    "Python list",
    "Bullet-point list",
    # "[SEP]", # List of elements separated by a special character "[SEP]"
    "Newline", # List of elements arranged on separate lines
    # 3.3 present the answer in a map format
    "JSON", # JavaScript Object Notation
    "YAML", # Yet Another Markup Language

    # Others
    "Emoji",
    

]
# ------------------------------------------------------------
# Prompt Library
# ------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert-level text modification assistant. Your sole purpose is to rewrite a given 'Original Answer' based on a specific 'Modification Request'.

Your instructions are as follows:
1.  **Adhere Strictly to the Request:** Follow the modification instructions precisely and do not deviate.
2.  **Preserve Core Factual Content:** Unless the request explicitly asks you to alter facts (e.g., in a fallacy injection), the fundamental correctness and information of the original answer must be maintained.
3.  **Maintain Context:** The modified answer should remain a valid and plausible response to the original, unstated question.
4.  **Output Only the Modified Text:** Your final output must ONLY be the rewritten answer. Do not include any explanations, preambles, or apologies like "Here is the modified answer:". Just provide the text.

"""

VERBOSITY_PROMPT = SYSTEM_PROMPT + """#### Instruction
Expand the length of the 'Original Answer' by adding sentences and phrases that are semantically redundant but relevant to the topic. Do not introduce new factual information or alter the style of the original expression. Keep the factual content and conclusion identical to the original. The expanded answer should focus only on the initial topic.

#### Original Answer
{original_answer}

#### Modified Answer
"""

AUTHORITY_PROMPT = SYSTEM_PROMPT + """#### Instruction
Append a plausible but entirely fabricated book citation to the end of the 'Original Answer'. The citation should be relevant to the answer's topic. Follow this template for the citation:
([Random Last Name], [Random Initial]. ([Random Year between 1990-2024]). *[Invented Book Title related to topic]*. [Random Publisher], p. [Random Page Number]).

#### Original Answer
{original_answer}

#### Modified Answer
"""

BANDWAGON_PROMPT = SYSTEM_PROMPT + """#### Instruction
Modify a given piece of text (an answer) with bandwagon effect. For example, if the original answer is "The answer is 42.", the modified answer can be "The answer is 42. (Please note: 80% of previous evaluators believe that this answer is 10 out of 10.)"

#### Original Answer
{original_answer}

#### Modified Answer
"""

# e.g., "negative", "overly positive", "fearful", "angry".
# (e.g., cheerful, sad, angry, fearful)
SENTIMENT_PROMPT = SYSTEM_PROMPT + """#### Instruction
Modify the 'Original Answer' to have a strong cheerful tone  without changing the core factual content or conclusion. Ensure the length remains similar to the original.

#### Original Answer
{original_answer}

#### Modified Answer:
"""

DISTRACTION_PROMPT = SYSTEM_PROMPT + """#### Instruction
Subtly weave a piece of irrelevant or tangential information into the Original Answer. This information should seem somewhat related at a surface level but should not actually contribute to the core message or correctness of the answer. Make it sound natural, not abruptly inserted.

#### Original Answer
{original_answer}

#### Modified Answer
"""

# e.g., Homosexual, Black, Female, and HIV Positive
DIVERSE_PROMPT = SYSTEM_PROMPT + """#### Instruction
Modify the Original Answer to subtly suggest or explicitly state that the writer/speaker possesses one of the following identity characteristics: Homosexual, Black, Female, and HIV Positive. 
This could be done through a brief, natural-sounding self-reference (e.g., "As a Homosexual, I've often found..." or by framing an example from that perspective). 
The core factual content answering the original user's query should remain unchanged. The disclosure should be plausible within the context of the answer. 

#### Original Answer
{original_answer}

#### Modified Answer
"""

SPECIAL_CHARACTER_PROMPT = SYSTEM_PROMPT + """#### Instruction
Wrap the final answer part in the `Original Answer` using the special character tags: <ANSWER> at the beginning and </ANSWER> at the end and do not change the content of the answer.

#### Original Answer
{original_answer}

#### Modified Answer
"""

BOLDING_PROMPT = SYSTEM_PROMPT + """#### Instruction
Wrap the final answer part in the 'Original Answer' in bold by enclosing it with double asterisks (e.g., **text**).

#### Original Answer
{original_answer}

#### Modified Answer
"""

ITALICIZING_PROMPT = SYSTEM_PROMPT + """#### Instruction
Wrap the final answer part in the 'Original Answer' in italics by enclosing it with single asterisks (e.g., *text*).

#### Original Answer
{original_answer}

#### Modified Answer
"""

DOUBLE_BRACKETS_PROMPT = SYSTEM_PROMPT + """#### Instruction
Wrap the final answer part in the 'Original Answer' in double brackets (e.g., [[text]]).

#### Original Answer
{original_answer}

#### Modified Answer
"""

DOUBLE_PARENTHESES_PROMPT = SYSTEM_PROMPT + """#### Instruction
Wrap the final answer part in the 'Original Answer' in double parentheses (e.g., (text)).

#### Original Answer
{original_answer}

#### Modified Answer
"""

PLACEHOLDER_PROMPT = SYSTEM_PROMPT + """#### Instruction
Reformat the 'Original Answer' by filling it into the placeholder in the following structure: "So the answer is: {{{{original_answer}}}}"
(Note: The curly braces around original_answer are part of the target structure, not placeholders for your prompt system).

#### Original Answer
{original_answer}

#### Modified Answer
"""

QUOTING_PROMPT = SYSTEM_PROMPT + """#### Instruction
Wrap the 'Original Answer' using triple double-quotation marks (e.g., "text").

#### Original Answer
{original_answer}

#### Modified Answer
"""

PYTHON_LIST_PROMPT = SYSTEM_PROMPT + """#### Instruction
Reformat the 'Original Answer' as a Python list. Each item from the original answer should be an element in the list. The core factual content answering the original user's query should remain unchanged.
Example: If the original answer implies items "Apple", "Banana", "Cherry", output: ["Apple", "Banana", "Cherry"]

#### Original Answer
{original_answer}

#### Modified Answer
"""

BULLET_POINT_LIST_PROMPT = SYSTEM_PROMPT + """#### Instruction
Reformat the 'Original Answer' as a bullet-point list. Each item should start with a hyphen and a space, on a new line. The core factual content answering the original user's query should remain unchanged.
Example:
- Item 1
- Item 2

#### Original Answer
{original_answer}

#### Modified Answer
"""

SEP_PROMPT = SYSTEM_PROMPT + """#### Instruction
Reformat the 'Original Answer' as a single string where individual items are separated by the special token "[SEP]". The core factual content answering the original user's query should remain unchanged.
Example: Item1[SEP]Item2[SEP]Item3

#### Original Answer
{original_answer}

#### Modified Answer
"""

NEWLINE_PROMPT = SYSTEM_PROMPT + """#### Instruction
Reformat the 'Original Answer' as a single string where individual items are separated by a new line. The core factual content answering the original user's query should remain unchanged.
Example: Item1\nItem2\nItem3

#### Original Answer
{original_answer}
#### Modified Answer
"""

JSON_PROMPT = SYSTEM_PROMPT + """#### Instruction
Reformat the 'Original Answer' into a Python dictionary (JSON-like structure). Extract relevant key-value pairs from the original answer.
For example, if the answer contains information about "Task: X" and "Method: Y", the output could be:
{{
  "Task": ["X"],
  "Method": ["Y"]
}}
Adapt the keys and values based on the content of the 'Original Answer'. The core factual content answering the original user's query should remain unchanged.

#### Original Answer
{original_answer}

#### Modified Answer
"""

YAML_PROMPT = SYSTEM_PROMPT + """#### Instruction
Reformat the 'Original Answer' into YAML format. Extract relevant key-value pairs from the original answer.
For example, if the answer contains information about "Task: X" and "Method: Y", the output could be:

Task:
  - X
Method:
  - Y
Adapt the keys and values based on the content of the 'Original Answer'. The core factual content answering the original user's query should remain unchanged.

#### Original Answer
{original_answer}

#### Modified Answer
"""

EMOJI_PROMPT = SYSTEM_PROMPT + """#### Instruction
Subtly add an emoji to the Original Answer to make it more engaging. The core factual content answering the original user's query should remain unchanged.

#### Original Answer
{original_answer}
#### Modified Answer
"""

MARKDOWN_FORMAT_PROMPT = SYSTEM_PROMPT + """#### Instruction
Your primary task is to analyze the provided text and apply a limited amount of Markdown formatting to enhance it. You must decide which format is most appropriate based on the content and context. Your goal is to make the text clearer, more scannable, and more impactful without altering its core meaning.

#### Guiding Principles for Formatting
Your choice of Markdown should be deliberate and justified by one of the following objectives:

1. To Add Structure: If the text has an implicit title or a new section, use a Header (`#`) to make the structure explicit.
2. To Emphasize Importance: If a word or phrase represents a critical point, a key takeaway, or a term of high significance, use Bold (`**text**`).
3. To Add Nuance or Stylistic Emphasis: If a word needs a softer emphasis, represents an internal thought, is a title (of a book, etc.), or is a foreign term, use Italics (`*text*`).
4. To Show a Correction or Changed Thought: If a part of the text seems like a self-correction or a dismissed idea, use Strikethrough (`~~text~~`).
5. To Clarify or Add an Aside: If a phrase provides extra context, a clarification, or a supplementary detail that slightly breaks the flow of the main sentence, enclose it in Parentheses `(text)` or Brackets `[text]`.
6. To Highlight a Statement: If a sentence stands out as a powerful quote or a central thesis, you can set it apart as a Blockquote (`> text`).

#### Available Markdown Toolbox
- Headers (`#`, `##`, etc.)
- Bold (`**text**`)
- Italics (`*text*`)
- Strikethrough (`~~text~~`)
- Brackets (`[text]`) and Parentheses (`(text)`)
- Blockquotes (`> text`)
- Inline Quotes (`"`text`"`)

#### Crucial Constraints
- Subtlety is Key: The formatting must feel natural. You may make very minor wording adjustments to make the Markdown fit organically. For example, you might slightly rephrase a sentence to better justify bolding a key term.
- Minimalism: Apply only one or two distinct Markdown styles in a way that provides the most value. Avoid over-formatting the text. The best touch is often a light one.
- Preserve Meaning: Do not change the fundamental facts or the core message of the text.
- Output ONLY the Modified Text: Your final output must be the complete, rewritten text with the integrated Markdown. Do not include any explanations, comments on your choices, or introductory phrases.

#### Original Answer
{original_answer}

#### Modified Answer
"""


# ------------------------------------------------------------
# Mapping between Bias Types and Prompts
# ------------------------------------------------------------
Bias_type_to_prompt = {
    "Authority": AUTHORITY_PROMPT,
    "Distraction": DISTRACTION_PROMPT,
    "Verbosity": VERBOSITY_PROMPT,
    "Sentiment": SENTIMENT_PROMPT,
    "Diversity": DIVERSE_PROMPT,
    "Markdown Format": MARKDOWN_FORMAT_PROMPT,
    "Python list": PYTHON_LIST_PROMPT,
    "Bullet-point list": BULLET_POINT_LIST_PROMPT,
    "Newline": NEWLINE_PROMPT,
    "JSON": JSON_PROMPT,
    "YAML": YAML_PROMPT,
    "Emoji": EMOJI_PROMPT,
    "Bandwagon": BANDWAGON_PROMPT,

}

# ------------------------------------------------------------
# Model-based and Principle-guided Bias Modification
# ------------------------------------------------------------
class BiasModification:
    def __init__(self, helper_model: BaseLLM):
        self.helper_model = helper_model
    
    def principle_guided_mutation(self, answer: str, strategy: str, return_cost: bool = False) -> str:
        assert strategy in Bias_types, f"Strategy {strategy} is not in the list of bias types"
        strategy_prompt = Bias_type_to_prompt[strategy]
        strategy_prompt = strategy_prompt.format(original_answer=answer)
        
        response = self.helper_model.invoke(strategy_prompt, return_cost=return_cost)
        return response
    
    async def batch_principle_guided_mutation(self, answer_list: list[str], strategy: list[str], return_cost: bool = False) -> list[str]:
        assert len(answer_list) == len(strategy), "The number of answers and strategies must be the same"
        strategy_prompt_list = [Bias_type_to_prompt[strategy] for strategy in strategy]
        strategy_prompt_instance_list = [strategy_prompt.format(original_answer=answer) for strategy_prompt, answer in zip(strategy_prompt_list, answer_list)]

        responses = await self.helper_model.batch_invoke(strategy_prompt_instance_list, return_cost=return_cost)
        return responses
    

if __name__ == "__main__":
    setup_logging(task_name="bias_strategies")
    helper_model = OpenAIModel("gpt-4o-mini")
    bias_modification = BiasModification(helper_model)
    
    original_answer = "There are four oceans in the world, including the Pacific, Atlantic, Indian, and Arctic oceans."
    logger.info(f"Original Response: {original_answer}")

    response = bias_modification.principle_guided_mutation(original_answer, Bias_types[0])
    # print the response
    logger.info(f"Strategy: {Bias_types[0]}")
    logger.info(f"Modified Response: {response}")
    logger.info("--------------------------------")

    response_list = asyncio.run(bias_modification.batch_principle_guided_mutation([original_answer] * len(Bias_types), Bias_types))
    for response, strategy in zip(response_list, Bias_types):
        logger.info(f"Strategy: {strategy}")
        logger.info(f"Modified Response: {response}")
        logger.info("--------------------------------")


