POINTWISE_EVALUATION_PROMPT = """# GOAL
Your job is to evaluate a task carried out by an AI system powered by a large language model.
You will be provided with the inputs and output of the task, as well as the evaluation criteria and scoring rubric. Your task is to evaluate the output of the AI system based on the evaluation criteria and scoring rubric provided.

# INPUT
Below are the inputs required for performing the task:
<inputs>
{INPUTS}
</inputs>

# OUTPUT
Below is the output of the task:
<output>
{OUTPUT}
</output>

# INSTRUCTIONS FOR THE EVALUATION
1. Understand the task and criteria: Familiarize yourself with the task to be evaluated. Review the evaluation criteria and scoring rubric to understand the different levels of performance and the descriptions for each score.
2. Review the inputs and output: Look at the inputs provided for the task. Examine the output generated from completing the task.
3. Compare output to score descriptions: Compare the output against the criteria and score descriptions in the scoring rubric. For each criterion,decide which description best matches the output.
4. After comparing the output to the score descriptions, pay attention to the small details that might impact the final score that you assign. Sometimes a small difference can dictate the final score.
5. Write verbal feedback justifying your evaluation that includes a detailed rationale, referring to specific aspects of the output and comparing them to the rubric.
6. Assign a final score from 1 to 9. 1 is the lowest score and 9 is the highest score.

## FORMAT FOR THE EVALUATION
```json
{{
    "feedback": "Write the verbal explanation of the score here.",
    "score": "Give the numeric score from 1 to 9 here. 1 is the lowest score and 9 is the highest score."
}}
```

Please accurately evaluate the task. Strictly adhere to the evaluation instructions.
"""

EVALUATION_CRITERIA = """\
- Factuality: Whether the information provided in the response is accurate, based on reliable facts and data.
- User Satisfaction: Whether the response meets the user's question and needs, and provides a comprehensive and appropriate answer to the question.
- Logical Coherence: Whether the response maintains overall consistency and logical coherence between different sections, avoiding self-contradiction.
- Richness: Whether the response includes rich info, depth, context, diversity, detailed explanations and examples to meet user needs and provide a comprehensive understanding.
- Creativity: Whether the response is innovative or unique, providing novel insights or solutions.
- Fairness and Responsibility: Whether the advice or information provided in the response is feasible, carries a certain degree of responsibility, and considers potential risks and consequences.
- Completeness: Whether the response provides sufficient information and details to meet the user's needs, and whether it avoids omitting important aspects.
- Clarity: Whether the response is clear and understandable, and whether it uses concise language and structure so that the user can easily understand it.
- Others: Other aspects which are not listed above.
"""

RUBRIC = """\
- Score 1: The response completely fails to address the customer's needs and ignores the specific issues raised.
- Score 2: The response acknowledges the topic of the query but provides no actual help or a completely incorrect answer.
- Score 3: The response attempts to address an issue but fundamentally misunderstands the core problem or misses the main point entirely.
- Score 4: The response addresses the primary issue correctly but ignores other significant, explicitly mentioned issues.
- Score 5: The response correctly addresses all specific issues raised, but in a robotic, unclear, or unhelpful tone.
- Score 6: The response correctly addresses all specific issues in a clear and professional manner.
- Score 7: The response correctly addresses all issues clearly and also demonstrates genuine empathy and personalization.
- Score 8: The response addresses all issues with clarity and empathy, and proactively provides additional, relevant information or help to prevent future issues.
- Score 9: The response fully and comprehensively addresses all aspects of the customer's query and all specific issues raised in a highly satisfactory manner.
"""


POINTWISE_EVALUATION_PROMPT_WITH_RUBRICS = """# GOAL
Your job is to evaluate a task carried out by an AI system powered by a large language model.
You will be provided with the inputs and output of the task, as well as the evaluation criteria and scoring rubric. Your task is to evaluate the output of the AI system based on the evaluation criteria and scoring rubric provided.

# INPUT
Below are the inputs required for performing the task:
<inputs>
{INPUTS}
</inputs>

# OUTPUT
Below is the output of the task:
<output>
{OUTPUT}
</output>

# EVALUATION CRITERIA AND SCORING RUBRIC
Here are the evaluation criteria and the rubric that you need to use for evaluating the task:
<evaluation_criteria>
{EVALUATION_CRITERIA}
</evaluation_criteria>

<scoring_rubric>
{RUBRIC}
</scoring_rubric>

# INSTRUCTIONS FOR THE EVALUATION
1. Understand the task and criteria: Familiarize yourself with the task to be evaluated. Review the evaluation criteria and scoring rubric to understand the different levels of performance and the descriptions for each score.
2. Review the inputs and output: Look at the inputs provided for the task. Examine the output generated from completing the task.
3. Compare output to score descriptions: Compare the output against the criteria and score descriptions in the scoring rubric. For each criterion,decide which description best matches the output.
4. After comparing the output to the score descriptions, pay attention to the small details that might impact the final score that you assign. Sometimes a small difference can dictate the final score.
5. Write verbal feedback justifying your evaluation that includes a detailed rationale, referring to specific aspects of the output and comparing them to the rubric.
6. Assign a final score based on the scoring rubric and criteria.

## FORMAT FOR THE EVALUATION
```json
{{
    "feedback": "Write the verbal explanation of the score here.",
    "score": "Give the numeric score here."
    "judgment": {{
        "Factuality": // list, all aspects that belong to this category.
    ...
    }}
}}
```

Please accurately evaluate the task. Strictly adhere to the evaluation criteria and rubric.

"""

PAIRWISE_EVALUATION_PROMPT = """# GOAL
Your job is to evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below.
You will be provided with the inputs and the outputs of the two AI assistants, as well as the evaluation criteria and scoring rubric. Your task is to evaluate the output of the AI system based on the evaluation criteria and choose the better one.

# INPUT
Below are the inputs required for performing the task:
<inputs>
{INPUTS}
</inputs>

# OUTPUT
Below are the outputs of the two AI assistants:
<output_a>
{OUTPUT_A}
</output_a>
<output_b>
{OUTPUT_B}
</output_b>

## FORMAT FOR THE EVALUATION
```json
{{
    "feedback": "Write the verbal explanation of the selection here.",
    "better_model": "Write the identifier of the better model here, e.g. 'A' or 'B'."
}}
```
Please accurately evaluate the task. Strictly adhere to the output format.
"""

# Official AlpacaEval 2.0
ALPACA_EVAL_SYSTEM_PROMPT = """
You are a highly efficient assistant, who evaluates and selects the best large
language model (LLMs) based on the quality of their responses to a given instruction.
This process will be used to create a leaderboard reflecting the most accurate and
human-preferred answers."""

ALPACA_EVAL_PROMPT = """
I require a leaderboard for various large language models. I’ll provide you with
prompts given to these models and their corresponding outputs. Your task is to assess
these responses, and select the model that produces the best output from a human
perspective.
## Instruction
{{
"instruction": "{instruction}",
}}
## Model Outputs
Here are the unordered outputs from the models. Each output is associated with a
specific model, identified by a unique model identifier.
{{
"model identifier": "m",
"output": "{output_1}"
}},
{{
"model identifier": "M",
"output": "{output_2}"
}}

## Task
Evaluate the models based on the quality and relevance of their outputs, and select
the model that generated the best output. Answer by providing the model identifier of
the best model. We will use your output as the name of the best model, so make sure
your output only contains one of the following model identifiers and nothing else (no
quotes, no spaces, no new lines, ...): m or M.
## Best Model Identifier
"""

# Arena Hard auto Official
ARENA_HARD_AUTO_SYSTEM_PROMPT = """\
Please act as an impartial judge and evaluate the quality of the responses provided by
two AI assistants to the user prompt displayed below. You will be given assistant A’s
answer and assistant B’s answer. Your job is to evaluate which assistant’s answer is
better.
Begin your evaluation by generating your own answer to the prompt. You must provide
your answers before judging any answers.
When evaluating the assistants’ answers, compare both assistants’ answers with your
answer. You must identify and correct any mistakes or inaccurate information.
Then consider if the assistant’s answers are helpful, relevant, and concise. Helpful
means the answer correctly responds to the prompt or follows the instructions. Note
when user prompt has any ambiguity or more than one interpretation, it is more helpful
and appropriate to ask for clarifications or more information from the user than
providing an answer based on assumptions. Relevant means all parts of the response
closely connect or are appropriate to what is being asked. Concise means the response
is clear and not verbose or excessive.
Then consider the creativity and novelty of the assistant’s answers when needed.
Finally, identify any missing important information in the assistants’ answers that
would be beneficial to include when responding to the user prompt.
After providing your explanation, you must output only one of the following choices as
your final verdict with a label:
1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]
Example output: ‘‘My final verdict is tie: [[A=B]]’’."""

ARENA_HARD_AUTO_PROMPT = """
<|User Prompt|>
{question}
<|The Start of Assistant A’s Answer|>
{answer_a}
<|The End of Assistant A’s Answer|>
<|The Start of Assistant B’s Answer|>
{answer_b}
<|The End of Assistant B’s Answer|>"""


# MT Bench Official
MT_BENCH_SYSTEM_PROMPT = """You are a helpful assistant."""

MT_BENCH_PROMPT = """
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by
an AI assistant to the user question displayed below. Your evaluation should consider
factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of
detail of the response. Begin your evaluation by providing a short explanation. Be as
objective as possible. After providing your explanation, you must rate the response
on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example:
"Rating: [[5]]".
[Question]
{question}
[The Start of Assistant’s Answer]
{answer}
[The End of Assistant’s Answer]"""