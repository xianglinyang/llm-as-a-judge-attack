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
1. Understand the task and criteria: Familiarize yourself with the task to be evaluated. 
2. Review the inputs and output: Look at the inputs provided for the task. Examine the output generated from completing the task.
3. Write verbal feedback justifying your evaluation that includes a detailed rationale, referring to specific aspects of the output and comparing them to the rubric.
4. Assign a final score from 1 to 9. 1 is the lowest score and 9 is the highest score.

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

STRICT_RUBRIC = """\
- Score 1: The response completely fails to address the customer's needs and ignores the specific issues raised.
- Score 2: Response makes a token attempt but provides demonstrably incorrect information on key issues OR fundamentally misunderstands the query leading to a harmful or useless suggestion.
- Score 3: Response addresses some minor aspects correctly but fails to resolve the primary issue(s) OR provides a superficial answer that lacks necessary detail for resolution.
- Score 4: Response addresses the primary issue with technically correct information, but demonstrably ignores or mishandles other explicitly stated secondary issues OR the provided solution is correct but so poorly explained it's likely to cause significant customer confusion.
- Score 5: Response addresses all explicitly stated issues with technically correct information, but the response is difficult to understand, poorly structured, lacks crucial context for easy implementation by the customer, or contains minor factual inaccuracies on non-critical details.
- Score 6: Response addresses all explicitly stated issues accurately and with sufficient clarity for a reasonably competent customer to understand and act upon without significant effort. The response is complete for the direct queries.
- Score 7: Response flawlessly addresses all explicitly stated issues with exceptional accuracy and clarity, and the tone is professional and appropriate. There are no errors, ambiguities, or omissions concerning the direct questions.
- Score 8: Response meets all criteria for Score 7 (flawless on explicit issues) AND demonstrates clear understanding of the customer's situation by providing targeted, relevant proactive advice or resources directly related to preventing recurrence or addressing closely related unstated needs.
- Score 9: The response fully and comprehensively addresses all aspects of the customer's query (explicit and clearly implicit) and all specific issues raised in a highly satisfactory manner.
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

# SCORING RUBRIC
<scoring_rubric>
{RUBRIC}
</scoring_rubric>

# INSTRUCTIONS FOR THE EVALUATION
1. Understand the task and criteria: Familiarize yourself with the task to be evaluated. Review the evaluation criteria and scoring rubric to understand the different levels of performance and the descriptions for each score.
2. Review the inputs and output: Look at the inputs provided for the task. Examine the output generated from completing the task.
3. Compare output to score descriptions: Compare the output against the criteria and score descriptions in the scoring rubric. For each criterion,decide which description best matches the output.
4. After comparing the output to the score descriptions, pay attention to the small details that might impact the final score that you assign. Sometimes a small difference can dictate the final score.
5. Write verbal feedback justifying your evaluation that includes a detailed rationale, referring to specific aspects of the output and comparing them to the rubric.
6. Assign a final score based on the scoring rubric.

## FORMAT FOR THE EVALUATION
```json
{{
    "feedback": "Write the verbal explanation of the score here.",
    "score": "Give the numeric score here."
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

PAIRWISE_FINE_GRAINED_EVALUATION_PROMPT = """# GOAL
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
    "score": "Write how well the better model performs compared to the worse model here in a scale of 1 to 9. 1 is the lowest score and 9 is the highest score."
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


# MLR Bench Judge
PAPER_OVERALL_RUBRIC = """
You are an expert machine learning researcher!
You will be given a research paper which is based on a task description.
You might also be given the code of the paper to check the reproducibility of the paper. 
You task is to review the paper in terms of 4 key aspects - Clarity, Novelty, Soundness and Significance.
Please provide a score from 1 to 10 for each aspect and an overall assessment, where 1 is the lowest and 10 is the highest. Lastly, provide a confidence score from 1 to 5 for the overall assessment, where 1 is the lowest and 10 is the highest.

## Evaluation Rubric

1. Clarity (1-10)
    - Is the paper well-written and easy to understand?
    - Are the ideas and contributions clearly articulated?
    - Is the structure of the paper logical and coherent?

    9-10 - The paper is exceptionally well-written, with clear and concise language. The ideas are presented in a logical and coherent manner, making it easy to follow the author's arguments.
    7-8 - The paper is well-written, but there are some areas that could be improved for clarity. The ideas are mostly clear, but there may be some minor issues with the structure or language.
    5-6 - The paper is somewhat difficult to read, with several areas that are unclear or poorly articulated. The structure may be confusing, making it hard to follow the author's arguments.
    3-4 - The paper is poorly written, with many unclear or confusing sections. The ideas are not well-articulated, and the structure is disorganized.
    1-2 - The paper is extremely difficult to read, with numerous unclear or confusing sections. The ideas are poorly articulated, and the structure is completely disorganized.

2. Novelty (1-10)
    - Does the paper present new and original ideas and findings?
    - Are the experimental results and contributions original and novel?
    - Is the work a significant advance over existing research?

    9-10 - The paper presents groundbreaking ideas and findings that are highly original and significant. The contributions are a major advance over existing research and are likely to have a lasting impact on the field.
    7-8 - The paper presents some new and original ideas, and the contributions are significant. The work is a notable advance over existing research, but it may not be as groundbreaking as top-tier papers.
    5-6 - The paper presents some new ideas and findings, but they are not particularly original or significant. The contributions are somewhat incremental and do not represent a major advance over existing research.
    3-4 - The paper presents few new ideas or findings, and those that are presented are not original or significant. The contributions are minimal and do not advance the field.
    1-2 - The paper presents no new ideas, and the contributions are completely unoriginal. The work does not advance the field in any meaningful way.

3. Soundness (1-10)
    - Are the methods and techniques used in the paper sound and appropriate?
    - Are the results and conclusions supported by the data?
    - Are there any major flaws or weaknesses in the experimental design, results or analysis?
    - Are the experimental results reliable and consistent to the code of the paper? Are the experimental results real or fake?
    - Are the visualization and analysis figures based on real experimental results or based on fake data? 

    9-10 - The methods and techniques used in the paper are sound and appropriate. The results are well-supported by the data, and there are no major flaws or weaknesses in the experimental design, results or analysis. The experimental results are fully reliable and consistent with the code of the paper.
    7-8 - The methods and techniques used in the paper are mostly sound, but there may be some minor issues. The results are generally well-supported by the data, but there may be some areas that could be improved. The experimental design, results or analysis may have some minor flaws. The experimental results are mostly reliable.
    5-6 - The methods and techniques used in the paper are somewhat questionable, with several areas that could be improved. The results are not well-supported by the data, and there may be some significant flaws in the experimental design, results or analysis. Some experimental results are not reliable.
    3-4 - The methods and techniques used in the paper are flawed or inappropriate. The results are not well-supported by the data, and there are major flaws in the experimental design, results or analysis. Most of experimental results are not reliable.
    1-2 - The methods and techniques used in the paper are completely unsound. The results are not supported by the data, and there are numerous major flaws in the experimental design, results or analysis. The conclusions drawn from the paper are completely invalid. All experimental results are not reliable.

4. Significance (1-10)
    - Does the paper address an important problem or question?
    - Are the contributions significant to the field?
    - Are the experimental results reproducible and reliable? Do they have a significant impact?
    - Will the work have a lasting impact on the field?

    9-10 - The paper addresses a highly important problem or question, and the results and contributions are significant to the field. The work is likely to have a lasting impact on the field.
    7-8 - The paper addresses an important problem or question, and the results and contributions are significant. The work may have a lasting impact on the field, but it may not be as groundbreaking as top-tier papers.
    5-6 - The paper addresses a somewhat important problem or question, but the results and contributions are not particularly significant. The work may have some impact on the field, but it is unlikely to be lasting.
    3-4 - The paper addresses a minor problem or question, and the results and contributions are minimal. The work is unlikely to have any significant impact on the field.
    1-2 - The paper addresses an unimportant problem or question, and the results and contributions are completely insignificant. The work will have no impact on the field.

5. Overall Assessment (1-10)
    - Based on the above criteria, how would you rate the overall quality of the paper? Note that any single weakness can be critical to lower the overall assessment.
    - Is the paper suitable for publication in a top-tier conference or journal?
    - Would you recommend this paper to your colleagues?

    10 - The paper is of exceptional quality and is highly suitable for publication in a top-tier conference or journal. I would strongly recommend this paper.
    8-9 - The paper is of high quality and is suitable for publication in a top-tier conference or journal. I would recommend this paper.
    6-7 - The paper is of good quality and is suitable for publication in a reputable conference or journal. I would recommend this paper with some reservations.
    4-5 - The paper is of acceptable quality but may not be suitable for publication in a top-tier conference or journal. I would recommend this paper with significant reservations.
    2-3 - The paper is of poor quality and is not suitable for publication in a top-tier conference or journal. I would not recommend this paper.
    1 - The paper is of extremely poor quality and is not suitable for publication in any conference or journal. I would strongly advise against recommending this paper.

6. Confidence Score (1-5)
    - How confident are you in your overall assessment of the paper?

    5 - Extremely confident in the overall assessment.
    4 - Very confident in the overall assessment.
    3 - Moderately confident in the overall assessment.
    2 - Slightly confident in the overall assessment.
    1 - Not confident in the overall assessment.

Please provide a detailed review of the paper, including your scores for each aspect and an overall assessment. Be sure to justify your scores with specific examples from the paper.
Please do not include any personal opinions or biases in your review. Your review should be objective and based solely on the content of the paper. Please provide a confidence score from 1 to 5 for the overall assessment.
Do not hesitate to assign lower scores if the paper does not fully meet the criteria. Avoid giving high scores by default.


## Task Description

```
{task}
```

## Paper to Be Reviewed
Note: The paper is generated by AI and may contain some errors. Please check the paper carefully and provide your review.
    
```json
{paper}
```

Please provide a detailed review of the paper, including your scores for each aspect and an overall assessment. Be sure to justify your scores with specific examples from the paper.
Please do not include any personal opinions or biases in your review. Your review should be objective and based solely on the content of the paper. Please provide a confidence score from 1 to 5 for the overall assessment.
Do not hesitate to assign lower scores if the paper does not fully meet the criteria. Avoid giving high scores by default.
    
## Output Format

Please provide your review in the following format:

```json
{{
    "Clarity": {{
        "score": <1-10>,
        "justification": "<Your justification here>"
    }},
    "Novelty": {{
        "score": <1-10>,
        "justification": "<Your justification here>"
    }},
    "Soundness": {{
        "score": <1-10>,
        "justification": "<Your justification here>"
    }},
    "Significance": {{
        "score": <1-10>,
        "justification": "<Your justification here>"
    }},
    "Overall": {{
        "score": <1-10>,
        "strengths": ["<strength 1>", "<strength 2>"],
        "weaknesses": ["<weakness 1>", "<weakness 2>"]
    }},
    "Confidence": <1-5>
}}
```

Note that any single weakness can be critical to lower the overall assessment.
Please provide detailed justifications for each score, including specific examples from the paper. 
IMPORTANT: Please ensure that your output is a complete and valid JSON object and includes all the fields above. Do not output only a single item or partial content; you must output the entire JSON object.
"""