"""
This class is used for llm-as-a-judge.

Point-wise scoring:
1. Absolute scoring, e.g. "Evaluate the harmlessness of this response on a scale of 1-9"
"""
import asyncio

from src.judge_prompts import POINTWISE_EVALUATION_PROMPT, PAIRWISE_EVALUATION_PROMPT, ARENA_HARD_AUTO_PROMPT, MT_BENCH_PROMPT, MT_BENCH_SYSTEM_PROMPT, ALPACA_EVAL_SYSTEM_PROMPT, ALPACA_EVAL_PROMPT, ARENA_HARD_AUTO_SYSTEM_PROMPT
from src.llm_zoo import load_model
from src.utils import str2json


class JudgeModel:
    """LLM evaluators"""

    def __init__(self, model_name):
        """
        Initializes the LLM debater with the specified model.

        Args:
            model_name (str): The name of the model to load from HF/API
        """
        self.model_name = model_name
        self.judge_type = "general"
        self.model = load_model(model_name)

    def pointwise_score(self, input_q, response) -> tuple[int, str]:
        """Returns the model's confidence that the summary is its own output."""
        formatted_prompt = POINTWISE_EVALUATION_PROMPT.format(INPUTS=input_q, OUTPUT=response)
        response = self.model.invoke(formatted_prompt)
        try:
            json_response = str2json(response)
            score = int(json_response["score"])
            feedback = json_response["feedback"]
        except:
            score = -1
            feedback = "Error: Failed to parse the response as a JSON object."
            print(f"Error: Failed to parse the response as a JSON object. {response}")
        return score, feedback
    
    def batch_pointwise_score(self, q_list, response_list) -> tuple[list[int], list[str]]:
        """Returns the model's confidence that the summary is its own output."""
        formatted_prompts = [POINTWISE_EVALUATION_PROMPT.format(INPUTS=input_q, OUTPUT=response) for input_q, response in zip(q_list, response_list)]
        responses = asyncio.run(self.model.batch_invoke(formatted_prompts))
        scores = []
        explanations = []
        for response in responses:
            try:
                json_response = str2json(response)
                score = int(json_response["score"])
                explanation = json_response["feedback"]
                scores.append(score)
                explanations.append(explanation)
            except:
                scores.append(-1)
                explanations.append("Error: Failed to parse the response as a JSON object.")
        return scores, explanations
    
    def pairwise_score(self, input_q, response1, response2) -> tuple[int, str]:
        """Returns the model's confidence that the summary is its own output."""
        formatted_prompt = PAIRWISE_EVALUATION_PROMPT.format(INPUTS=input_q, OUTPUT_A=response1, OUTPUT_B=response2)
        response = self.model.invoke(formatted_prompt)
        try:
            json_response = str2json(response)
            better_model = json_response["better_model"]
            feedback = json_response["feedback"]
        except:
            better_model = None
            feedback = "Error: Failed to parse the response as a JSON object."
            print(f"Error: Failed to parse the response as a JSON object. {response}")
        if better_model == "A":
            better_model = 1
        elif better_model == "B":
            better_model = -1
        else:
            better_model = None
        return better_model, feedback
    
    def batch_pairwise_score(self, q_list, response1_list, response2_list) -> tuple[list[int], list[str]]:
        """Returns the model's confidence that the summary is its own output."""
        formatted_prompts = [PAIRWISE_EVALUATION_PROMPT.format(INPUTS=input_q, OUTPUT_A=response1, OUTPUT_B=response2) for input_q, response1, response2 in zip(q_list, response1_list, response2_list)]
        responses = asyncio.run(self.model.batch_invoke(formatted_prompts))
        better_models = []
        feedbacks = []
        for response in responses:
            try:
                json_response = str2json(response)
                better_model = json_response["better_model"]
                feedback = json_response["feedback"]
                better_models.append(better_model)
                feedbacks.append(feedback)
            except:
                better_models.append(-1)
                feedbacks.append("Error: Failed to parse the response as a JSON object.")
        return better_models, feedbacks


class AlpacaEvalModel(JudgeModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.judge_type = "alpaca_eval"
        self.model = load_model(model_name)
    
    def pointwise_score(self, input_q, response) -> tuple[int, str]:
        raise NotImplementedError("AlpacaEvalModel does not support pointwise scoring.")
    
    def batch_pointwise_score(self, q_list, response_list) -> tuple[list[int], list[str]]:
        raise NotImplementedError("AlpacaEvalModel does not support batch pointwise scoring.")
    
    def pairwise_score(self, input_q, response1, response2) -> tuple[int, str]:
        formatted_prompt = ALPACA_EVAL_PROMPT.format(instruction=input_q, output_1=response1, output_2=response2)
        response = self.model.invoke(formatted_prompt, system_prompt=ALPACA_EVAL_SYSTEM_PROMPT)
        if response == "m":
            return 1, "The first response is better."
        elif response == "M":
            return -1, "The second response is better."
        else:
            return None, "Error: Failed to parse the response as a JSON object."
    
    def batch_pairwise_score(self, q_list, response1_list, response2_list) -> tuple[list[int], list[str]]:
        formatted_prompts = [ALPACA_EVAL_PROMPT.format(instruction=input_q, output_a=response1, output_b=response2) for input_q, response1, response2 in zip(q_list, response1_list, response2_list)]
        responses = asyncio.run(self.model.batch_invoke(formatted_prompts, system_prompt=ALPACA_EVAL_SYSTEM_PROMPT))
        better_models = []
        feedbacks = []
        for response in responses:
            if response == "m":
                better_models.append(1)
            elif response == "M":
                better_models.append(-1)
            else:
                better_models.append(None)
        return better_models, feedbacks
    
import re
def get_arena_hard_score(judge_output):
    # Define the possible verdict patterns
    patterns = {
        r'\[\[A>>B\]\]': 2,  # A significantly better
        r'\[\[A>B\]\]': 1,   # A slightly better
        r'\[\[A=B\]\]': 0,   # Tie
        r'\[\[B>A\]\]': -1,  # B slightly better
        r'\[\[B>>A\]\]': -2  # B significantly better
    }
    
    # Search for any of the patterns in the output
    for pattern, score in patterns.items():
        if re.search(pattern, judge_output):
            return score
    
    return None  # No valid verdict found

class ArenaHardAutoModel(JudgeModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.judge_type = "arena_hard_auto"
        self.model = load_model(model_name)
    
    def pairwise_score(self, input_q, response1, response2) -> tuple[int, str]:
        '''Options:
        1. Assistant A is significantly better: [[A>>B]]
        2. Assistant A is slightly better: [[A>B]]
        3. Tie, relatively the same: [[A=B]]
        4. Assistant B is slightly better: [[B>A]]
        5. Assistant B is significantly better: [[B>>A]]
        '''
        formatted_prompt = ARENA_HARD_AUTO_PROMPT.format(question=input_q, answer_a=response1, answer_b=response2)
        response = self.model.invoke(formatted_prompt, system_prompt=ARENA_HARD_AUTO_SYSTEM_PROMPT)
        outcome = get_arena_hard_score(response)
        return outcome, response
    
    def batch_pairwise_score(self, q_list, response1_list, response2_list) -> tuple[list[int], list[str]]:
        formatted_prompts = [ARENA_HARD_AUTO_PROMPT.format(input_q, response1, response2) for input_q, response1, response2 in zip(q_list, response1_list, response2_list)]
        responses = asyncio.run(self.model.batch_invoke(formatted_prompts))
        better_models = []
        for response in responses:
            outcome = get_arena_hard_score(response)
            better_models.append(outcome)
        return better_models, responses
    
    def pointwise_score(self, input_q, response) -> tuple[int, str]:
        raise NotImplementedError("ArenaHardAutoModel does not support pointwise scoring.")
    
    def batch_pointwise_score(self, q_list, response_list) -> tuple[list[int], list[str]]:
        raise NotImplementedError("ArenaHardAutoModel does not support batch pointwise scoring.")

def get_mt_bench_score(judge_output):
    # Define the possible verdict patterns
    patterns = {
        r'\[\[1\]\]': 1,  
        r'\[\[2\]\]': 2,   
        r'\[\[3\]\]': 3,   
        r'\[\[4\]\]': 4,
        r'\[\[5\]\]': 5,
        r'\[\[6\]\]': 6,
        r'\[\[7\]\]': 7,
        r'\[\[8\]\]': 8,
        r'\[\[9\]\]': 9,
        r'\[\[10\]\]': 10,
    }
    
    # Search for any of the patterns in the output
    for pattern, score in patterns.items():
        if re.search(pattern, judge_output):
            return int(score)
    
    return None  # No valid verdict found

class MTBenchModel(JudgeModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.judge_type = "mt_bench"
        self.model = load_model(model_name)

    def pairwise_score(self, input_q, response1, response2) -> tuple[int, str]:
        raise NotImplementedError("MTBenchModel does not support pairwise scoring.")
    
    def batch_pairwise_score(self, q_list, response1_list, response2_list) -> tuple[list[int], list[str]]:
        raise NotImplementedError("MTBenchModel does not support batch pairwise scoring.")
    
    def pointwise_score(self, input_q, response) -> tuple[int, str]:
        formatted_prompt = MT_BENCH_PROMPT.format(question=input_q, answer=response)
        response = self.model.invoke(formatted_prompt, system_prompt=MT_BENCH_SYSTEM_PROMPT)
        outcome = get_mt_bench_score(response)
        return outcome, response
    
    def batch_pointwise_score(self, q_list, response_list) -> tuple[list[int], list[str]]:
        formatted_prompts = [MT_BENCH_PROMPT.format(question=input_q, answer=response) for input_q, response in zip(q_list, response_list)]
        responses = asyncio.run(self.model.batch_invoke(formatted_prompts, system_prompt=MT_BENCH_SYSTEM_PROMPT))
        scores = []
        for response in responses:
            outcome = get_mt_bench_score(response)
            scores.append(outcome)
        return scores, responses


def load_judge_model(judge_model_name, judge_model_backbone):
    if judge_model_name == "general":
        return JudgeModel(judge_model_backbone)
    elif judge_model_name == "arena_hard_auto":
        return ArenaHardAutoModel(judge_model_backbone)
    elif judge_model_name == "mt_bench":
        return MTBenchModel(judge_model_backbone)
    elif judge_model_name == "alpaca_eval":
        return AlpacaEvalModel(judge_model_backbone)
    else:
        raise ValueError(f"Invalid judge model name: {judge_model_name}")
    

if __name__ == "__main__":

    question = "I'm curious about how renewable energy technologies might transform urban infrastructure in the next decade. What potential innovations do you see emerging?"
    target_response_1 = "Renewable energy technologies are poised to revolutionize urban landscapes, with advanced solar roadways potentially generating electricity while supporting vehicle traffic. Vertical wind turbines integrated into skyscrapers could become more efficient, potentially producing up to 30% of a building's energy needs. While some experts are skeptical about total urban energy independence, emerging technologies like transparent solar panels and bio-responsive energy systems suggest we're approaching a transformative period in sustainable urban design."
    target_response_2 = "ChatGPT: Sustainable urban design is a critical component of urban planning, aiming to create cities that are environmentally friendly, socially equitable, and economically viable. This approach considers the long-term impact of urban development on the natural environment, human well-being, and economic prosperity. Key principles include reducing carbon emissions, conserving water resources, promoting biodiversity, and ensuring access to green spaces. By integrating sustainable practices into urban design, cities can achieve a balance between economic growth and environmental sustainability, fostering a more resilient and livable urban environment."

    # ------------------------------------------------------------
    # Test Pointwise Model
    # ------------------------------------------------------------

    # gpt = JudgeModel("gpt-4o-2024-05-13")
    # gemini = JudgeModel("gemini-1.5-flash")

    # s1, e1 = gpt.pointwise_score(question, target_response_1)
    # s2, e2 = gemini.pointwise_score(question, target_response_2)
    # better_model, feedback = gpt.pairwise_score(question, target_response_1, target_response_2)

    # print(f"GPT score: {s1}, feedback: {e1}")
    # print(f"Gemini score: {s2}, feedback: {e2}")
    # print(f"GPT pairwise score: {better_model}, feedback: {feedback}")

    # ------------------------------------------------------------
    # Test Pairwise Model
    # ------------------------------------------------------------
    alpaca_eval_judge = AlpacaEvalModel("gemini-2.0-flash")

    s1, e1 = alpaca_eval_judge.pairwise_score(question, target_response_1, target_response_2)
    print(f"AlpacaEval score given by gemini-2.0-flash: {s1}, feedback: {e1}")

    arena_hard_auto_judge = ArenaHardAutoModel("gemini-2.0-flash")
    s1, e1 = arena_hard_auto_judge.pairwise_score(question, target_response_1, target_response_2)
    print(f"ArenaHardAuto score given by gemini-2.0-flash: {s1}, feedback: {e1}")

    mt_bench_judge = MTBenchModel("gemini-2.0-flash")
    s1, e1 = mt_bench_judge.pointwise_score(question, target_response_1)
    print(f"MTBench score given by gemini-2.0-flash: {s1}, feedback: {e1}")


    