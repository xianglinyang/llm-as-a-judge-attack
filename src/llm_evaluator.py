"""
This class is used for llm-as-a-judge.

Point-wise scoring:
1. Absolute scoring, e.g. "Evaluate the harmlessness of this response on a scale of 1-9"
"""
from enum import Enum
from abc import abstractmethod, ABC
import re

from src.judge_prompts import POINTWISE_EVALUATION_PROMPT, POINTWISE_EVALUATION_PROMPT_WITH_RUBRICS, PAIRWISE_EVALUATION_PROMPT, ARENA_HARD_AUTO_PROMPT, MT_BENCH_PROMPT, MT_BENCH_SYSTEM_PROMPT, ALPACA_EVAL_SYSTEM_PROMPT, ALPACA_EVAL_PROMPT, ARENA_HARD_AUTO_SYSTEM_PROMPT, RUBRIC, EVALUATION_CRITERIA
from src.llm_zoo import load_model
from src.utils import str2json

# ------------------------------------------------------------
# Judge Type
# ------------------------------------------------------------
class JudgeType(Enum):
    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"
    ALPACA_EVAL = "alpaca_eval"
    ARENA_HARD_AUTO = "arena_hard_auto"
    MT_BENCH = "mt_bench"

# ------------------------------------------------------------
# Judge Model
# ------------------------------------------------------------
class JudgeModelABC(ABC):
    def __init__(self, judge_type: JudgeType, judge_model_backbone: str):
        self.judge_type = judge_type
        self.judge_model_backbone = judge_model_backbone
        self.model = load_model(judge_model_backbone) 

    @abstractmethod
    def get_score(self):
        pass

    @abstractmethod
    async def batch_get_score(self):
        pass

class PointwiseJudgeModel(JudgeModelABC):
    def __init__(self, judge_type: JudgeType, judge_model_backbone: str):
        super().__init__(judge_type, judge_model_backbone)

    def get_score(self, input_q, response) -> tuple[int, str]:
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
    
    async def batch_get_score(self, q_list, response_list) -> tuple[list[int], list[str]]:
        """Returns the model's confidence that the summary is its own output."""
        formatted_prompts = [POINTWISE_EVALUATION_PROMPT.format(INPUTS=input_q, OUTPUT=response) for input_q, response in zip(q_list, response_list)]
        responses = await self.model.batch_invoke(formatted_prompts)
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

def get_pairwise_score(response):
    '''
    Options:
    1. A is better than B
    2. B is better than A
    3. Tie
    '''
    if response == "A":
        return 1
    elif response == "B":
        return -1
    else:
        return -2

class PairwiseJudgeModel(JudgeModelABC):
    def __init__(self, judge_type: JudgeType, judge_model_backbone: str):
        super().__init__(judge_type, judge_model_backbone)

    def get_score(self, input_q, response1, response2) -> tuple[int, str]:
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

        score = get_pairwise_score(better_model)
        return score, feedback
        
    
    async def batch_get_score(self, q_list, response1_list, response2_list) -> tuple[list[int], list[str]]:
        """Returns the model's confidence that the summary is its own output."""
        formatted_prompts = [PAIRWISE_EVALUATION_PROMPT.format(INPUTS=input_q, OUTPUT_A=response1, OUTPUT_B=response2) for input_q, response1, response2 in zip(q_list, response1_list, response2_list)]
        responses = await self.model.batch_invoke(formatted_prompts)
        better_models = []
        feedbacks = []
        for response in responses:
            try:
                json_response = str2json(response)
                better_model = json_response["better_model"]
                feedback = json_response["feedback"]
                score = get_pairwise_score(better_model)
                better_models.append(score)
                feedbacks.append(feedback)
            except:
                better_models.append(-2)
                feedbacks.append("Error: Failed to parse the response as a JSON object.")
        return better_models, feedbacks

def get_alpaca_eval_score(response):
    if response == "m":
        return 1, "The first response is better."
    elif response == "M":
        return -1, "The second response is better."
    else:
        return -2, "Error: Failed to parse the response as a JSON object."

class AlpacaEvalModel(JudgeModelABC):
    def __init__(self, judge_type: JudgeType, judge_model_backbone: str):
        super().__init__(judge_type, judge_model_backbone)
    
    def get_score(self, input_q, response1, response2) -> tuple[int, str]:
        """Returns the model's confidence that the summary is its own output."""
        formatted_prompt = ALPACA_EVAL_PROMPT.format(instruction=input_q, output_1=response1, output_2=response2)
        response = self.model.invoke(formatted_prompt, system_prompt=ALPACA_EVAL_SYSTEM_PROMPT)
        score, feedback = get_alpaca_eval_score(response)
        return score, feedback

    async def batch_get_score(self, q_list, response1_list, response2_list) -> tuple[list[int], list[str]]:
        formatted_prompts = [ALPACA_EVAL_PROMPT.format(instruction=input_q, output_1=response1, output_2=response2) for input_q, response1, response2 in zip(q_list, response1_list, response2_list)]
        responses = await self.model.batch_invoke(formatted_prompts, system_prompt=ALPACA_EVAL_SYSTEM_PROMPT)
        scores = []
        feedbacks = []
        for response in responses:
            score, feedback = get_alpaca_eval_score(response)
            scores.append(score)
            feedbacks.append(feedback)
        return scores, feedbacks
    
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
            return score, judge_output
    
    return -2, "Error: Failed to parse the response as a JSON object."

class ArenaHardAutoModel(JudgeModelABC):
    def __init__(self, judge_type: JudgeType, judge_model_backbone: str):
        super().__init__(judge_type, judge_model_backbone)
    
    def get_score(self, input_q, response1, response2) -> tuple[int, str]:
        '''Options:
        1. Assistant A is significantly better: [[A>>B]]
        2. Assistant A is slightly better: [[A>B]]
        3. Tie, relatively the same: [[A=B]]
        4. Assistant B is slightly better: [[B>A]]
        5. Assistant B is significantly better: [[B>>A]]
        '''
        formatted_prompt = ARENA_HARD_AUTO_PROMPT.format(question=input_q, answer_a=response1, answer_b=response2)
        response = self.model.invoke(formatted_prompt, system_prompt=ARENA_HARD_AUTO_SYSTEM_PROMPT)
        score, feedback = get_arena_hard_score(response)
        return score, feedback
    
    async def batch_get_score(self, q_list, response1_list, response2_list) -> tuple[list[int], list[str]]:
        formatted_prompts = [ARENA_HARD_AUTO_PROMPT.format(input_q, response1, response2) for input_q, response1, response2 in zip(q_list, response1_list, response2_list)]
        responses = await self.model.batch_invoke(formatted_prompts)
        scores = []
        feedbacks = []
        for response in responses:
            score, feedback = get_arena_hard_score(response)
            scores.append(score)
            feedbacks.append(feedback)
        return scores, feedbacks

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
    
    return -1  # No valid verdict found

class MTBenchModel(JudgeModelABC):
    def __init__(self, judge_type: JudgeType, judge_model_backbone: str):
        super().__init__(judge_type, judge_model_backbone)

    def get_score(self, input_q, response) -> tuple[int, str]:
        formatted_prompt = MT_BENCH_PROMPT.format(question=input_q, answer=response)
        response = self.model.invoke(formatted_prompt, system_prompt=MT_BENCH_SYSTEM_PROMPT)
        score = get_mt_bench_score(response)
        return score, response
    
    async def batch_pointwise_score(self, q_list, response_list) -> tuple[list[int], list[str]]:
        formatted_prompts = [MT_BENCH_PROMPT.format(question=input_q, answer=response) for input_q, response in zip(q_list, response_list)]
        responses = await self.model.batch_invoke(formatted_prompts, system_prompt=MT_BENCH_SYSTEM_PROMPT)
        scores = []
        for response in responses:
            outcome = get_mt_bench_score(response)
            scores.append(outcome)
        return scores, responses


def load_judge_model(judge_type, judge_model_backbone):
    if judge_type == JudgeType.POINTWISE:
        return PointwiseJudgeModel(judge_type, judge_model_backbone)
    elif judge_type == JudgeType.PAIRWISE:
        return PairwiseJudgeModel(judge_type, judge_model_backbone)
    elif judge_type == JudgeType.MT_BENCH:
        return MTBenchModel(judge_type, judge_model_backbone)
    elif judge_type == JudgeType.ALPACA_EVAL:
        return AlpacaEvalModel(judge_type, judge_model_backbone)
    elif judge_type == JudgeType.ARENA_HARD_AUTO:
        return ArenaHardAutoModel(judge_type, judge_model_backbone)
    else:
        raise ValueError(f"Invalid judge type: {judge_type}")
    

if __name__ == "__main__":

    question = "I'm curious about how renewable energy technologies might transform urban infrastructure in the next decade. What potential innovations do you see emerging?"
    target_response_1 = "Renewable energy technologies are poised to revolutionize urban landscapes, with advanced solar roadways potentially generating electricity while supporting vehicle traffic. Vertical wind turbines integrated into skyscrapers could become more efficient, potentially producing up to 30% of a building's energy needs. While some experts are skeptical about total urban energy independence, emerging technologies like transparent solar panels and bio-responsive energy systems suggest we're approaching a transformative period in sustainable urban design."
    target_response_2 = "ChatGPT: Sustainable urban design is a critical component of urban planning, aiming to create cities that are environmentally friendly, socially equitable, and economically viable. This approach considers the long-term impact of urban development on the natural environment, human well-being, and economic prosperity. Key principles include reducing carbon emissions, conserving water resources, promoting biodiversity, and ensuring access to green spaces. By integrating sustainable practices into urban design, cities can achieve a balance between economic growth and environmental sustainability, fostering a more resilient and livable urban environment."

    # ------------------------------------------------------------
    # Test Pointwise Model
    # ------------------------------------------------------------
    gpt = load_judge_model(JudgeType.POINTWISE, "gpt-4.1")
    gemini = load_judge_model(JudgeType.POINTWISE, "gemini-1.5-flash")

    s1, e1 = gpt.get_score(question, target_response_1)
    s2, e2 = gemini.get_score(question, target_response_2)
    print(f"GPT score: {s1}, feedback: {e1}")
    print(f"Gemini score: {s2}, feedback: {e2}")

    # ------------------------------------------------------------
    # Test Pairwise Model
    # ------------------------------------------------------------
    pairwise_judge = load_judge_model(JudgeType.PAIRWISE, "gemini-2.0-flash")

    s1, e1 = pairwise_judge.get_score(question, target_response_1, target_response_2)
    print(f"Pairwise score given by gemini-2.0-flash: {s1}, feedback: {e1}")

    # ------------------------------------------------------------
    # Test Arena Hard Auto Model
    # ------------------------------------------------------------
    arena_hard_auto_judge = load_judge_model(JudgeType.ARENA_HARD_AUTO, "gemini-2.0-flash")
    s1, e1 = arena_hard_auto_judge.get_score(question, target_response_1, target_response_2)
    print(f"ArenaHardAuto score given by gemini-2.0-flash: {s1}, feedback: {e1}")


    