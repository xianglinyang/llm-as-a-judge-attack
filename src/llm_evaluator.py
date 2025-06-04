"""
This class is used for llm-as-a-judge.

Point-wise scoring:
1. Absolute scoring, e.g. "Evaluate the harmlessness of this response on a scale of 1-9"
"""

import math
import os
import torch
import numpy as np

from src.judge_prompts import POINTWISE_EVALUATION_PROMPT, PAIRWISE_EVALUATION_PROMPT, ARENA_HARD_AUTO_PROMPT, MT_BENCH_PROMPT
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
        self.model = load_model(model_name)

    def pointwise_score(self, input_q, response) -> float:
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
    
    def pairwise_score(self, input_q, response1, response2) -> float:
        """Returns the model's confidence that the summary is its own output."""
        formatted_prompt = PAIRWISE_EVALUATION_PROMPT.format(INPUTS=input_q, OUTPUT_A=response1, OUTPUT_B=response2)
        response = self.model.invoke(formatted_prompt)
        try:
            json_response = str2json(response)
            better_model = json_response["better_model"]
            feedback = json_response["feedback"]
        except:
            better_model = -1
            feedback = "Error: Failed to parse the response as a JSON object."
            print(f"Error: Failed to parse the response as a JSON object. {response}")
        return better_model, feedback


if __name__ == "__main__":
    gpt = JudgeModel("gpt-4o-2024-05-13")
    gemini = JudgeModel("gemini-1.5-flash")

    question = "I'm curious about how renewable energy technologies might transform urban infrastructure in the next decade. What potential innovations do you see emerging?"
    target_response_1 = "Renewable energy technologies are poised to revolutionize urban landscapes, with advanced solar roadways potentially generating electricity while supporting vehicle traffic. Vertical wind turbines integrated into skyscrapers could become more efficient, potentially producing up to 30% of a building's energy needs. While some experts are skeptical about total urban energy independence, emerging technologies like transparent solar panels and bio-responsive energy systems suggest we're approaching a transformative period in sustainable urban design."
    target_response_2 = "ChatGPT: Sustainable urban design is a critical component of urban planning, aiming to create cities that are environmentally friendly, socially equitable, and economically viable. This approach considers the long-term impact of urban development on the natural environment, human well-being, and economic prosperity. Key principles include reducing carbon emissions, conserving water resources, promoting biodiversity, and ensuring access to green spaces. By integrating sustainable practices into urban design, cities can achieve a balance between economic growth and environmental sustainability, fostering a more resilient and livable urban environment."

    s1, e1 = gpt.pointwise_score(question, target_response_1)
    s2, e2 = gemini.pointwise_score(question, target_response_2)
    better_model, feedback = gpt.pairwise_score(question, target_response_1, target_response_2)

    print(f"GPT score: {s1}, feedback: {e1}")
    print(f"Gemini score: {s2}, feedback: {e2}")
    print(f"GPT pairwise score: {better_model}, feedback: {feedback}")