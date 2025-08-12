import random

import time
from src.data.data_utils import load_dataset_for_exploration
from src.llm_zoo import load_model
from src.evolve_agent.direct_prompting import BASELINE_PROMPT
from src.evolve_agent.bias_strategies import BiasModification, Bias_types
from src.judge_prompts import POINTWISE_EVALUATION_PROMPT
from src.llm_zoo.api_zoo import get_model_name



if __name__ == "__main__":
    dataset_name = "AlpacaEval"
    judge_model_implementation_name = "openai/o3"
    response_model_implementation_name = "openai/gpt-4.1-mini"
    llm_agent_implementation_name = "openai/gpt-5-nano"
    data_dir = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data"
    budget = 20

    response_model_name = get_model_name(response_model_implementation_name)
    llm_agent_name = get_model_name(llm_agent_implementation_name)
    judge_model_name = get_model_name(judge_model_implementation_name)

    # 1. load dataset
    baseline_dataset = load_dataset_for_exploration(data_dir, dataset_name, response_model_name, judge_model_name)
    question_list = [item['instruction'] for item in baseline_dataset]
    init_response_list = [item['output'] for item in baseline_dataset]
    N = len(question_list)

    idx = random.randint(0, N - 1)
    question = question_list[idx]
    init_response = init_response_list[idx]

    # 2. direct_prompting
    llm_agent = load_model(response_model_implementation_name)
    prompt = BASELINE_PROMPT.format(original_answer=init_response)
    start_time = time.time()
    call_result = llm_agent.invoke(prompt, return_cost=True)
    end_time = time.time()
    cost = call_result.cost
    input_tokens = call_result.input_tokens
    output_tokens = call_result.output_tokens

    print("--------------------------------")
    print("Direct Prompting")
    print(f"Cost: {cost}, Input tokens: {input_tokens}, Output tokens: {output_tokens}")
    print(f"Estimate all: {cost * N * budget}")
    print(f"Time: {(end_time - start_time) * budget/60:.1f} minutes")

    # 3. bias_evolution
    llm_agent = load_model(llm_agent_implementation_name)
    bias_modification = BiasModification(llm_agent)
    start_time = time.time()
    response = bias_modification.principle_guided_mutation(init_response, Bias_types[1], return_cost=True)
    end_time = time.time()
    cost = response.cost
    input_tokens = response.input_tokens
    output_tokens = response.output_tokens

    print("--------------------------------")
    print("Bias Evolution")
    print(f"Cost: {cost}, Input tokens: {input_tokens}, Output tokens: {output_tokens}")
    print(f"Estimate all: {cost * N * budget}")
    print(f"Time: {(end_time - start_time) * budget/60:.1f} minutes")

    # 4. pointwise
    judge_model = load_model(judge_model_implementation_name)
    formatted_prompt = POINTWISE_EVALUATION_PROMPT.format(INPUTS=question, OUTPUT=init_response)
    start_time = time.time()
    call_result = judge_model.invoke(formatted_prompt, return_cost=True)
    end_time = time.time()
    cost = call_result.cost
    input_tokens = call_result.input_tokens
    output_tokens = call_result.output_tokens
    print("--------------------------------")
    print("Pointwise")
    print(f"Cost: {cost}, Input tokens: {input_tokens}, Output tokens: {output_tokens}")
    print(f"Estimate all: {cost * N * budget}")
    print(f"Time: {(end_time - start_time) * budget/60:.1f} minutes")

    # 5. pairwise
    # judge_model = load_judge_model(JudgeType.PAIRWISE, judge_model_name)
    # formatted_prompt = PAIRWISE_EVALUATION_PROMPT.format(INPUTS=question, OUTPUT=init_response)
    # call_result = judge_model.invoke(formatted_prompt, return_cost=True)
    # cost = call_result.cost
    # input_tokens = call_result.input_tokens
    # output_tokens = call_result.output_tokens
    # print(f"Cost: {cost}, Input tokens: {input_tokens}, Output tokens: {output_tokens}")

