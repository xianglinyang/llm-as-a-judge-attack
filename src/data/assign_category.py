'''
Dataset for evaluation:
1. AlpacaEval
2. ArenaHard
3. MTBench
4. UltraFeedback
5. Dolly
6. Oasst
7. JudgeBench
'''
import json
import os
import logging
import asyncio
import argparse

from src.llm_zoo import load_model
from src.utils import str2json
from src.logging_utils import setup_logging
from src.data.data_utils import load_metadata

logger = logging.getLogger(__name__)

template='''Given a question, please categorize it to one of the following categories:

1. Computer Science & Programming
2. Mathematics & Statistics
3. Science & Engineering
4. Business & Finance
5. Writing & Communication
6. Social & Daily Life
7. Others

## Question: {}

Please output the generated content in a json format, for example:
```json
{{
"category": "integer from 1 to 7" // specific category, such as 1 for "Computer Science & Programming"
}}
```

Formatted the abovementioned schema and categorize the given question:'''

CATEGORIES = [
    "Computer Science & Programming",
    "Mathematics & Statistics",
    "Science & Engineering",
    "Business & Finance",
    "Writing & Communication",
    "Social & Daily Life",
    "Others"
]

async def assign_category(save_dir, dataset_name, assign_model_name="gpt-4.1-mini"):
    # load the dataset
    metadata = load_metadata(save_dir, dataset_name)
    
    # check if the dataset is already assigned category
    if "category" in metadata[0]:
        logger.info(f"Dataset {dataset_name} is already assigned category")
        return metadata
    
    model = load_model(assign_model_name)
    
    # batch invoke
    prompts = [template.format(item["instruction"]) for item in metadata]
    responses = await model.batch_invoke(prompts)
    
    for item, response in zip(metadata, responses):
        try:
            category = int(str2json(response)["category"])
            item["category"] = CATEGORIES[category - 1]
        except:
            item["category"] = "None"
    
    save_path = os.path.join(save_dir, dataset_name, f"metadata.json")
    with open(save_path, "w") as f:
        json.dump(metadata, f, indent=4)
    return metadata


async def main(args):

    data_dir = args.data_dir
    dataset_name = args.dataset_name
    model_name = args.model_name

    await assign_category(data_dir, dataset_name, assign_model_name=model_name)

    logger.info(f"Assign category for {dataset_name} done")


if __name__ == "__main__":
    setup_logging(task_name="assign_category")

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="AlpacaEval")
    parser.add_argument("--assign_model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--data_dir", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data")
    args = parser.parse_args()

    asyncio.run(main(args))
