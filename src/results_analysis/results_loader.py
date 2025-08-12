'''Read the results from the json file and load them into a pandas dataframe.
a sample result is like this:
{
        "categories": {
            "Others": {
                "up_num": 0,
                "down_num": 0,
                "tie_num": 3,
                "skip_num": 3,
                "exploration_length": 4.0,
                "avg_score_before": 9.0,
                "avg_score_after": 9.0,
                "average_improvement": 0.0
            },
            "Social & Daily Life": {
                "up_num": 0,
                "down_num": 0,
                "tie_num": 9,
                "skip_num": 9,
                "exploration_length": 3.2222222222222223,
                "avg_score_before": 9.0,
                "avg_score_after": 9.0,
                "average_improvement": 0.0
            },
            "Computer Science & Programming": {
                "up_num": 2,
                "down_num": 0,
                "tie_num": 11,
                "skip_num": 11,
                "exploration_length": 2.923076923076923,
                "avg_score_before": 8.76923076923077,
                "avg_score_after": 9.0,
                "average_improvement": 0.23076923076923078
            },
            "Mathematics & Statistics": {
                "up_num": 0,
                "down_num": 0,
                "tie_num": 16,
                "skip_num": 16,
                "exploration_length": 3.375,
                "avg_score_before": 9.0,
                "avg_score_after": 9.0,
                "average_improvement": 0.0
            },
            "Business & Finance": {
                "up_num": 2,
                "down_num": 0,
                "tie_num": 6,
                "skip_num": 6,
                "exploration_length": 3.0,
                "avg_score_before": 8.75,
                "avg_score_after": 9.0,
                "average_improvement": 0.25
            },
            "Writing & Communication": {
                "up_num": 3,
                "down_num": 0,
                "tie_num": 14,
                "skip_num": 14,
                "exploration_length": 3.235294117647059,
                "avg_score_before": 8.529411764705882,
                "avg_score_after": 8.941176470588236,
                "average_improvement": 0.4117647058823529
            },
            "Science & Engineering": {
                "up_num": 2,
                "down_num": 0,
                "tie_num": 12,
                "skip_num": 12,
                "exploration_length": 3.2857142857142856,
                "avg_score_before": 8.285714285714286,
                "avg_score_after": 9.0,
                "average_improvement": 0.7142857142857143
            }
        },
        "strategy": "UCB",
        "judge_type": "pointwise",
        "answer_position": null,
        "dataset_name": "MTBench",
        "judge_backbone": "google/gemini-2.5-flash",
        "baseline_response_model_name": null,
        "llm_agent_name": "gemini-2.5-flash-lite",
        "response_model_name": "openai/gpt-4.1-mini",
        "test_mode": "single",
        "lambda_reg": 1.0,
        "n_features": 384,
        "budget": 20,
        "pool_size": 3,
        "eval_num": 9,
        "reward_type": "absolute",
        "alpha": 1.0,
        "timestamp": "2025-08-06 19:45:14",
        "time_taken": 634.6476163864136
    },
'''

import pandas as pd
import json
from typing import List, Dict, Any, Union


def load_results(result_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Load results from a list of JSON results into a pandas DataFrame.
    
    Args:
        result_list: List of dictionaries containing experiment results
        
    Returns:
        pd.DataFrame: DataFrame with flattened results, one row per category per experiment
    """
    flattened_results = []
    
    for result in result_list:
        # Extract metadata fields (same for all categories in this result)
        metadata = {
            'strategy': result.get('strategy'),
            'judge_type': result.get('judge_type'),
            'answer_position': result.get('answer_position'),
            'dataset_name': result.get('dataset_name'),
            'judge_backbone': result.get('judge_backbone'),
            'baseline_response_model_name': result.get('baseline_response_model_name'),
            'llm_agent_name': result.get('llm_agent_name'),
            'response_model_name': result.get('response_model_name'),
            'test_mode': result.get('test_mode'),
            'lambda_reg': result.get('lambda_reg'),
            'n_features': result.get('n_features'),
            'budget': result.get('budget'),
            'pool_size': result.get('pool_size'),
            'eval_num': result.get('eval_num'),
            'reward_type': result.get('reward_type'),
            'alpha': result.get('alpha'),
            'timestamp': result.get('timestamp'),
            'time_taken': result.get('time_taken')
        }
        
        # Extract category-specific results
        categories = result.get('categories', {})

        total_up_num = 0
        total_down_num = 0
        total_tie_num = 0
        total_skip_num = 0
        total_exploration_length = 0
        total_avg_score_before = 0
        total_avg_score_after = 0
        total_average_improvement = 0
        
        for category_name, category_data in categories.items():
            # Combine metadata with category data
            row_data = metadata.copy()
            row_data['category'] = category_name
            
            # Add category-specific metrics
            row_data.update({
                'up_num': category_data.get('up_num', 0),
                'down_num': category_data.get('down_num', 0),
                'tie_num': category_data.get('tie_num', 0),
                'skip_num': category_data.get('skip_num', 0),
                'exploration_length': category_data.get('exploration_length', 0.0),
                'avg_score_before': category_data.get('avg_score_before', 0.0),
                'avg_score_after': category_data.get('avg_score_after', 0.0),
                'average_improvement': category_data.get('average_improvement', 0.0)
            })
            flattened_results.append(row_data)

            total_up_num += category_data.get('up_num', 0)
            total_down_num += category_data.get('down_num', 0)
            total_tie_num += category_data.get('tie_num', 0)
            total_skip_num += category_data.get('skip_num', 0)
            total_exploration_length += category_data.get('exploration_length', 0.0)*(category_data.get('up_num', 0) + category_data.get('down_num', 0) + category_data.get('tie_num', 0))
            total_avg_score_before += category_data.get('avg_score_before', 0.0)*(category_data.get('up_num', 0) + category_data.get('down_num', 0) + category_data.get('tie_num', 0))
            total_avg_score_after += category_data.get('avg_score_after', 0.0)*(category_data.get('up_num', 0) + category_data.get('down_num', 0) + category_data.get('tie_num', 0))
            total_average_improvement += category_data.get('average_improvement', 0.0)*(category_data.get('up_num', 0) + category_data.get('down_num', 0) + category_data.get('tie_num', 0))
        
        total_exploration_length = total_exploration_length/(total_up_num + total_down_num + total_tie_num)
        total_avg_score_before = total_avg_score_before/(total_up_num + total_down_num + total_tie_num)
        total_avg_score_after = total_avg_score_after/(total_up_num + total_down_num + total_tie_num)
        total_average_improvement = total_average_improvement/(total_up_num + total_down_num + total_tie_num)

        row_data = metadata.copy()
        row_data['category'] = "Overall"

        row_data['up_num'] = total_up_num
        row_data['down_num'] = total_down_num
        row_data['tie_num'] = total_tie_num
        row_data['skip_num'] = total_skip_num
        row_data['exploration_length'] = total_exploration_length
        row_data['avg_score_before'] = total_avg_score_before
        row_data['avg_score_after'] = total_avg_score_after
        row_data['average_improvement'] = total_average_improvement
        flattened_results.append(row_data)

    
    # Create DataFrame
    df = pd.DataFrame(flattened_results)
    
    # Reorder columns to put category first, then metadata, then metrics
    column_order = ['category'] + [col for col in df.columns if col != 'category']
    df = df[column_order]
    
    return df


def load_results_from_file(file_path: str) -> pd.DataFrame:
    """
    Load results from a JSON file into a pandas DataFrame.
    
    Args:
        file_path: Path to the JSON file containing results
        
    Returns:
        pd.DataFrame: DataFrame with flattened results
    """
    with open(file_path, 'r') as f:
        result_list = json.load(f)
    
    return load_results(result_list)


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for the results DataFrame.
    
    Args:
        df: DataFrame from load_results function
        
    Returns:
        pd.DataFrame: Summary statistics grouped by category
    """
    summary = df.groupby('category').agg({
        'up_num': ['sum', 'mean', 'std'],
        'down_num': ['sum', 'mean', 'std'],
        'tie_num': ['sum', 'mean', 'std'],
        'skip_num': ['sum', 'mean', 'std'],
        'exploration_length': ['mean', 'std'],
        'avg_score_before': ['mean', 'std'],
        'avg_score_after': ['mean', 'std'],
        'average_improvement': ['mean', 'std']
    }).round(4)
    
    return summary


def filter_results(df: pd.DataFrame, 
                  strategy: str = None,
                  judge_type: str = None,
                  dataset_name: str = None,
                  category: str = None) -> pd.DataFrame:
    """
    Filter results DataFrame based on specified criteria.
    
    Args:
        df: DataFrame from load_results function
        strategy: Filter by strategy name
        judge_type: Filter by judge type
        dataset_name: Filter by dataset name
        category: Filter by category name
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    filtered_df = df.copy()
    
    if strategy:
        filtered_df = filtered_df[filtered_df['strategy'] == strategy]
    if judge_type:
        filtered_df = filtered_df[filtered_df['judge_type'] == judge_type]
    if dataset_name:
        filtered_df = filtered_df[filtered_df['dataset_name'] == dataset_name]
    if category:
        filtered_df = filtered_df[filtered_df['category'] == category]
    
    return filtered_df



if __name__ == "__main__":
    # load the results from the json file
    result_list = load_results_from_file("results/evaluation_results.json")
    print(result_list)
    # print columns names
    print(result_list.columns)
