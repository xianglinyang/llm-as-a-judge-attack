from typing import Dict, List


def parse_criteria(criteria_string: str, criteria_type: str = "criteria") -> Dict[str, List[str]]:
    """
    Parse criteria from 'key1=value1,key2=value2' format.
    
    Supports multiple values for the same key:
    'dataset_name=AlpacaEval,dataset_name=UltraFeedback' -> {'dataset_name': ['AlpacaEval', 'UltraFeedback']}
    
    Args:
        criteria_string: String in format 'key1=value1,key2=value2'
        criteria_type: Type of criteria for error messages ('filter', 'exclude', 'criteria')
        
    Returns:
        Dictionary mapping keys to lists of values
    """
    if not criteria_string:
        return {}
    
    criteria = {}
    pairs = criteria_string.split(',')
    
    for pair in pairs:
        if '=' in pair:
            key, value = pair.split('=', 1)  # Split only on first =
            key = key.strip()
            value = value.strip()
            
            if key in criteria:
                criteria[key].append(value)
            else:
                criteria[key] = [value]
        else:
            print(f"Warning: Invalid {criteria_type} format '{pair}'. Expected 'key=value'")
    
    return criteria


def parse_exclude_criteria(exclude_string: str) -> Dict[str, List[str]]:
    """
    Parse exclude criteria from 'key1=value1,key2=value2' format.
    
    Args:
        exclude_string: String in format 'key1=value1,key2=value2'
        
    Returns:
        Dictionary mapping keys to lists of values to exclude
    """
    return parse_criteria(exclude_string, "exclude")


def parse_filter_criteria(filter_string: str) -> Dict[str, List[str]]:
    """
    Parse filter criteria from 'key1=value1,key2=value2' format.
    
    Args:
        filter_string: String in format 'key1=value1,key2=value2'
        
    Returns:
        Dictionary mapping keys to lists of values to include
    """
    return parse_criteria(filter_string, "filter")