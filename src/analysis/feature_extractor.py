'''
This file is used to extract features from the answer text.
The features are used to analyze the stylistic biases in LLM judges.

Bias Types:
-Bandwagon
-Authority
-Distraction

Bias Strategies:
-Verbosity
-Sentiment
-Diversity

Format Types:
-Markdown Format
-Python list
-Bullet-point list
-Newline
-JSON
-YAML

- Emoji

The features are:
- the final score of the answer

- use authority
- use distraction
- length char of answer
- length word of answer
- number of special character
- number of bolding
- number of italicizing
- number of double brackets
- number of double parentheses
- number of placeholder
- number of quoting

- number of python list
- number of bullet-point list
- number of \n

- use of json format
- use of yaml format
- number of emoji
'''
import re
import emoji
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from src.analysis.data_collector import load_analysis_data

class FeatureExtractor:
    """
    A class to extract a feature set from a single experimental trial
    for analyzing stylistic biases in LLM judges.
    """
    def __init__(self):
        """Initializes the feature extractor and its components."""
        # Initialize the sentiment analyzer once to be efficient
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        print("FeatureExtractor initialized.")
    # - the final score of the answer
    # - use authority
    # - use distraction

    # - number of python list
    # - number of bullet-point list
    # - number of \n

    # - use of json format
    # - use of yaml format
    # - number of emoji

    def _count_char(self, text: str) -> int:
        """Counts the number of characters in a string."""
        return len(text)

    def _count_words(self, text: str) -> int:
        """Counts the number of words in a string."""
        return len(text.split())
    
    def _count_special_char(self, text: str) -> int:
        """Counts the number of special characters in a string."""
        return len(re.findall(r'[^\w\s.,!?;:]', text))
    
    def _count_bold(self, text: str) -> int:
        """Counts the number of bolding in a string."""
        return text.count('**')
    
    def _count_italic(self, text: str) -> int:
        """Counts the number of italicizing in a string."""
        # Count single asterisks that are not part of bold formatting
        # This regex matches single * that are not followed by another *
        return len(re.findall(r'(?<!\*)\*(?!\*)', text))
    
    def _count_double_bracket(self, text: str) -> int:
        """Counts the number of double brackets in a string."""
        return text.count('[[') + text.count(']]')
    
    def _count_double_paren(self, text: str) -> int:
        """Counts the number of double parentheses in a string."""
        return text.count('((') + text.count('))')
    
    def _count_placeholder(self, text: str) -> int:
        """Counts the number of placeholders in a string."""
        # Look for square brackets that are not part of markdown links
        # This excludes [text](url) patterns
        return len(re.findall(r'\[(?!.*?\]\(.*?\))[^\]]*?\]', text))
    
    def _count_quote(self, text: str) -> int:
        """Counts the number of quoting in a string."""
        return text.count('"') + text.count('"') + text.count('"')

    def _count_emojis(self, text: str) -> int:
        """Counts the number of emojis in a string using the emoji library."""
        return emoji.emoji_count(text)
    
    def _count_python_list(self, text: str) -> int:
        """Counts the number of python list in a string."""
        # Look for Python list patterns: [item1, item2, ...]
        return len(re.findall(r'\[[^\]]*,[^\]]*\]', text))
    
    def _count_bullet_list(self, text: str) -> int:
        """Counts the number of bullet list in a string."""
        return len(re.findall(r'^\s*[-*+]\s+', text, re.MULTILINE))
    
    def _count_line_break(self, text: str) -> int:
        """Counts the number of line breaks in a string."""
        return text.count('\n')

    def _get_sentiment_score(self, text: str) -> float:
        """
        Calculates the compound sentiment score (from -1 to 1).
        -1 is most negative, +1 is most positive.
        """
        return self.sentiment_analyzer.polarity_scores(text)['compound']
    
    def _detect_json_format(self, text: str) -> bool:
        """Detects if text contains JSON-like formatting."""
        # Look for JSON-like patterns: key-value pairs with quotes and colons
        json_pattern = r'"[^"]*"\s*:\s*["\d\[\]{}\w]'
        return bool(re.search(json_pattern, text))

    def _detect_yaml_format(self, text: str) -> bool:
        """Detects if text contains YAML-like formatting."""
        # Look for YAML-like patterns: key-value pairs with colons and indentation
        yaml_pattern = r'^\s*\w+\s*:\s*'
        return bool(re.search(yaml_pattern, text, re.MULTILINE))

    def _detect_markdown_format(self, text: str) -> bool:
        """Detects if text contains Markdown formatting."""
        # Look for common Markdown patterns
        markdown_patterns = [
            r'^#+\s+',  # Headers
            r'\*\*.*?\*\*',  # Bold
            r'\*.*?\*',  # Italic
            r'\[.*?\]\(.*?\)',  # Links
            r'^\s*[-*+]\s+',  # Bullet lists
        ]
        for pattern in markdown_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False

    def extract_features(self, text: str, strategies: list[str]) -> dict:
        """
        Processes a single trial and returns a dictionary of features.

        Args:
            text (str): The answer text.
            strategies (list[str]): The strategies applied to the answer.

        Returns:
            dict: A flat dictionary of feature names and their calculated values.
        """
        features = {}

        # 1. binary features
        # -Bandwagon
        # -Authority
        # -Distraction
        features['use_bandwagon'] = 1 if 'Bandwagon' in strategies else 0
        features['use_authority'] = 1 if 'Authority' in strategies else 0
        features['use_distraction'] = 1 if 'Distraction' in strategies else 0
        

        # 2. verbosity features
        features['char_count'] = self._count_char(text)
        features['word_count'] = self._count_words(text)

        # 3. Sentiment features
        features['sentiment_score'] = self._get_sentiment_score(text)

        # 4. Diversity features
        features['use_diversity'] = 1 if "Diversity" in strategies else 0

        # 5. Format features
        features['use_json_format'] = 1 if self._detect_json_format(text) else 0
        features['use_yaml_format'] = 1 if self._detect_yaml_format(text) else 0
        features['use_markdown_format'] = 1 if self._detect_markdown_format(text) else 0
        features['use_python_list'] = 1 if self._count_python_list(text) > 0 else 0

        # 6. Text formatting features
        features['bold_count'] = self._count_bold(text)
        features['italic_count'] = self._count_italic(text)
        features['double_bracket_count'] = self._count_double_bracket(text)
        features['double_paren_count'] = self._count_double_paren(text)
        features['quote_count'] = self._count_quote(text)
        features['special_char_count'] = self._count_special_char(text)
        features['line_break_count'] = self._count_line_break(text)

        # 7. emoji features
        features['emoji_count'] = self._count_emojis(text)
        return features

    def get_feature_names(self):
        return list(self.extract_features("", []).keys())


def extract_features_for_analysis(data_dir: str, data_type: str, dataset_name: str, judge_type: str, judge_backbone: str, response_model_name: str, helper_model_name: str, reward_type=None, baseline_model_name=None, answer_position=None):
    data_pairs = load_analysis_data(data_dir, data_type, dataset_name, judge_type, judge_backbone, response_model_name, helper_model_name, reward_type=reward_type, baseline_model_name=baseline_model_name, answer_position=answer_position)
    
    # question_list = [item['instruction'] for item in data_pairs]
    modified_response_list = [item['modified_response'] for item in data_pairs]
    init_response_list = [item['init_response'] for item in data_pairs]
    modified_score_list = [item['modified_score'] for item in data_pairs]
    init_score_list = [item['init_score'] for item in data_pairs]
    bias_strategy_list = [item['bias_strategy'] for item in data_pairs]
    
    # 1. Create an instance of our extractor
    extractor = FeatureExtractor()
    init_X = [extractor.extract_features(response, bias_strategy) for response, bias_strategy in zip(init_response_list, bias_strategy_list)]
    modified_X = [extractor.extract_features(response, bias_strategy) for response, bias_strategy in zip(modified_response_list, bias_strategy_list)]

    feature_names = extractor.get_feature_names()

    init_df = pd.DataFrame(init_X, columns=pd.Index(feature_names))
    modified_df = pd.DataFrame(modified_X, columns=pd.Index(feature_names))

    init_y = pd.Series(init_score_list, name='score')
    modified_y = pd.Series(modified_score_list, name='score')

    return init_df, modified_df, init_y, modified_y, feature_names


# ==============================================================================
#                      EXAMPLE USAGE DEMONSTRATION
# ==============================================================================
if __name__ == "__main__":
    # 0. load data
    # data_dir = "/data2/xianglin/llm-as-a-judge-attack/trajectories"
    # data_type = "trajectory"
    # reward_type = "relative"
    data_dir = "/data2/xianglin/llm-as-a-judge-attack/data"
    data_type = "perturbation"
    reward_type = None

    dataset_name = "AlpacaEval"
    judge_type = "pointwise"
    judge_backbone = "gemini-2.0-flash"
    response_model_name = "gpt-4o-mini"
    helper_model_name = "gpt-4.1-nano"
    reward_type = "relative"

    # 1. extract features
    init_df, modified_df, init_y, modified_y, feature_names = extract_features_for_analysis(data_dir, data_type, dataset_name, judge_type, judge_backbone, response_model_name, helper_model_name, reward_type=reward_type, baseline_model_name=None, answer_position=None)

    print(init_df)
    print(modified_df)
    print(init_y)
    print(modified_y)