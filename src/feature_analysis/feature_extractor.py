'''
This file is used to extract features from the answer text.
The features are used to analyze the stylistic biases in LLM judges.

Bias used:

"Sentiment", # The preference for expressions of positive or negative emotions, affecting its judgment of emotional content.
"Authority", # Generate citation. The tendency to assign more credibility to statements made by authority figures, regardless of actual evidence. (citation)
"Markdown Format", # Replace the following into one prompt
"Distraction", # The inclination to give more attention to irrelevant or unimportant details.
"Emoji",
"Verbosity",# LLM judges favor longer responses, even if they are not as clear, high-quality, or accurate as shorter alternatives.
"Bullet-point list",
"Bandwagon",
"Newline", # List of elements arranged on separate lines
"JSON", # JavaScript Object Notation 

Feature Extracted:
1. Linguistic & Readability Features (Overall Text Quality)
- Token Count
- Word Count
- Readability Score (e.g., Flesch-Kincaid Grade Level)
- Sentiment score

2. Structural & Formatting Features (Visual Organization)
- Paragraph Count
- List Item Count
- number of special character
- number of bolding
- number of italicizing
- number of double brackets
- number of double parentheses
- number of quoting
- number of bullet-point list
- Citation Marker Count: Generalizes your specific ideas (double brackets, parentheses). Use a regular expression to count occurrences of common citation patterns like [1], (Author, 2024), [source], etc. This is a direct measure of the "Authority" strategy's output.
- Is Formatted Code (Binary): json or yaml format

3. Lexical & Stylistic Features
- Emoji Count
- Formality Score: formal tone from -1 to 1

4. Label: judge score
- the final score of the answer

'''

import pandas as pd
import re
import tiktoken
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji
from transformers import pipeline
import torch
from typing import List
from tqdm.auto import tqdm

# --- Initialize Models and Tools (do this once for efficiency) ---

# 1. Tokenizer for token count (used by GPT-4)
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except:
    tokenizer = tiktoken.encoding_for_model("gpt-4")

# 2. Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# 3. Formality Classifier from Hugging Face
# This will download the model the first time it's run.
# It's a heavier model, so a GPU can speed this up, but it will run on CPU.
# Set to False to disable formality analysis if it's causing issues
ENABLE_FORMALITY_ANALYSIS = True

if ENABLE_FORMALITY_ANALYSIS:
    try:
        formality_classifier = pipeline(
            "text-classification",
            model="s-nlp/roberta-base-formality-ranker",
            device=0 if torch.cuda.is_available() else -1, # Use GPU if available
            padding=True,       # Pad shorter sequences to max_length
            truncation=True,    # Truncate longer sequences to max_length
            max_length=512      # Set max_length to the model's limit
        )
        print("Formality model loaded successfully with padding and truncation enabled.")
    except Exception as e:
        print(f"Could not load formality model: {e}. Formality scores will be None.")
        formality_classifier = None
else:
    print("Formality analysis disabled.")
    formality_classifier = None


def get_feature_names():
    feature_names = []
    feature_names.extend(['token_count', 'word_count', 'readability_flesch_grade', 'sentiment_compound', 'sentiment_positive', 'sentiment_negative'])
    feature_names.extend(['paragraph_count', 'list_item_count', 'special_char_count', 'bold_count', 'italic_count', 'double_bracket_count', 'double_parentheses_count', 'quote_block_count', 'citation_marker_count', 'is_formatted_code'])
    feature_names.extend(['emoji_count', 'formality_score'])
    return feature_names


def extract_features(answers: List[str], batch_size: int = 64) -> pd.DataFrame:
    """
    Extracts a comprehensive set of linguistic, structural, and stylistic
    features from a list of text answers.

    Args:
        answers: A list of strings, where each string is an answer.
        batch_size: Batch size for formality classifier inference (default: 512).

    Returns:
        A pandas DataFrame where each row corresponds to an answer and each
        column corresponds to an extracted feature.
    """
    all_features = []
    
    # Pre-compute formality scores in batches for efficiency
    formality_scores = []
    if formality_classifier:
        print(f"Analyzing formality for {len(answers)} answers (batch size: {batch_size})...")
        try:
            # The pipeline returns a generator. We pass the list of answers directly
            # and then wrap the resulting generator with tqdm to show progress.
            results_generator = formality_classifier(answers, batch_size=batch_size, top_k=2)

            for result in tqdm(results_generator, total=len(answers), desc="Formality Analysis"):
                # The pipeline yields a list of dicts for each input text when top_k > 1
                if isinstance(result, list):
                    formal_score = next((item['score'] for item in result if item['label'] == 'formal'), 0.5)
                    informal_score = next((item['score'] for item in result if item['label'] == 'informal'), 0.5)
                    formality_scores.append(formal_score - informal_score)
                else:  # Fallback for unexpected, single-item result (e.g., if top_k=1)
                    formal_score = result.get('score', 0.5) if result.get('label') == 'formal' else 0.5
                    informal_score = result.get('score', 0.5) if result.get('label') == 'informal' else 0.5
                    formality_scores.append(formal_score - informal_score)

        except Exception as e:
            print(f"Error during optimized batch formality analysis: {e}. All formality scores will be None.")
            formality_scores = [None] * len(answers)
    else:
        formality_scores = [None] * len(answers)
    

    # Process each text for other features
    for idx, text in enumerate(tqdm(answers, desc="Extracting Other Features")):
        features = {}

        # --- Group 1: Linguistic & Readability Features ---
        features['token_count'] = len(tokenizer.encode(text))
        features['word_count'] = len(text.split())
        try:
            features['readability_flesch_grade'] = textstat.flesch_kincaid_grade(text)
        except:
            features['readability_flesch_grade'] = None # Handle empty or short text
        sentiment_scores = sentiment_analyzer.polarity_scores(text)
        features['sentiment_compound'] = sentiment_scores['compound']
        features['sentiment_positive'] = sentiment_scores['pos']
        features['sentiment_negative'] = sentiment_scores['neg']
        
        # --- Group 2: Structural & Formatting Features ---
        features['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
        # Count lines starting with *, -, or number patterns
        features['list_item_count'] = len(re.findall(r'^\s*(\*|-|\d+\.)\s+', text, re.MULTILINE))
        # Non-alphanumeric, non-whitespace characters
        features['special_char_count'] = len(re.findall(r'[^\w\s]', text))
        features['bold_count'] = text.count('**') // 2
        features['italic_count'] = (text.count('*') // 2) - features['bold_count'] # Avoid double counting
        features['double_bracket_count'] = text.count('[[')
        features['double_parentheses_count'] = text.count('((')
        features['quote_block_count'] = len(re.findall(r'^\s*>', text, re.MULTILINE))
        # Regex for patterns like [1], [23], (Author, 2024), [source]
        features['citation_marker_count'] = len(re.findall(r'\[\d+\]|\(\w+,\s*\d{4}\)|\[source\]', text, re.IGNORECASE))
        # Check for json/yaml code fences
        is_formatted_code = 1 if re.search(r'```(json|yaml)', text) else 0
        features['is_formatted_code'] = is_formatted_code
        
        # --- Group 3: Lexical & Stylistic Features ---
        features['emoji_count'] = emoji.emoji_count(text)
        # Use pre-computed formality score
        features['formality_score'] = formality_scores[idx]
        
        all_features.append(features)

    return pd.DataFrame(all_features)

# --- Example Usage ---
if __name__ == '__main__':
    sample_answers = [
        # 1. A formal, structured, and cited answer
        "The theory of relativity, proposed by Albert Einstein, is foundational to modern physics. It is primarily divided into two parts:\n\n1. **Special Relativity**: Deals with physics in the absence of gravity.\n2. **General Relativity**: A theory of gravitation.\n\nThis framework has been experimentally verified multiple times (Einstein, 1905). For more details, see [source].",

        # 2. An informal, emoji-heavy answer
        "OMG, you HAVE to check this out! It's literally the best thing ever 😂. Super easy, barely an inconvenience. I give it a 10/10 ✨.",

        # 3. An answer formatted as JSON
        "```json\n{\n  \"status\": \"success\",\n  \"data\": {\n    \"item_id\": 12345,\n    \"is_available\": true\n  }\n}\n```",

        # 4. A simple, neutral answer
        "The capital of France is Paris.",
        
        # 5. An empty string to test edge cases
        ""
    ]

    # Extract features from the sample answers
    features_df = extract_features(sample_answers)

    # Display the resulting DataFrame
    pd.set_option('display.max_columns', None)
    print("\n--- Extracted Features DataFrame ---")
    print(features_df)