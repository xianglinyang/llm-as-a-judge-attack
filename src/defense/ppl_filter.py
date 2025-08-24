# First, ensure you have the required libraries installed:
# pip install torch transformers

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm

class PPLFilter:
    """
    Implements a windowed Perplexity (PPL) filter.

    This filter is designed to detect adversarial text by identifying localized fluctuations in perplexity. It works in two stages:
    
    1. Calibration: Sets a PPL threshold based on the maximum windowed PPL
       found in a list of "clean" or "baseline" text samples.
    2. Filtering: Checks if a candidate text has a maximum windowed PPL that
       exceeds this calibrated threshold.
    """
    def __init__(self, model_id: str = "gpt2", window_size: int = 32, device: str = "cuda"):
        """
        Initializes the PPL filter.

        Args:
            model_id (str): The model ID for the GPT-2 model to use (e.g., "gpt2", "gpt2-medium").
            window_size (int): The number of tokens in the sliding window.
        """
        print(f"Loading model '{model_id}' for PPL calculation...")
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model.eval()
        self.window_size = window_size
        self.threshold = None
        print(f"Model loaded on {self.device}.")

    def _calculate_text_perplexity(self, text: str) -> float:
        """Calculates the perplexity of a given string of text."""
        if not text.strip():
            return float('inf') # Return high perplexity for empty/whitespace strings

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # If there are no tokens, we can't calculate perplexity.
        if inputs.input_ids.size(1) == 0:
            return float('inf')

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        
        # Perplexity is the exponential of the loss
        ppl = torch.exp(loss)
        return ppl.item()

    def _get_max_windowed_ppl(self, text: str) -> float:
        """
        Calculates the maximum perplexity across all sliding windows for a text.

        Args:
            text (str): The input text to analyze.

        Returns:
            float: The highest perplexity value found in any window.
        """
        tokens = self.tokenizer.encode(text)

        # If the text is shorter than the window size, calculate PPL for the whole text
        if len(tokens) <= self.window_size:
            return self._calculate_text_perplexity(text)

        max_ppl = 0.0
        
        # Slide the window across the tokens
        for i in range(len(tokens) - self.window_size + 1):
            window_tokens = tokens[i : i + self.window_size]
            window_text = self.tokenizer.decode(window_tokens)
            
            current_ppl = self._calculate_text_perplexity(window_text)
            
            if current_ppl > max_ppl:
                max_ppl = current_ppl
                
        return max_ppl

    def calibrate(self, baseline_responses: list[str]):
        """
        Calibrates the filter by setting the PPL threshold based on clean data.
        The threshold is the maximum windowed PPL found in the baseline responses.

        Args:
            baseline_responses (list[str]): A list of non-adversarial text samples.
        """
        print(f"Calibrating PPL threshold with {len(baseline_responses)} baseline samples...")
        max_ppl_overall = 0.0
        for response in tqdm(baseline_responses, desc="Calibrating"):
            response_max_ppl = self._get_max_windowed_ppl(response)
            if response_max_ppl > max_ppl_overall:
                max_ppl_overall = response_max_ppl
        
        self.threshold = max_ppl_overall
        print(f"\n✅ Calibration complete. PPL threshold set to: {self.threshold:.2f}")

    def should_filter(self, candidate_response: str) -> bool:
        """
        Determines if a candidate response should be filtered based on its PPL.

        Args:
            candidate_response (str): The text to check.

        Returns:
            bool: True if the text should be filtered, False otherwise.
            
        Raises:
            ValueError: If the filter has not been calibrated yet.
        """
        if self.threshold is None:
            raise ValueError("Filter must be calibrated with `calibrate()` before use.")
            
        candidate_max_ppl = self._get_max_windowed_ppl(candidate_response)
        
        print(f"Candidate response max windowed PPL: {candidate_max_ppl:.2f}")
        
        return candidate_max_ppl > self.threshold

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Initialize the filter
    ppl_filter = PPLFilter(model_id="gpt2", window_size=32)

    # 2. Define baseline (clean) responses for calibration
    # These should be examples of normal, expected model outputs.
    baseline_responses = [
        "The capital of France is Paris, a bustling city known for its art, fashion, and culture.",
        "To make a simple pasta dish, you'll need pasta, tomatoes, garlic, olive oil, and some fresh basil.",
        "The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space, time, and gravity.",
        "Artificial intelligence is a branch of computer science that aims to create machines capable of intelligent behavior.",
    ]

    # 3. Calibrate the filter using the baseline data
    ppl_filter.calibrate(baseline_responses)
    print("-" * 50)
    
    # 4. Test the filter on different inputs

    # A normal, clean response (should NOT be filtered)
    clean_response = "Great question! The best exercises for beginners are those that are simple, effective, and help build a solid foundation of strength, flexibility, and cardiovascular fitness. Here are some excellent beginner-friendly exercises across different categories:\n\n### 1. **Bodyweight Exercises**\nThese require no equipment and are perfect for building strength and learning proper movement patterns.\n- **Squats:** Great for legs and glutes.\n- **Push-ups (modified on knees if needed):** Build upper body and core strength.\n- **Lunges:** Improve balance and leg strength.\n- **Planks:** Strengthen the core.\n- **Glute bridges:** Activate and strengthen the glutes and lower back.\n\n### 2. **Cardiovascular Exercises**\nGood for improving endurance and heart health.\n- **Walking:** Easy to start and low impact.\n- **Jogging or light running:** Once comfortable with walking.\n- **Cycling:** Low-impact cardio option.\n- **Jumping jacks:** Great full-body warm-up and cardio move.\n- **Swimming:** Full-body, low-impact cardio.\n\n### 3. **Flexibility and Mobility**\nHelps prevent injury and improve range of motion.\n- **Basic stretching:** Hamstrings, quads, calves, shoulders.\n- **Yoga or beginner yoga flows:** Enhance flexibility and balance.\n\n### 4. **Basic Strength Training with Equipment**\nIf you have access to equipment like dumbbells or resistance bands:\n- **Dumbbell rows:** For back and arms.\n- **Bicep curls:** For arm strength.\n- **Resistance band pull-aparts:** For shoulder stability.\n\n---\n\n### Tips for Beginners:\n- Start slow and focus on learning proper form.\n- Aim for 2-3 sessions per week, gradually increasing frequency and intensity.\n- Warm up before exercising and cool down/stretch after.\n- Listen to your body to avoid injury.\n\nIf you want, I can help you create a simple beginner workout routine!"
    print(f"Testing a clean response:\n'{clean_response}'")
    is_filtered_clean = ppl_filter.should_filter(clean_response)
    print(f"Result: {'FILTERED' if is_filtered_clean else 'PASSED'}\n")


    
    