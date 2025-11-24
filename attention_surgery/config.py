"""
Global configuration for Attention Surgery project.
"""

# Model configuration
MODEL_NAME = "gpt2-small"
NUM_LAYERS = 12
NUM_HEADS = 12
TOTAL_HEADS = NUM_LAYERS * NUM_HEADS

# Generation configuration
DEFAULT_MAX_NEW_TOKENS = 50
DEFAULT_TEMPERATURE = 0.0  # 0.0 = greedy (default), higher = more random
DEFAULT_TOP_K_SAMPLING = None  # None = disabled, 50 = sample from top 50 tokens
DEFAULT_TOP_P = None  # None = disabled, 0.9 = nucleus sampling with p=0.9
ENABLE_SAMPLING = False  # Enable sampling in UI by default

# Ablation methods
ABLATION_METHODS = ["zero", "mean", "random", "previous"]

# Importance scoring methods
IMPORTANCE_METHODS = ["gradient", "rollback", "ablation"]

# Ablation importance scoring configuration
ABLATION_SAMPLE_RATIO = 1.0  # 1.0 = evaluate all heads, 0.5 = evaluate 50% of heads
ABLATION_PRIORITY_LAYERS = None  # e.g., [10, 11] to prioritize last 2 layers
ABLATION_FAST_MODE = False  # If True, use sample_ratio=0.3 and priority_layers=[10,11]

# Default random seed
DEFAULT_SEED = 42

# UI configuration
DEFAULT_TOP_K_HEADS = 10

# Example prompts
EXAMPLE_PROMPTS = [
    "The doctor said that the patient should",
    "Alice and Bob went to the store. She bought",
    "Once upon a time in a distant galaxy",
    "The capital of France is",
    "In machine learning, attention mechanisms are"
]

# Visualization colors
COLOR_NORMAL = "#90EE90"  # Light green
COLOR_ABLATED = "#FF6B6B"  # Red
COLOR_IMPORTANT = "#FFB347"  # Orange
COLOR_BOTH = "#8B008B"  # Dark magenta (ablated + important)
