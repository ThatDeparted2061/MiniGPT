"""MiniGPT: LLM pretraining and alignment from scratch."""

__version__ = "0.1.0"

from minigpt.model.config import GPTConfig
from minigpt.model.gpt import GPT

__all__ = ["GPTConfig", "GPT", "__version__"]
