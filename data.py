# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="deepseek-ai/deepseek-coder-6.7b-instruct")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)