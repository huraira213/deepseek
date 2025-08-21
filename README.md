
# DeepSeek Model (Placeholder)

## What’s this about
This repository contains a placeholder version of the DeepSeek model I’m working on for AgensAI.  
Currently it uses GPT-2 for testing, but later it will be switched to the full DeepSeek 7B base.

The main things it can do:

- Generate text from a single prompt or multiple prompts
- Placeholder decode functions (they don’t return real values yet)
- Placeholder image encoding (returns zeros)
- Rerank a list of documents based on similarity to a query
  

## How to run it
Make sure you have Python 3.9 or higher and install the required packages:

```bash
pip install torch transformers
```
## Then, you can use it like this:

```python
from deepseek import DeepSeekModel

# Create the model
model = DeepSeekModel()

# Generate text from one prompt
print(model.encode_text("Hello, how are you?"))

# Generate text from multiple prompts
print(model.encode_text_batch(["Hello", "Good morning"]))

# Rerank documents based on a query
query = "What is AgensAI?"
documents = [
    "AgensAI is an AI extension for Postgres.",
    "AgensAI allows to run language models inside Postgres.",
    "Hello world!"
]
print(model.rerank_text(query, documents))
```

**Notes**
This is a placeholder, so functions like decode_text and encode_image just return dummy values.

The model currently uses GPT-2; later it will be replaced with DeepSeek 7B.

The reranking function uses hidden states from the model and cosine similarity to sort documents by relevance.
