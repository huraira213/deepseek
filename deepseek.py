from models.model import GenerativeModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

class DeepSeekModel(GenerativeModel):
    """
    DeepSeek model placeholder.
    
    Initially uses GPT-2 as a small test model. Can later
    switch to DeepSeek 7B base for production.
    """

    MODEL_NAME = "gpt2"

    def __init__(self, **kwargs):
        self.tokenizer = None
        self.model = None
        self.initialized = False

    def initialize(self):
        """Load the model and tokenizer once."""
        if self.initialized:
            return

        print(f"[DeepSeek] Loading model '{self.MODEL_NAME}' ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to("cpu")
        self.initialized = True
        print("[DeepSeek] Model loaded successfully!")

    def encode_text(self, text, **kwargs):
        """Generate text from a single input string."""
        self.initialize()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(inputs["input_ids"], max_new_tokens=64)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def encode_text_batch(self, texts, **kwargs):
        """Generate text for multiple inputs."""
        return [self.encode_text(t) for t in texts]

    def decode_text(self, **kwargs):
        """Placeholder for decode_text."""
        self.initialize()
        return "decoded placeholder"

    def decode_text_batch(self, **kwargs):
        """Placeholder for decode_text_batch."""
        self.initialize()
        return ["decoded placeholder" for _ in range(2)]

    def encode_image(self, **kwargs):
        """Placeholder for image encoding."""
        self.initialize()
        return [0.0, 0.0, 0.0, 0.0]

    def encode_image_batch(self, **kwargs):
        """Placeholder for batch image encoding."""
        self.initialize()
        return [[0.0, 0.0, 0.0, 0.0]]

    def rerank_text(self, query, documents, **kwargs):
        """
        Rerank documents based on similarity to the query.
        Uses hidden states from the model as embeddings and cosine similarity.
        """
        self.initialize()

        # Encode query
        query_inputs = self.tokenizer(query, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            query_outputs = self.model(**query_inputs, output_hidden_states=True)
        query_emb = query_outputs.hidden_states[-1].mean(dim=1)

        # Encode documents and compute similarity
        scores = []
        for idx, doc in enumerate(documents):
            doc_inputs = self.tokenizer(doc, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                doc_outputs = self.model(**doc_inputs, output_hidden_states=True)
            doc_emb = doc_outputs.hidden_states[-1].mean(dim=1)
            score = F.cosine_similarity(query_emb, doc_emb).item()
            scores.append({"text": doc, "score": score, "index": idx})

        # Sort by similarity (highest first)
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores
