"""Implement text encoder using multiple embedding models."""

from typing import Dict

from sentence_transformers import SentenceTransformer

class TextEncoder:
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def encode(self, text: str):
        pass

    def batch_encode(self, texts: list[str]):
        pass

class XLMROBERTaTextEncoder(TextEncoder):
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        super().__init__("xlm-roberta-base")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
    
    def encode(self, text: str):
        # TODO still not sure if this is correct
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        embedding = outputs.logits[0, -1, :]
        return embedding
    
    def batch_encode(self, texts: list[str]):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.logits.mean(dim=1)
        return embeddings

class SentenceTransformerTextEncoder(TextEncoder):
    def __init__(self, model_name: str, precision: str = "bf16"):
        super().__init__(model_name)
        self.precision = precision.lower()
        self.model = SentenceTransformer(self.model_name)
        self._apply_precision()

    def _apply_precision(self):
        if self.precision == "fp16":
            self.model = self.model.half()
        elif self.precision == "bf16":
            self.model = self.model.bfloat16()
        elif self.precision != "fp32":
            raise ValueError(
                f"Unsupported embedding precision '{self.precision}'. "
                "Use one of: fp32, fp16, bf16."
            )

    def encode(self, text: str):
        return self.model.encode([text])[0]

    def batch_encode(self, texts: list[str]):
        return self.model.encode(texts)


_EMBEDDING_MODEL_REGISTRY: Dict[str, str] = {
    # aliases for convenience
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "e5-base-v2": "intfloat/e5-base-v2",
    "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
}


def get_available_embedding_model_names() -> list[str]:
    return sorted(_EMBEDDING_MODEL_REGISTRY.keys())


def get_available_embedding_precisions() -> list[str]:
    return ["fp32", "fp16", "bf16"]


def load_text_encoder(model_name: str, precision: str = "fp32") -> TextEncoder:

    normalized = model_name.strip().lower()
    if normalized not in _EMBEDDING_MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported embedding model '{model_name}'. "
            f"Available: {get_available_embedding_model_names()}"
        )

    resolved = _EMBEDDING_MODEL_REGISTRY[normalized]
    return SentenceTransformerTextEncoder(resolved, precision=precision)


if __name__ == "__main__":
    text_encoder = load_text_encoder("qwen3-embedding-0.6b")
    # text_encoder = XLMROBERTaTextEncoder()
    print(text_encoder.encode("Hello, world!").shape)
    print(text_encoder.batch_encode(["Hello, world!"]*10).shape)
