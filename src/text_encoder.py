'''Implement text encoder using multiple embedding models'''

from transformers import AutoTokenizer, AutoModelForMaskedLM
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

class MiniLMTextEncoder(TextEncoder):
    def __init__(self):
        super().__init__("sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name)
        # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        # model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    def encode(self, text: str):
        text = [text]
        return self.model.encode(text)[0]
    
    def batch_encode(self, texts: list[str]):
        return self.model.encode(texts)
    
    
if __name__ == "__main__":
    text_encoder = MiniLMTextEncoder()
    # text_encoder = XLMROBERTaTextEncoder()
    print(text_encoder.encode("Hello, world!").shape)
    print(text_encoder.batch_encode(["Hello, world!"]*10).shape)
