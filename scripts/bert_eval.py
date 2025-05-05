from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np

class BERTScorer:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()

    def score_text(self, text: str) -> float:
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) < 5:
            return 0.0
        scores = []
        for i, token in enumerate(tokens):
            masked = tokens[:]
            masked[i] = self.tokenizer.mask_token
            inputs = self.tokenizer(" ".join(masked), return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits
            mask_index = (inputs.input_ids[0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()
            true_id = self.tokenizer.convert_tokens_to_ids(token)
            prob = torch.softmax(logits[0, mask_index], dim=0)[true_id].item()
            scores.append(prob)
        return float(np.mean(scores))