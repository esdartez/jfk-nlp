from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
import re

class BERTClozeEvaluator:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()

    def score_text(self, text):
        tokens = self.tokenizer.tokenize(text)
        scores = []

        for i, token in enumerate(tokens):
            masked_tokens = tokens.copy()
            masked_tokens[i] = self.tokenizer.mask_token
            masked_input = self.tokenizer(" ".join(masked_tokens), return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**masked_input)
                logits = outputs.logits

            mask_token_index = (masked_input["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            token_prob = torch.softmax(logits[0, mask_token_index], dim=0)[token_id].item()
            scores.append(token_prob)

        return np.mean(scores), scores

if __name__ == "__main__":
    evaluator = BERTClozeEvaluator()
    test_text = "The presldent was shot in Dallas on November 22, 1963."
    mean_score, token_scores = evaluator.score_text(test_text)
    print(f"Mean BERT token confidence: {mean_score:.4f}")
