import torch

from config import config
from model import BERTBaseUncased

PATH = "./saved_models/model.bin"

def predict(sentence, model, device):
    model.eval()
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation = True,
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(device, dtype=torch.long)
    token_type_ids = token_type_ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)

    outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
    
    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return { 
            'positive': outputs[0][0], 
            'negative': 1 - outputs[0][0],
           }

if __name__ == "__main__":
    model = BERTBaseUncased()
    model.load_state_dict(torch.load(PATH))

    sentence = "I love the weather but hated the concert"
    predict(sentence, model, device="cuda")