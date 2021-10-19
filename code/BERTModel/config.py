import transformers

class CONFIG:
    DEVICE = "cuda"
    MAX_LEN = 64
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 4
    EPOCHS = 20
    BERT_PATH = "bert-base-uncased"
    MODEL_PATH = "./saved_models/model.bin"
    TRAIN_CSV_PATH = "../../dataset/train.csv"
    TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)

config = CONFIG()