# BERT
It stands for *Bidirectional Encoder Representations from Transformers*. You can read about it [here](https://en.wikipedia.org/wiki/BERT_(language_model)).

## How to run

### Training
* It uses the csv files in the `/dataset` in the root directory by default so please maintain the relative path or else you may edit the `TRAIN_CSV_PATH` variable in `config.py` file to use a different dataset.
* Install the requirements

```
pip install -r requirements.txt
```
* Train by running the `train.py` file.

### Prediction
* If you trained the model by fine-tuning it, the model weights are saved in `saved_models` directory as `model.bin`. The `predict.py` script, by default uses the `model.bin` file so you will have no issues running it.

* You can also download (400 MB) and use pretrained weights that have been trained on this dataset by reading the instructions in `saved_models/README.md`.

### Configuration

You can modify parameters in the `config.py` file.