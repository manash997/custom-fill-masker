import torch
from transformers import AutoModelForMaskedLM,AutoTokenizer
#from transformers import BertForMaskedLM
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained('xlm-roberta-base')