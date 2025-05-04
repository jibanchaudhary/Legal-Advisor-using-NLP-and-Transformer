from pathlib import Path
from transformers import BertForSequenceClassification, BertTokenizer, BartForConditionalGeneration, BartTokenizer,BertModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR/'advisory'/'_FINAL__fine_tuned_Legal_BERT'

legal_bert_model = BertForSequenceClassification.from_pretrained(str(MODEL_DIR))
legal_bert_tokenizer = BertTokenizer.from_pretrained(str(MODEL_DIR))

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
