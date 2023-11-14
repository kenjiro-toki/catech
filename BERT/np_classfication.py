from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import torch
# 日本語の事前学習モデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

text = input()
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
#エンコーディング
encoding = tokenizer(
    text,
    padding = "longest",
    return_tensors="pt"
)
encoding = { k : v for k, v in encoding.items()}
# モデルのロード
bert_sc = BertForSequenceClassification.from_pretrained("model_transformers")

#BERTへ入力し分類スコアを得る
with torch.no_grad():
    output = bert_sc(**encoding)
classification_results = output.logits[0].argmax(-1).item()

if classification_results == 0:
    classification_results = "negative"
elif classification_results == 1:
    classification_results = "positive"
else:
    classification_results = "neutral"

print(classification_results)