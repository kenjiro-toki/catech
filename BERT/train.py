# -*- coding: utf-8 -*-
"""train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ONSFI7ebaFxFRSh3RDoKnCj7rTt5vnA5
"""

# 必要なライブラリのインストール
# !pip install transformers==4.18.0 fugashi==1.1.0 ipadic==1.0.0 pytorch-lightning==1.6.1

# インポート
import random
import glob
from tqdm import tqdm
import json
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl
import pickle

# 日本語の事前学習モデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# 日本語の事前学習モデルを利用したトークン化
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
# positive,negative,neutralの3つのラベル
bert_sc = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3
)
# gpuに載せる
# bert_sc = bert_sc.cuda()

# データセットの取得
# !wget https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/public/chABSA-dataset.zip
# !unzip chABSA-dataset.zip

# データセットの整形する関数
def create_rating(sentences):
    rating = []
    for obj in sentences:
        s = obj["sentence"]  #文章の取得
        op = obj["opinions"]  #options部分を取得
        # positive:+1,negative:-1
        porarity = 0
        for o in op:
            p = o["polarity"]
            if p == "positive":
                porarity += 1
            elif p == "negative":
                porarity -= 1
        rating.append((porarity, s))
    return rating   # ネガポジと文章を返す

# 全てのデータに対して整形し、ratingに格納
rating = []
for file in glob.glob('chABSA-dataset/*.json'):
    with open(file,"br") as f:
        j =  json.load(f)
    s = j["sentences"]
    rating += create_rating(s)

# polarityが正でpositive、負でnegative、0でneutralにする
dataset = []
for r in rating:
    text = r[1]
    rate = r[0]
    if rate > 0:
        labels = 1
    elif rate < 0:
        labels = 0
    else:
        labels = 2
    sample = {'text': text, 'labels': labels}
    dataset.append(sample)

# デバック用コード
"""
count_neutral = 0
for label in dataset:
  if label["labels"] == 2:
    count_neutral += 1

print(count_neutral)
"""

# 各データの形式を整える(トークンにする)
max_length =128
dataset_for_loader =[]
for sample in dataset:
    text = sample["text"]
    labels = sample["labels"]
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding = "max_length",
        truncation =True
    )
    encoding["labels"] =labels
    encoding = {k:torch.tensor(v) for k, v in encoding.items()}
    dataset_for_loader.append(encoding)

# データセットの分割
random.shuffle(dataset_for_loader)
n = len(dataset_for_loader)
n_train = int(n*0.6)
n_val = int(n*0.2)
dataset_train = dataset_for_loader[:n_train]
dataset_val = dataset_for_loader[n_train:n_train+n_val]
dataset_test = dataset_for_loader[n_train+n_val:] # テストデータ

# データセットからデータローダを作成
dataloader_train = DataLoader(
    dataset_train,batch_size=32,shuffle=True
)
dataloader_val = DataLoader(dataset_val, batch_size=256)
dataloader_test = DataLoader(dataset_test, batch_size=256)

# PyTorch Lightningモデル(ファインチューニングのclass)
class BertForSequenceClassification_pl(pl.LightningModule):

    def __init__(self, model_name, num_labels, lr):
        # model_name: Transformersのモデルの名前
        # num_labels: ラベルの数(3)
        # lr: 学習率

        super().__init__()

        # 引数のnum_labelsとlrを保存。
        # 例えば、self.hparams.lrでlrにアクセスできる。
        # チェックポイント作成時にも自動で保存される。
        self.save_hyperparameters()

        # BERTのロード
        self.bert_sc = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    # 学習データのミニバッチ(`batch`)が与えられた時に損失を出力する関数を書く。
    # batch_idxはミニバッチの番号であるが今回は使わない。
    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        loss = output.loss
        self.log('train_loss', loss) # 損失を'train_loss'の名前でログをとる。
        return loss

    # 検証データのミニバッチが与えられた時に、
    # 検証データを評価する指標を計算する関数を書く。
    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss) # 損失を'val_loss'の名前でログをとる。

    # テストデータのミニバッチが与えられた時に、
    # テストデータを評価する指標を計算する関数を書く。
    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels') # バッチからラベルを取得
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        num_correct = ( labels_predicted == labels ).sum().item()
        accuracy = num_correct/labels.size(0) #精度
        self.log('accuracy', accuracy) # 精度を'accuracy'の名前でログをとる。

    # 学習に用いるオプティマイザを返す関数を書く。
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# 学習時にモデルの重みを保存する条件を指定
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_weights_only=True,
    dirpath='model/',
)

# 学習の方法を指定
trainer = pl.Trainer(
    gpus=1,
    max_epochs=10,
    callbacks = [checkpoint]
)

# PyTorch Lightningモデルのロード
model = BertForSequenceClassification_pl(
    MODEL_NAME, num_labels=3, lr=1e-5
)

# ファインチューニングを行う。
trainer.fit(model, dataloader_train, dataloader_val)

# 確認用
"""
best_model_path = checkpoint.best_model_path # ベストモデルのファイル
print('ベストモデルのファイル: ', checkpoint.best_model_path)
print('ベストモデルの検証データに対する損失: ', checkpoint.best_model_score)
"""

# Commented out IPython magic to ensure Python compatibility.
# 損失の時間変化
"""
# %load_ext tensorboard
# %tensorboard --logdir ./
"""

# 評価
"""
test = trainer.test(dataloaders=dataloader_test)
print(f'Accuracy: {test[0]["accuracy"]:.2f}')
"""

# PyTorch Lightningモデルのロード
best_model_path = checkpoint.best_model_path # ベストモデルのファイル
model = BertForSequenceClassification_pl.load_from_checkpoint(
    best_model_path
)

# モデルを保存する
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# テスト用
"""
# Transformers対応のモデルを./model_transformesに保存
model.bert_sc.save_pretrained('./model_transformers')
text = "以上の結果、当連結会計年度における売上高1,785百万円(前年同期比357百万円減、16.7％減)、営業損失117百万円(前年同期比174百万円減、前年同期　営業利益57百万円)、経常損失112百万円(前年同期比183百万円減、前年同期　経常利益71百万円)、親会社株主に帰属する当期純損失58百万円(前年同期比116百万円減、前年同期　親会社株主に帰属する当期純利益57百万円)となりました"
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
#エンコーディング
encoding = tokenizer(
    text,
    padding = "longest",
    return_tensors="pt"
)
encoding = { k : v for k, v in encoding.items()}
# モデルのロード
bert_sc = BertForSequenceClassification.from_pretrained("/content/model_transformers")

#BERTへ入力し分類スコアを得る
with torch.no_grad():
    output = bert_sc(**encoding)
scores = output.logits.argmax(-1)

print(scores)
"""