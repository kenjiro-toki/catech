from fastapi import FastAPI, UploadFile, File
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn  # nnをインポートすることを追加
from PIL import Image
import io
import numpy as np
import json


app = FastAPI()

# モデルを読み込む
model_path = 'fine_tuned_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = models.resnet18(pretrained=False)
net.fc = nn.Linear(net.fc.in_features, 10)  # 分類先のクラス数に対応する全結合層を設定
net.load_state_dict(torch.load(model_path, map_location=device))
net = net.to(device)
net.eval()

# クラス名の定義
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 画像の前処理関数
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    image = transform(image).unsqueeze(0)  # バッチ次元を追加
    return image.to(device)

# 予測関数
def predict(image):
    with torch.no_grad():
        output = net(image)
        probabilities = torch.softmax(output, dim=1)
        scores, indices = torch.topk(probabilities, k=1)
        scores = scores.cpu().numpy().tolist()[0]
        indices = indices.cpu().numpy().tolist()[0]
        classifications = [classes[i] for i in indices]
        return classifications, scores

# エンドポイント
@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    # 画像を読み込む
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # 画像を前処理してモデルに渡す
    image_tensor = preprocess_image(image)

    # 予測を実行
    classifications, scores = predict(image_tensor)

    # レスポンスを作成
    response = {
        "predictions": [
            {
                "classification_results": classifications,
                "score": scores
            }
        ]
    }
    return response
