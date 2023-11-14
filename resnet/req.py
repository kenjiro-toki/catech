import requests

# APIのURLを設定
api_url = "http://127.0.0.1:8000/predict/"  # APIのURLを適切に設定する

# 画像ファイルのパスを指定
image_path = input()

# 画像ファイルをバイナリ形式で読み込む
with open(image_path, "rb") as f:
    image_data = f.read()

# POSTリクエストを送信
files = {"file": ("image.jpg", image_data, "image/jpeg")}
response = requests.post(api_url, files=files)

# レスポンスの内容を確認
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print("APIリクエストが失敗しました。")
