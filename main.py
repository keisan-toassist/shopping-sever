from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from datetime import datetime
import asyncio
import time
import os
import base64
import io
import json
import urllib.request
import urllib.parse
import ssl

import certifi
from PIL import Image

app = FastAPI()

# 写真を保存するフォルダ（このファイルと同じ場所の uploads/）
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)   # フォルダが無ければ作る


# APIキーは「環境変数」優先、無ければローカルの鍵ファイルから読む
# （Render等の本番＝環境変数、手元＝鍵ファイル。コードにもアプリにも埋め込まない）
def _load_key(env_name: str, filename: str) -> str:
    v = os.environ.get(env_name)
    if v:
        return v.strip()
    f = Path(__file__).parent / filename
    return f.read_text().strip() if f.exists() else ""


OCR_API_KEY = _load_key("OCR_API_KEY", "ocr_api_key.txt")
OCR_API_URL = "https://api.ocr.space/parse/image"

# SSL証明書の検証に certifi の証明書を使う（macOSのPythonでの検証エラー対策）
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# Gemini（無料・写真を見て商品名を特定する用）
GEMINI_API_KEY = _load_key("GEMINI_API_KEY", "gemini_api_key.txt")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Yahoo!ショッピング（無料・商品名から最安値を検索）
YAHOO_APP_ID = _load_key("YAHOO_APP_ID", "yahoo_app_id.txt")
YAHOO_API_URL = "https://shopping.yahooapis.jp/ShoppingWebService/V3/itemSearch"


def _resize_jpeg(image_bytes: bytes, max_side: int = 1500, limit: int = 1_000_000) -> bytes:
    """画像を縮小してJPEGバイト列にする（長辺max_side・limit以下を目安に品質調整）"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((max_side, max_side))   # 縦横比を保ったまま縮小
    quality = 80
    while True:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        jpeg = buf.getvalue()
        if len(jpeg) < limit or quality <= 30:
            return jpeg
        quality -= 10


def recognize_text(image_bytes: bytes) -> str:
    """写真の文字を OCR.space で読み取る（参考表示用・同期処理）"""
    try:
        jpeg = _resize_jpeg(image_bytes)
        b64 = base64.b64encode(jpeg).decode("ascii")
        payload = urllib.parse.urlencode({
            "apikey": OCR_API_KEY,
            "language": "jpn",        # 日本語として読む
            "OCREngine": "3",         # エンジン3＝高精度
            "base64Image": "data:image/jpeg;base64," + b64,
        }).encode("ascii")
        req = urllib.request.Request(OCR_API_URL, data=payload)
        with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as res:
            result = json.loads(res.read().decode("utf-8"))
        if result.get("IsErroredOnProcessing"):
            return "OCRエラー: " + str(result.get("ErrorMessage"))
        parsed = result.get("ParsedResults")
        if parsed and parsed[0].get("ParsedText", "").strip():
            return parsed[0]["ParsedText"].strip()
        return "(文字が見つかりませんでした)"
    except Exception as e:
        return f"認識に失敗: {e}"


def identify_product(image_bytes: bytes, ocr_hint: str = "") -> str:
    """写真を直接 Gemini に見せて商品名を1つ特定（画像認識・最大3回リトライ）"""
    try:
        jpeg = _resize_jpeg(image_bytes)
        b64 = base64.b64encode(jpeg).decode("ascii")
    except Exception as e:
        return f"画像の準備に失敗: {e}"

    prompt = ("この写真は、私が買いたい商品です。写っている商品を特定し、"
              "通販サイトで検索するための短い商品名（ブランド名＋商品の一般名）を1つだけ返してください。"
              "宣伝文句・容量・型番は除き、商品名だけを返してください。説明やマークダウンは不要です。")
    # OCRで何か読めていれば、ヒントとして添える（エラー文言は渡さない）
    if ocr_hint and not ocr_hint.startswith(("OCRエラー", "認識に失敗", "(文字")):
        prompt += f"\n（参考：パッケージから読めた文字: {ocr_hint}）"

    body = json.dumps({
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
        ]}],
        "generationConfig": {"temperature": 0.2},
    }).encode("utf-8")

    last_err = ""
    for _ in range(3):   # 503など一時的エラーに備えて最大3回試す
        try:
            req = urllib.request.Request(GEMINI_API_URL, data=body, headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY,
            })
            with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as res:
                r = json.loads(res.read().decode("utf-8"))
            return r["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            last_err = str(e)
            time.sleep(1.5)
    return f"商品名の特定に失敗: {last_err}"


def _yahoo_search(query: str) -> list:
    """Yahoo!ショッピングを安い順に検索して hits リストを返す"""
    params = urllib.parse.urlencode({
        "appid": YAHOO_APP_ID,
        "query": query,
        "sort": "+price",   # 価格の安い順
        "results": 3,
    })
    with urllib.request.urlopen(YAHOO_API_URL + "?" + params, timeout=30, context=SSL_CONTEXT) as res:
        r = json.loads(res.read().decode("utf-8"))
    return r.get("hits") or []


def search_price(product_name: str) -> dict:
    """商品名で最安値を検索。0件なら短いキーワードで再検索（同期処理）"""
    try:
        hits = _yahoo_search(product_name)
        if not hits:
            # 具体的すぎて0件のとき、ブランド＋商品種別で再検索
            words = product_name.split()
            if len(words) >= 2:
                hits = _yahoo_search(words[0] + " " + words[-1])
        if not hits:
            return {"price": None, "name": "", "store": "", "url": ""}
        top = hits[0]   # 安い順の先頭＝最安
        return {
            "price": top.get("price"),
            "name": top.get("name", ""),
            "store": (top.get("seller") or {}).get("name", ""),
            "url": top.get("url", ""),
        }
    except Exception as e:
        return {"price": None, "name": f"価格検索エラー: {e}", "store": "", "url": ""}


# 動作確認用：ブラウザで開くと {"status": "ok"} が返る
@app.get("/")
def health():
    return {"status": "ok", "message": "server is running"}


# 写真を受け取って保存し、商品名・最安値・楽天リンクを返す
@app.post("/upload")
async def upload(photo: UploadFile = File(...)):
    data = await photo.read()                 # 送られてきた写真の中身を読む
    size_kb = len(data) // 1024               # サイズ（KB）を計算

    # 保存ファイル名は重複しないよう日時を付ける
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = UPLOAD_DIR / f"{stamp}.jpg"
    save_path.write_bytes(data)               # 写真を保存

    # 文字を読み取る（参考表示用）＆ 写真を直接Geminiに見せて商品名を特定
    text = await asyncio.to_thread(recognize_text, data)
    product = await asyncio.to_thread(identify_product, data, text)
    # 商品名で最安値を検索（Yahoo!）
    cheapest = await asyncio.to_thread(search_price, product)
    # 楽天で購入するための検索リンク（API不要）
    rakuten_url = "https://search.rakuten.co.jp/search/mall/" + urllib.parse.quote(product) + "/"

    print(f"保存しました: {save_path.name} ({size_kb} KB)")
    print(f"読み取り結果(OCR):\n{text}")
    print(f"商品名(Gemini vision): {product}")
    print(f"最安値(Yahoo!): {cheapest}")
    print(f"楽天検索URL: {rakuten_url}")
    return {
        "status": "ok",
        "message": "受け取りました",
        "saved_as": save_path.name,
        "size_kb": size_kb,
        "recognized_text": text,
        "product_name": product,
        "cheapest": cheapest,
        "rakuten_url": rakuten_url,
    }
