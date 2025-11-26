# utils/faiss_downloader.py

import os
import requests

GDRIVE_FILES = {
    "index.faiss": "12envQnrflaDqugR520Srfnqa7ZDiwXoy",      # 🔹 네가 준 ID
    "index.pkl":   "1o2iDL75VQeKtZcNnhU7EWcvlQ7rAK5-M",
}

def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def _save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

def download_from_gdrive(file_id: str, destination: str):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        response = session.get(
            URL,
            params={"id": file_id, "confirm": token},
            stream=True,
        )

    _save_response_content(response, destination)

    # 🔍 HTML 은 아닌지 체크
    with open(destination, "rb") as f:
        head = f.read(32)

    if head.startswith(b"<!DOCTYPE html") or head.lstrip().startswith(b"<html"):
        raise RuntimeError(
            f"Google Drive에서 {destination}를 받는 데 실패했습니다. "
            f"HTML 페이지가 내려왔습니다. 공유 링크/권한을 다시 확인해 주세요."
        )

def ensure_faiss_index(index_dir: str = "faiss_index"):
    os.makedirs(index_dir, exist_ok=True)

    for filename, file_id in GDRIVE_FILES.items():
        dest_path = os.path.join(index_dir, filename)

        need_download = True
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            try:
                with open(dest_path, "rb") as f:
                    head = f.read(32)
                if not (
                    head.startswith(b"<!DOCTYPE html")
                    or head.lstrip().startswith(b"<html")
                ):
                    need_download = False
            except Exception:
                need_download = True

        if need_download:
            print(f"[FAISS] {filename}을(를) Google Drive에서 다운로드합니다...")
            download_from_gdrive(file_id, dest_path)
        else:
            print(f"[FAISS] {filename} 이미 존재, 재사용합니다.")
