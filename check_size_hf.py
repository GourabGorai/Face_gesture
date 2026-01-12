import requests

url = "https://huggingface.co/datasets/testdummyvt/hagRIDv2_512px_10GB/resolve/main/yolo_format.zip?download=true"
try:
    response = requests.head(url, allow_redirects=True)
    size = int(response.headers.get('content-length', 0))
    print(f"URL: {response.url}")
    print(f"Size: {size / (1024*1024*1024):.2f} GB")
except Exception as e:
    print(f"Error: {e}")
