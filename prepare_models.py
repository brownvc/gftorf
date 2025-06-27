import os
import gdown
import zipfile

output_dir = "output/pretrained_models"
os.makedirs(output_dir, exist_ok=True)

models = {
    "ftorf_real_scenes": "https://drive.google.com/uc?id=1qrBbsKvf6vborl-q219P9ER07vsypyKS",
    "ftorf_synthetic_scenes": "https://drive.google.com/uc?id=1gIXcLLUCHUxALTko4v5MmPOgZ32OptYa",
    "torf_scenes": "https://drive.google.com/uc?id=1sw-adMvqQfqUKhbwL792quxrAL3rHrDY",
}

for file_name, url in models.items():
    output_file = os.path.join(output_dir, file_name)
    gdown.download(url, output=output_file+".zip", quiet=False)
    
    with zipfile.ZipFile(output_file+".zip", 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    os.remove(output_file+".zip")
