import os
import gdown
import zipfile

output_dir = "output/pretrained_models"
os.makedirs(output_dir, exist_ok=True)

models = {
    "ftorf_real_scenes": "https://drive.google.com/uc?id=1iFGHI4ZwaLzDYZIRJCEUkPc5REWHWklW",
    "ftorf_synthetic_scenes": "https://drive.google.com/uc?id=1bY53ood_-RPm-HsFynkz5EwJxTgnqDfZ",
    "torf_scenes": "https://drive.google.com/uc?id=1JZjrkucOaz9Ci-0kammNdT-QB-J_vCfe",
}

for file_name, url in models.items():
    output_file = os.path.join(output_dir, file_name)
    gdown.download(url, output=output_file+".zip", quiet=False)
    
    with zipfile.ZipFile(output_file+".zip", 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    os.remove(output_file+".zip")