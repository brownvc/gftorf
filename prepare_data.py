import os
import gdown
import zipfile
import shutil

target_dir = "data"

# Step 1: Download F-ToRF data
# Please download the `real_scenes.zip` and `synthetic_scenes.zip` files from 
# https://1drv.ms/f/c/4dd35d8ee847a247/EsiF6mb15ZlKlTZmg8N_OIcBCaQGUmWWVNOldMTaRsQXeQ?e=eIy7Rz
# to the data/ directory.

real_scenes_zip = "data/real_scenes.zip"
with zipfile.ZipFile(real_scenes_zip, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(target_dir, "ftorf_real_scenes"))
ftorf_real_scenes = ["baseball", "fan", "jacks1", "pillow", "target1"]
for scene in ftorf_real_scenes:
    with zipfile.ZipFile(os.path.join(target_dir, "ftorf_real_scenes", f"{scene}.zip"), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(target_dir, "ftorf_real_scenes", f"{scene}"))
    os.remove(os.path.join(target_dir, "ftorf_real_scenes", f"{scene}.zip"))
os.remove(os.path.join(target_dir, "ftorf_real_scenes", f"data_color25.zip"))

synthetic_scenes_zip = "data/synthetic_scenes.zip"
with zipfile.ZipFile(synthetic_scenes_zip, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(target_dir, "ftorf_synthetic_scenes"))
for folder in os.listdir(os.path.join(target_dir, "ftorf_synthetic_scenes")):
    if folder.startswith("occlusion_"):
        folder_path = os.path.join(target_dir, "ftorf_synthetic_scenes", folder)
        shutil.rmtree(folder_path)
        print(f"Removed folder: {folder_path}")

# Step 2: Download ToRF data
# Please download the `copier`, `cupboard`, `deskbox`, `phonebooth`, and `studbook` folders from
# https://drive.google.com/drive/folders/18QsJeCYjqtfYCtduzeDMuulgW6EpF4wO?usp=sharing
# to the data/ directory.
# You should have file names like `copier-20250515T130607Z-1-001.zip`, ... in the data/ directory. 

torf_scenes = ["copier", "cupboard", "deskbox", "phonebooth", "studybook"]
for fp in os.listdir("data"):
    if fp.split("-")[0] in torf_scenes:
        file_path = os.path.join("data", fp)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(target_dir, "torf_scenes"))
        scene_path = os.path.join(target_dir, "torf_scenes", fp.split("-")[0])
        for folder in os.listdir(scene_path):
            if not folder.endswith(".npy") and not folder.startswith("cams"):
                files = sorted([f for f in os.listdir(os.path.join(scene_path, folder)) if f[:4].isdigit()])
                for file in files:
                    old_path = os.path.join(scene_path, folder, file)
                    fid = (int(file[:4]) - 1) if (int(file[:4]) - 1) >= 0 else 0
                    new_name = f"{fid:04d}"+file[4:]
                    new_path = os.path.join(scene_path, folder, new_name)
                    os.rename(old_path, new_path)
                    print(f"Renamed {old_path} to {new_path}")

# Step 3: Merge with auxiliary files for GF-ToRF, https://drive.google.com/file/d/1BYlRveLFE5hyYviWs8ZJe1aU8nGdTWgX/view?usp=sharing
auxiliary_zip_url = "https://drive.google.com/uc?id=1BYlRveLFE5hyYviWs8ZJe1aU8nGdTWgX"
gdown.download(auxiliary_zip_url, quiet=False)
with zipfile.ZipFile("gftorf_data_aux_files.zip", 'r') as zip_ref:
    zip_ref.extractall(".")
os.remove("gftorf_data_aux_files.zip")

def merge_folders(src_root, dst_root):
    for dirpath, _, filenames in os.walk(src_root):
        relative_path = os.path.relpath(dirpath, src_root)
        dst_path = os.path.join(dst_root, relative_path)
        os.makedirs(dst_path, exist_ok=True)

        for filename in filenames:
            src_file = os.path.join(dirpath, filename)
            dst_file = os.path.join(dst_path, filename)

            if os.path.exists(dst_file):
                print(f"[OVERWRITE] {dst_file}")
            shutil.copy2(src_file, dst_file)

src = "gftorf_data_aux_files"
dst = "data"
merge_folders(src, dst)
print("✅ Merge completed.")

shutil.rmtree(src)
print("✅ Auxiliary files removed.")


