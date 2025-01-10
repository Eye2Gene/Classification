import os
import sys
import tarfile

drive_url = "https://drive.google.com/uc?id=19kzVda8iIKsTxrjybfpk6EQstgUIVcBE"
output_fname = "stylegan2_synthetic_100perclass_onefold.tar.gz"
output_path = os.path.join("datasets", output_fname)

try:
    import gdown
except:
    print("Missing gdown package, please install using: \n        python3 -m pip install gdown")
    exit(1)

print("Attempting to retrieve file from Google Drive...")
gdown.download(drive_url, output_path, quiet=False)

print("Extracting tarfile")
with tarfile.open(output_path) as tar:
    tar.extractall(path="datasets")
