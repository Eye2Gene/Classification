import os
import sys
import zipfile

DRIVE_URL = "https://drive.google.com/uc?id=1Y-nGfxzcmRsWwIaCvdaUehLUVnSianBH"
OUTPUT_DIR = "./example_data"
ZIP_NAME = "synthetic_dataset.zip"
output_path = os.path.join(OUTPUT_DIR, ZIP_NAME)

try:
    import gdown
except:
    print("Missing gdown package, please install using: \n        python3 -m pip install gdown")
    exit(1)

print("Attempting to retrieve file from Google Drive...")
# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    print("Attempting to retrieve folder from Google Drive...")
    gdown.download(url=DRIVE_URL, output=output_path, quiet=False)
    print(f"Download complete. Files saved to {OUTPUT_DIR}")

except Exception as e:
    print(f"Error downloading folder: {e}")
    sys.exit(1)
    
print("Extracting zip file")
with zipfile.ZipFile(output_path, 'r') as zip_ref:
    zip_ref.extractall(OUTPUT_DIR)




