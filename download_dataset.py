import os
import gdown
import zipfile
import tarfile
import sys
# import tdqm

URL_ALL = "https://drive.google.com/drive/folders/1BgosYlaRkQkSa43Jpgb6hoGK88n3bXLL?usp=sharing"
OUT_DIR = os.makedirs(os.path.join(".", "dataset"), exist_ok=True)

skip_list = ['seq_3.tar.gz', 'seq_4.tar.gz', 'seq_5.tar.gz', 'seq_7.tar.gz']

def main():
    try:
        print("Downloading dataset...")
        # gdown.download_folder(url=URL_ALL, output=OUT_DIR, quiet=False)
        print("Dataset downloaded!")

        print("Decompressing files...")
    
        for root, dirs, files in os.walk(os.path.join(".", "SonarOdometryDataset")):
            for file in files:

                if file.endswith(".tar.gz"):
                    filepath = os.path.join(root, file)
                
                    with tarfile.open(filepath, "r:gz") as tar:
                        tar.extractall(path=root)

                    os.remove(filepath)

        print("Dataset decompressed!")

    except Exception as e:
    
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()