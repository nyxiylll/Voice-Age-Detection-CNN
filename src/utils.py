import pandas as pd 
import numpy as np 
import cv2, librosa, os
from tqdm import tqdm

# Base directory of THIS file (src/), so paths work regardless of cwd
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SRC_DIR, "..", "Data")

def load_csv(path, size):
    data = pd.read_csv(path)
    # Ensure we only work with rows that have both a filename and an age
    clean_data = data.dropna(subset=["age", "filename"]).copy()

    age_map = {
        "teens": 0, "twenties": 1, "thirties": 2, "fourties": 3,
        "fifties": 4, "sixties": 5, "seventies": 6, "eighties": 7
    }

    # Map labels and filter out any ages not in our dictionary
    clean_data['label_idx'] = clean_data["age"].map(age_map)
    clean_data = clean_data.dropna(subset=['label_idx']).copy()
    clean_data['label_idx'] = clean_data['label_idx'].astype(int)
    
    # Take the requested sample size
    final_data = clean_data.head(size)
    
    return final_data["filename"].values, final_data["label_idx"].values

def convert_train(filenames, labels):
    # CSV filenames are like 'cv-valid-train/sample-000000.mp3'
    # Actual disk path: Data/cv-valid-train/cv-valid-train/sample-000000.mp3
    # So basepath = Data/ parent, and filename provides 'cv-valid-train/sample-...'
    audio_basepath = os.path.join(_DATA_DIR, "cv-valid-train")
    save_basepath = os.path.join(_DATA_DIR, "Image")
    
    # Create subfolders for each class (0-7)
    for label_id in np.unique(labels):
        os.makedirs(os.path.join(save_basepath, str(label_id)), exist_ok=True)

    Max_length = 128

    for filename, label in tqdm(zip(filenames, labels), total=len(filenames)):
        try:
            audio_path = os.path.join(audio_basepath, filename)
            if not os.path.exists(audio_path):
                continue

            # 1. Load Audio
            audio, sr = librosa.load(audio_path, sr=16000)

            # 2. Generate Mel Spectrogram
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # 3. Handle Length (Padding/Truncating)
            if mel_db.shape[1] < Max_length:
                padding = Max_length - mel_db.shape[1]
                mel_db = np.pad(mel_db, ((0, 0), (0, padding)), mode="constant")
            else:
                mel_db = mel_db[:, :Max_length]

            # 4. Normalize to 0-255 for Image Conversion
            # We use min-max scaling to ensure the full range of the colormap is used
            mel_min, mel_max = mel_db.min(), mel_db.max()
            mel_norm = 255 * (mel_db - mel_min) / (mel_max - mel_min + 1e-6)
            mel_norm = mel_norm.astype(np.uint8)

            # 5. Apply Color Map (VIRIDIS is great for thermal-style data)
            mel_color = cv2.applyColorMap(mel_norm, cv2.COLORMAP_VIRIDIS)

            # 6. Save to Label-Specific Folder
            base_name = os.path.splitext(os.path.basename(filename))[0]
            img_path = os.path.join(save_basepath, str(label), f"{base_name}.png")
            
            cv2.imwrite(img_path, mel_color)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Done! Images saved in {save_basepath} categorized by folders 0-7.")

def convert_test(filenames, labels):
    # CSV filenames are like 'cv-valid-train/sample-000000.mp3'
    # Actual disk path: Data/cv-valid-train/cv-valid-train/sample-000000.mp3
    # So basepath = Data/ parent, and filename provides 'cv-valid-test/sample-...'
    audio_basepath = os.path.join(_DATA_DIR, "cv-valid-test")
    save_basepath = os.path.join(_DATA_DIR, "test")
    
    # Create subfolders for each class (0-7)
    for label_id in np.unique(labels):
        os.makedirs(os.path.join(save_basepath, str(label_id)), exist_ok=True)

    Max_length = 128

    for filename, label in tqdm(zip(filenames, labels), total=len(filenames)):
        try:
            audio_path = os.path.join(audio_basepath, filename)
            if not os.path.exists(audio_path):
                continue

            # 1. Load Audio
            audio, sr = librosa.load(audio_path, sr=16000)

            # 2. Generate Mel Spectrogram
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # 3. Handle Length (Padding/Truncating)
            if mel_db.shape[1] < Max_length:
                padding = Max_length - mel_db.shape[1]
                mel_db = np.pad(mel_db, ((0, 0), (0, padding)), mode="constant")
            else:
                mel_db = mel_db[:, :Max_length]

            # 4. Normalize to 0-255 for Image Conversion
            # We use min-max scaling to ensure the full range of the colormap is used
            mel_min, mel_max = mel_db.min(), mel_db.max()
            mel_norm = 255 * (mel_db - mel_min) / (mel_max - mel_min + 1e-6)
            mel_norm = mel_norm.astype(np.uint8)

            # 5. Apply Color Map (VIRIDIS is great for thermal-style data)
            mel_color = cv2.applyColorMap(mel_norm, cv2.COLORMAP_VIRIDIS)

            # 6. Save to Label-Specific Folder
            base_name = os.path.splitext(os.path.basename(filename))[0]
            img_path = os.path.join(save_basepath, str(label), f"{base_name}.png")
            
            cv2.imwrite(img_path, mel_color)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Done! Images saved in {save_basepath} categorized by folders 0-7.")

# Execution
if __name__ == "__main__":
    #csv_path = os.path.join(_DATA_DIR, "cv-valid-train.csv")
    #filenames, labels = load_csv(csv_path, 73768)
    #convert_train(filenames, labels)

    csv_path = os.path.join(_DATA_DIR, "cv-valid-test.csv")
    filenames, labels = load_csv(csv_path, 1542)
    convert_test(filenames, labels)
    