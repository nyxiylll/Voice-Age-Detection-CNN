# 🎙️ Voice Age Detection CNN

A deep learning project that predicts a speaker's **age group** from their voice using **Mel Spectrogram** images and a custom **Convolutional Neural Network (CNN)** built with PyTorch.

---

## 🧠 How It Works

```
Microphone / Audio File
        │
        ▼
  Mel Spectrogram  (128 × 128)
        │
        ▼
  VIRIDIS Colour Map  →  RGB Image
        │
        ▼
  CNN (3 Conv layers + 2 FC layers)
        │
        ▼
  Predicted Age Group  (8 classes)
```

The pipeline converts raw audio into a **colourised Mel spectrogram image**, then feeds it into a CNN that classifies the speaker into one of 8 age groups.

---

## 🏷️ Age Group Classes

| Label | Age Range |
|-------|-----------|
| 0 | Teens (13–19) |
| 1 | Twenties (20–29) |
| 2 | Thirties (30–39) |
| 3 | Fourties (40–49) |
| 4 | Fifties (50–59) |
| 5 | Sixties (60–69) |
| 6 | Seventies (70–79) |
| 7 | Eighties (80–89) |

---

## 📁 Project Structure

```
Age Detection/
├── src/
│   ├── main.py          # Training script
│   ├── model.py         # CNN architecture
│   ├── dataset.py       # DataLoader config
│   ├── utils.py         # Audio → spectrogram image conversion
│   ├── evaluation.py    # Model evaluation on test set
│   └── live.py          # Real-time mic / file inference
├── Data/                # ⚠️ Not included — download separately
│   ├── cv-valid-train/
│   ├── cv-valid-test/
│   └── ... (CommonVoice CSVs and audio)
├── Model/               # ⚠️ Not included — generated after training
│   └── colour_model.pth
├── .gitignore
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/nyxiylll/Voice-Age-Detection-CNN.git
cd Voice-Age-Detection-CNN
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision torchaudio
pip install librosa opencv-python pillow pandas tqdm sounddevice
```

---

## 📦 Dataset

This project uses the **Mozilla Common Voice** dataset.

1. Download it from [https://commonvoice.mozilla.org/en/datasets](https://commonvoice.mozilla.org/en/datasets)
2. Place the extracted files inside a `Data/` folder at the project root:

```
Data/
├── cv-valid-train/        ← audio .mp3 files
├── cv-valid-train.csv
├── cv-valid-test/
├── cv-valid-test.csv
└── ...
```

---

## 🔄 Workflow

### Step 1 — Prepare Data (Audio → Spectrogram Images)

Run from the `src/` folder:

```bash
cd src
python utils.py
```

This reads the CSV, loads each `.mp3`, generates a **Mel spectrogram**, applies the **VIRIDIS colour map**, and saves `128×128` PNG images into `Data/Image/<class_id>/`.

### Step 2 — Train the Model

```bash
python main.py
```

Trains the CNN for 10 epochs. The best model checkpoint is saved to `Model/colour_model.pth`.

### Step 3 — Evaluate

```bash
python evaluation.py
```

Runs inference on the test set and prints accuracy.

---

## 🎤 Live Inference

Predict age group **in real time** from your microphone or an audio file:

```bash
# Record 5 seconds from mic (default)
python live.py

# Record a custom duration
python live.py --duration 3

# Predict from an existing audio file
python live.py --file path/to/audio.wav
```

**Example output:**
```
========================================
  Predicted age group : Twenties (20–29)
  Confidence          : 72.4%
========================================

Probabilities per class:
  Teens    (13–19)  ██                             4.1%
  Twenties (20–29)  █████████████████████         72.4% ◄
  Thirties (30–39)  ████                          12.3%
  ...
```

---

## 🏗️ Model Architecture

```
Input: 3 × 128 × 128 (RGB Mel spectrogram)

Conv2d(3 → 16, 3×3)  + ReLU + MaxPool2d(2)   → 16 × 64 × 64
Conv2d(16 → 32, 3×3) + ReLU + MaxPool2d(2)   → 32 × 32 × 32
Conv2d(32 → 64, 3×3) + ReLU + MaxPool2d(2)   → 64 × 16 × 16

Flatten → 16384
Linear(16384 → 128)  + ReLU
Linear(128 → 8)

Output: 8-class logits
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Model training & inference |
| Librosa | Audio loading & Mel spectrogram |
| OpenCV | VIRIDIS colour mapping |
| Torchvision | Image transforms & DataLoader |
| sounddevice | Real-time microphone recording |
| Mozilla Common Voice | Training dataset |

---

## 📄 License

This project is for educational purposes. Dataset usage is subject to [Mozilla Common Voice Terms](https://commonvoice.mozilla.org/en/terms).
