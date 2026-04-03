"""
live.py — Real-time age group prediction from microphone input.

Pipeline:
  Mic audio  →  Mel Spectrogram  →  VIRIDIS colour image  →  Model  →  Age label

Usage (from the src/ folder):
    python live.py                    # record 5 seconds from mic, then predict
    python live.py --duration 3       # record 3 seconds
    python live.py --file path.wav    # predict from an existing audio file
"""

import argparse
import os
import sys
import numpy as np
import cv2
import librosa
import torch
from torchvision import transforms
from PIL import Image

# ── make sure imports work whether run from src/ or project root ──────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)
from model import Model

# ── constants (must match training) ──────────────────────────────────────────
SAMPLE_RATE  = 16000
MAX_LENGTH   = 128          # time-axis frames
N_MELS       = 128          # frequency bins
MODEL_PATH   = os.path.join(_SRC_DIR, "..", "Model", "colour_model.pth")

AGE_LABELS = {
    0: "Teens    (13–19)",
    1: "Twenties (20–29)",
    2: "Thirties (30–39)",
    3: "Fourties (40–49)",
    4: "Fifties  (50–59)",
    5: "Sixties  (60–69)",
    6: "Seventies(70–79)",
    7: "Eighties (80–89)",
}

# ── transforms (must match dataset.py) ───────────────────────────────────────
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ── load model ────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model = Model(num_classes=8).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[✓] Model loaded from {os.path.abspath(MODEL_PATH)}  (device: {device})")
    return model

# ── audio → spectrogram image tensor ─────────────────────────────────────────
def audio_to_tensor(audio: np.ndarray, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Convert a raw audio array to the model-ready image tensor."""

    # 1. Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 2. Pad / truncate to fixed width
    if mel_db.shape[1] < MAX_LENGTH:
        pad = MAX_LENGTH - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant")
    else:
        mel_db = mel_db[:, :MAX_LENGTH]

    # 3. Normalise to 0-255
    mel_min, mel_max = mel_db.min(), mel_db.max()
    mel_norm = 255 * (mel_db - mel_min) / (mel_max - mel_min + 1e-6)
    mel_norm = mel_norm.astype(np.uint8)

    # 4. Apply VIRIDIS colour map  (same as training)
    mel_color_bgr = cv2.applyColorMap(mel_norm, cv2.COLORMAP_VIRIDIS)   # BGR
    mel_color_rgb = cv2.cvtColor(mel_color_bgr, cv2.COLOR_BGR2RGB)      # RGB

    # 5. PIL → transform → tensor  [1, 3, 128, 128]
    pil_img = Image.fromarray(mel_color_rgb)
    tensor  = _transform(pil_img).unsqueeze(0)   # add batch dim
    return tensor

# ── record from microphone ────────────────────────────────────────────────────
def record_audio(duration: float = 5.0) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError:
        print("[✗] sounddevice not installed.  Run:  pip install sounddevice")
        sys.exit(1)

    print(f"\n🎤  Recording for {duration} second(s)... speak now!")
    audio = sd.rec(int(duration * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype="float32")
    sd.wait()
    print("   Recording done.\n")
    return audio.flatten()

# ── predict ───────────────────────────────────────────────────────────────────
def predict(model: Model, audio: np.ndarray) -> None:
    tensor = audio_to_tensor(audio).to(device)

    with torch.no_grad():
        logits = model(tensor)                     # [1, 8]
        probs  = torch.softmax(logits, dim=1)[0]   # [8]
        pred   = torch.argmax(probs).item()

    print("=" * 40)
    print(f"  Predicted age group : {AGE_LABELS[pred]}")
    print(f"  Confidence          : {probs[pred]*100:.1f}%")
    print("=" * 40)
    print("\nProbabilities per class:")
    for idx, (label, prob) in enumerate(zip(AGE_LABELS.values(), probs.tolist())):
        bar  = "█" * int(prob * 30)
        mark = " ◄" if idx == pred else ""
        print(f"  {label}  {bar:<30} {prob*100:5.1f}%{mark}")

# ── CLI entry point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Live age-group predictor")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Seconds to record from microphone (default: 5)")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to an audio file to predict instead of recording")
    args = parser.parse_args()

    model = load_model()

    if args.file:
        print(f"[→] Loading audio from: {args.file}")
        audio, sr = librosa.load(args.file, sr=SAMPLE_RATE)
    else:
        audio = record_audio(args.duration)
        sr    = SAMPLE_RATE

    predict(model, audio)

if __name__ == "__main__":
    main()
