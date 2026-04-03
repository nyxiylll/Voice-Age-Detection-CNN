import torch
from model import Model
from dataset import config

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Model().to(device)
model.load_state_dict(torch.load("../Model/colour_model.pth", map_location=device))

dataset, loader = config("../Data/test")


def evaluate():
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            pred = torch.argmax(outputs, dim=1)

            correct += (pred == labels).sum().item()   
            total += labels.size(0)

    accuracy = correct / total

    print("===== EVALUATION RESULT =====")
    print(f"Total Samples: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    


if __name__ == "__main__":
    print(device)
    evaluate()