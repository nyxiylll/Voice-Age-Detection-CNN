import torch 
from utils import load_csv,convert as convert_data
from dataset import config
from tqdm import tqdm
from model import Model

################################

csv = r"C:\Users\ASUS\Desktop\Age Detection\Data\cv-valid-test.csv"#"../Data/cv-valid-train.csv"
images_path = "../Data/Image"

################################

def prepare_data():
    filenames, labels = load_csv(csv, 5000)
    convert_data(filenames, labels)

def run():
    EPOCHS = 10

    dataset, loader = config(images_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(num_classes=8).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # FIX: proper loss accumulation
            total_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

        epoch_loss = total_loss / total
        acc = correct / total

        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "../Model/colour_model.pth")

    print("Training complete!!")

################################

if __name__ == "__main__":
    #prepare_data()
    run()