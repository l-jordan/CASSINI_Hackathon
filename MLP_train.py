import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# MLP with four layers, ReLU activation and 20% drop out
class MLP_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP_Classifier, self).__init__()
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 2
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 3
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),

            # Layer 4 (output)
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)

df = pd.read_parquet("data\\training_data_balanced.parquet")
class_names = sorted(df["class"].unique().tolist())

# X - All features y - Class labels transformed into ids
X = df.iloc[:, 2:]
y = df.iloc[:, 0].map({name: i for i, name in enumerate(class_names)})  

# TRAIN SET (80% OF THE DATA)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# TEST & VALIDATE (10% EACH)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
X_val   = scaler.transform(X_val)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train.values)
X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.LongTensor(y_test.values)
X_val_t   = torch.FloatTensor(X_val)
y_val_t   = torch.LongTensor(y_val.values)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=16, shuffle=True
)

##############################################
# TRAINING THE MODEL
##############################################
input_size  = X_train.shape[1]
hidden_size = 256
num_classes  = len(np.unique(y))
epochs       = 20
lr           = 0.001

model     = MLP_Classifier(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_losses = []
val_losses   = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    # -- Validation loss (no gradients needed) --
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t), y_val_t).item()
    val_losses.append(val_loss)

    if (epoch + 1) % 2 == 0: #Print every two epochs
        print(f"Epoch [{epoch+1:>2}/{epochs}]  "
              f"Train Loss: {train_losses[-1]:.4f}  "
              f"Val Loss: {val_losses[-1]:.4f}")

##############################################
# SAVE MODEL & SCALER
##############################################
torch.save(model.state_dict(), "model.pth") # saves learned numbers from training
joblib.dump(scaler, "scaler.pkl") # mean and std of X_train
print("\nSaved: model.pth, scaler.pkl")

#################################################
# ANALYSIS
#################################################
model.eval()
with torch.no_grad():
    preds_test = torch.argmax(model(X_test_t), dim=1).numpy() 
    preds_val  = torch.argmax(model(X_val_t),  dim=1).numpy() 

# Accuracy
print(f"\nVal  Accuracy: {(preds_val  == y_val ).mean()*100:.2f}%")
print(f"Test Accuracy: {(preds_test == y_test).mean()*100:.2f}%")

# Confusion matrix — on TEST set
cm = confusion_matrix(y_test, preds_test, normalize="all") 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", colorbar=False, xticks_rotation=45)
plt.title(r"Confusion Matrix (Test Set) — % of all pixels")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# Loss curve
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.show()