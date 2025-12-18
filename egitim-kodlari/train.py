import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import re
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import json
import matplotlib.pyplot as plt

from dataset import Im2LatexCSV
from Im2LatexModel import Im2LatexModel
from collate_fn import collate_fn

# -----------------------------# 1. Ayarlar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dosya konumunu dinamik olarak alınır
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "training_data", "im2latex_train.csv")
VAL_PATH = os.path.join(BASE_DIR, "..", "training_data", "im2latex_validate.csv")
IMAGE_FOLDER = os.path.join(BASE_DIR, "..", "training_data", "formula_images_processed", "formula_images_processed")
LOSS_HISTORY_PATH = "loss_history.json"


BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_CORES = 4 # 0 - tek çekirdek, diğer hallerde çekirdek sayısı

# --- Vocab ---
df = pd.read_csv(CSV_PATH)
formulas = df['formula'].tolist()  # sütun adını kendi dosyana göre ayarla
vocab = set()
for formula in formulas:
    tokens = formula.split()
    for tok in tokens:
        vocab.add(tok)

stoi = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
for i, tok in enumerate(sorted(vocab), start=4):
    stoi[tok] = i
itos = {v:k for k,v in stoi.items()}

# JSON'a kaydet (sonradan inference için)
with open("stoi.json", "w", encoding="utf-8") as f:
    json.dump(stoi, f, ensure_ascii=False, indent=2)
with open("itos.json", "w", encoding="utf-8") as f:
    json.dump(itos, f, ensure_ascii=False, indent=2)

print("Vocab size:", len(stoi))  # 544

train_losses = []
val_losses = []

if os.path.exists(os.path.join(BASE_DIR,"..",LOSS_HISTORY_PATH)):
    try:
        with open(os.path.join(BASE_DIR,"..",LOSS_HISTORY_PATH), "r") as f:
            history = json.load(f)
            train_losses = history.get("train_losses", [])
            val_losses = history.get("val_losses", [])
            print(f"loss geçmişi yüklendi. {len(train_losses)} epoch veri bulundu.")
    except Exception as e:
        print(f"loss geçmişi yüklenemedi: {e}")


# -----------------------------# 2. Veri Yükleyiciler
train_ds = Im2LatexCSV(CSV_PATH, IMAGE_FOLDER, stoi)
val_ds = Im2LatexCSV(VAL_PATH, IMAGE_FOLDER, stoi)

#!!!! num_workers işlemci çekirdek sayısına göre değiştirilebilir
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, pad_idx=stoi["<pad>"]),
                          num_workers=CPU_CORES, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, pad_idx=stoi["<pad>"]),
                        num_workers=CPU_CORES, pin_memory=True)

# -----------------------------# 3. Model, Kayıp Fonksiyonu ve Optimizatör
model = Im2LatexModel(vocab_size=len(stoi)).to(DEVICE)

weights = torch.ones(len(stoi)).to(DEVICE)
weights[stoi["<unk>"]] = 10.0  # <unk> hatalarına 10 kat ceza

criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=stoi["<pad>"], label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# amp scaler (mixed precision) destekleyen donanımda daha hızlı eğitilir
scaler = torch.amp.GradScaler('cuda')

# epoch checkpointleri kontrolü
start_epoch = 1
checkpoints = glob.glob("checkpoint_epoch*.pth")
if checkpoints:
    # dosya isimlerinden epoch numaralarını ayıkla
    epochs_found = []
    for cp in checkpoints:
        match = re.search(r"checkpoint_epoch(\d+).pth", cp)
        if match:
            epochs_found.append(int(match.group(1)))
    
    if epochs_found:
        max_epoch = max(epochs_found)
        latest_cp = f"checkpoint_epoch{max_epoch}.pth"
        print(f"en son checkpoint: {latest_cp}. yukleniyor")
        
        try:
            model.load_state_dict(torch.load(latest_cp, map_location=device))
            print("model yüklendi.")
            
            # Optimizer checkpoint'i varsa yükle
            opt_cp = f"optimizer_epoch{max_epoch}.pth"
            if os.path.exists(opt_cp):
                print(f"optimizer yüklendi: {opt_cp}")
                optimizer.load_state_dict(torch.load(opt_cp, map_location=device))
            
            # Scaler checkpoint'i varsa yükle (AMP için)
            scaler_cp = f"scaler_epoch{max_epoch}.pth"
            if os.path.exists(scaler_cp):
                print(f"scaler yüklendi: {scaler_cp}")
                scaler.load_state_dict(torch.load(scaler_cp, map_location=device))

            start_epoch = max_epoch + 1
            print(f" epoch {start_epoch} devam ediyor")
        except Exception as e:
            print(f" checkpoint yüklenemedi: {e}. baştan başlanıyor.")
            start_epoch = 1
    else:
        print("checkpoint yüklenemedi. baştan başlanıyor.")
else:
    print("checkpoint yok. baştan başlanıyor.")



# -----------------------------# 4. Eğitim Döngüsü

for epoch in range(start_epoch, EPOCHS+1):
    model.train()
    total_loss = 0

    for imgs, tgts in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch"):
        imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)

        optimizer.zero_grad()
        
        # Mixed Precision
        with torch.amp.autocast('cuda'):
             # Generate mask for the current batch sequence length
             tgt_seq = tgts[:, :-1]
             seq_len = tgt_seq.shape[1]
             mask = model.generate_square_subsequent_mask(seq_len).to(DEVICE)
             
             logits = model(imgs, tgt_seq, tgt_mask=mask)
             loss = criterion(logits.reshape(-1, logits.size(-1)), tgts[:, 1:].reshape(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        

    torch.save(model.state_dict(), f"checkpoint_epoch{epoch}.pth")
    torch.save(optimizer.state_dict(), f"optimizer_epoch{epoch}.pth")
    torch.save(scaler.state_dict(), f"scaler_epoch{epoch}.pth")
    print(f"Checkpoint saved for epoch {epoch}")

    scheduler.step()

    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, tgts in val_loader:
            imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
            
            tgt_seq = tgts[:, :-1]
            seq_len = tgt_seq.shape[1]
            mask = model.generate_square_subsequent_mask(seq_len).to(DEVICE)
            
            logits = model(imgs, tgt_seq, tgt_mask=mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgts[:, 1:].reshape(-1))
            val_loss += loss.item()

    val_loss /= len(val_loader)
    tqdm.write(f"Epoch {epoch}/{EPOCHS}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
    train_losses.append(avg_loss)
    val_losses.append(val_loss)

    # loss geçmişini kaydet
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    with open("loss_history.json", "w") as f:
        json.dump(history, f)


# -----------------------------# 5. Kayıp Grafiği
plt.figure(figsize=(8,5))
epochs_range = range(start_epoch, start_epoch + len(train_losses))
plt.plot(epochs_range, train_losses, label="Train Loss", marker="o")
plt.plot(epochs_range, val_losses, label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_graph.png")
