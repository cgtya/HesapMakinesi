import torch
import torch.nn as nn
import os
import glob
import re
import json
from tqdm import tqdm
from torch.utils.data import DataLoader


from dataset import Im2LatexCSV
from Im2LatexModel import Im2LatexModel
from collate_fn import collate_fn

def recalc_val_loss():
    # dosya yolları

    AUGMENT = "_grid_thick"     # "_grid" ve/veya "_thick" olabilir. öncesinde validation için preprocessed veri seti oluşturulmalı

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VAL_PATH = os.path.join(BASE_DIR, "..", "..", "training_data", "mathwriting-2024", "mathwriting_valid"+AUGMENT+".csv")
    IMAGE_FOLDER = os.path.join(BASE_DIR, "..", "..", "training_data", "mathwriting-2024", "processed_images"+AUGMENT)
    SAVE_DIR = os.path.join(BASE_DIR, "..", "..", "egitim_sonuclari", "mathwriting_exp")
    STOI_PATH = os.path.join(SAVE_DIR, "stoi.json")
    OUTPUT_JSON_PATH = os.path.join(SAVE_DIR, "loss_history_nosmooth"+AUGMENT+".json")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    CPU_CORES = 4

    print(f"cihaz: {DEVICE}")
    print(f"checkpoint konumu: {SAVE_DIR}")

    # vocab yükle
    if not os.path.exists(STOI_PATH):
        print(f"stoi.json bulunamadı: {STOI_PATH}")
        return

    with open(STOI_PATH, "r", encoding="utf-8") as f:
        stoi = json.load(f)
    print(f"vocab boyutu: {len(stoi)}")

    # doğrulama veri setini yükle
    if not os.path.exists(VAL_PATH):
        print(f"validation csv bulunamadı: {VAL_PATH}")
        return
    
    val_ds = Im2LatexCSV(VAL_PATH, IMAGE_FOLDER, stoi, augment=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, pad_idx=stoi["<pad>"]),
                            num_workers=CPU_CORES, pin_memory=True)
    print(f"valid veri seti boyutu: {len(val_ds)}")

    # modeli ve kayıp fonksiyonunu başlat (label smoothing kapalı)
    model = Im2LatexModel(vocab_size=len(stoi)).to(DEVICE)
    
    # eğitimdeki ağırlıkları kullanıyoruz ama label smoothing kapalı
    weights = torch.ones(len(stoi)).to(DEVICE)
    weights[stoi["<unk>"]] = 10.0
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=stoi["<pad>"], label_smoothing=0.0)

    # checkpointleri bul
    checkpoints = glob.glob(os.path.join(SAVE_DIR, "checkpoint_epoch*.pth"))
    epoch_cp_map = {}
    for cp in checkpoints:
        match = re.search(r"checkpoint_epoch(\d+).pth", cp)
        if match:
            epoch = int(match.group(1))
            epoch_cp_map[epoch] = cp
    
    sorted_epochs = sorted(epoch_cp_map.keys())
    print(f"bulunan epochlar: {sorted_epochs}")

    results = {}

    # döngü ve hesaplama
    for epoch in sorted_epochs:
        cp_path = epoch_cp_map[epoch]
        
        try:
            # ağırlıkları yükle
            state_dict = torch.load(cp_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.eval()

            val_loss = 0
            num_batches = 0
            
            # progress bar
            loop = tqdm(val_loader, desc=f"epoch {epoch}", unit="batch")
            
            with torch.no_grad():
                for imgs, tgts in loop:
                    imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
                    
                    with torch.amp.autocast('cuda'):
                        tgt_seq = tgts[:, :-1]
                        seq_len = tgt_seq.shape[1]
                        mask = model.generate_square_subsequent_mask(seq_len).to(DEVICE)
                        
                        logits = model(imgs, tgt_seq, tgt_mask=mask)
                        # kayıp hesapla
                        loss = criterion(logits.reshape(-1, logits.size(-1)), tgts[:, 1:].reshape(-1))
                    val_loss += loss.item()
                    num_batches += 1
                    
                    loop.set_postfix(anlik_loss=loss.item())

            avg_val_loss = val_loss / num_batches if num_batches > 0 else 0
            results[epoch] = avg_val_loss
            tqdm.write(f"epoch {epoch} tamamlandı: val loss = {avg_val_loss:.4f}\n")

        except Exception as e:
            print(f"epoch {epoch} işlenirken hata oluştu: {e}")

    # sonuçları kaydet
    json_results = {"val_losses_nosmooth"+AUGMENT: [{"epoch": k, "loss": v} for k, v in results.items()]}
    
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"sonuç kaydedildi: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    recalc_val_loss()
