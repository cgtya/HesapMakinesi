import torch
import json
from PIL import Image
import torchvision.transforms as transforms
from Im2LatexModel import Im2LatexModel

import os

# opsiyonel opencv
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("opencv bulunamadı, gelişmiş işleme devre dışı")



# ayarlar
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "..", "mathwriting_model")
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint_epoch15.pth")
STOI_PATH = os.path.join(SAVE_DIR, "stoi.json")
ITOS_PATH = os.path.join(SAVE_DIR, "itos.json")

IMG_PATH = os.path.join(BASE_DIR, "ign_testfoto/long.png")

def preprocess_for_inference(image_path):
    # opencv kullanilamazsa standart PIL acilisi
    if not OPENCV_AVAILABLE:
        return Image.open(image_path).convert("RGB")
    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # bu islem acik gri (kagit dokusu gibi) pikselleri beyaza iter, koyu kisimlari korur
    cleaned = cv2.convertScaleAbs(gray, alpha=1.5, beta=20.0)
    
    rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(rgb)

def load_vocab():
    with open(STOI_PATH, "r", encoding="utf-8") as f:
        stoi = json.load(f)
    with open(ITOS_PATH, "r", encoding="utf-8") as f:
        itos = json.load(f)
        itos = {int(k): v for k, v in itos.items()}
    return stoi, itos

def greedy_decode(model, image_tensor, max_len=150, stoi=None, itos=None):
    model.eval()
    with torch.no_grad():
        # goruntuyu kodla
        encoded = model.encoder(image_tensor)
        B, C, H, W = encoded.shape
        encoded = encoded.view(B, C, H * W).permute(0, 2, 1)  # [B, seq_len, C]
        encoded = model.enc_proj(encoded) # [B, seq_len, hidden_dim]
        encoded = encoded.permute(1, 0, 2) # [seq_len, B, hidden_dim] (Transformer siralamasi)
        
        # cozumle
        # <sos> ile basla
        tgt_indices = [stoi["<sos>"]]
        
        for i in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(DEVICE) # [1, seq]
            
            # model tam hedef dizisini bekler
            # adim adim gitmek yerine her seferinde tum diziyi tekrar calistiriyoruz
            
            logits = model(image_tensor, tgt_tensor) # [B, seq_len, vocab_size]
            
            # son pozisyon icin tahmin edilen tokeni al
            last_token_logits = logits[0, -1, :]
            predicted_token = last_token_logits.argmax(dim=0).item()
            
            if predicted_token == stoi["<eos>"]:
                break
            
            tgt_indices.append(predicted_token)
            
        # indeksleri stringe cevir
        tokens = [itos[idx] for idx in tgt_indices[1:]] # <sos> atla
        return " ".join(tokens)

def main():
    print(f"Using device: {DEVICE}")
    stoi, itos = load_vocab()
    
    # modeli yukle
    model = Im2LatexModel(vocab_size=len(stoi)).to(DEVICE)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # fotografi hazirla
    try:
        image = preprocess_for_inference(IMG_PATH)
    except FileNotFoundError:
        print(f"Image not found at {IMG_PATH}")
        return

    transform = transforms.Compose([
        transforms.Resize((384, 384)),   
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(DEVICE) # [1, 3, 384, 384]
    
    # tahmin et
    print(f"Vocab size: {len(stoi)}")
    print("Running inference...")
    
    
    # goruntu istatistikleri (debug)
    extrema = image.getextrema()
    print(f"Image Extrema (RGB): {extrema}")
    
    # ilk birkac adimi kontrol et
    model.eval()
    with torch.no_grad():
         tgt_indices = [stoi["<sos>"]]
         tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(DEVICE)
         
         logits = model(img_tensor, tgt_tensor)
         last_logits = logits[0, -1, :]
         probs = torch.softmax(last_logits, dim=0)
         top_probs, top_ids = torch.topk(probs, 5)
         
         print("Top 5 predictions at start:")
         for p, idx in zip(top_probs, top_ids):
             token_str = itos.get(idx.item(), "Unknown")
             print(f"  Token: '{token_str}' (ID: {idx.item()}), Prob: {p.item():.4f}")


    latex_output = greedy_decode(model, img_tensor, stoi=stoi, itos=itos)
    
    print("-" * 30)
    print("Predicted LaTeX:")
    print(latex_output)
    print("-" * 30)

if __name__ == "__main__":
    main()
