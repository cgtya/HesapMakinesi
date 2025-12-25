import torch
import json
from PIL import Image
import torchvision.transforms as transforms
from Im2LatexModel import Im2LatexModel

import os

import base64
import io

# opsiyonel opencv
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# ayarlar
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ocr kaynak dizini
POSSIBLE_OCR_SRC = [
    os.path.join(BASE_DIR, "ocr_src"),
    os.path.join(BASE_DIR, "..", "ocr_src")
]

OCR_SRC_DIR = None
for path in POSSIBLE_OCR_SRC:
    if os.path.isdir(path):
        OCR_SRC_DIR = path
        break

if OCR_SRC_DIR is None:
    OCR_SRC_DIR = os.path.join(BASE_DIR, "..", "ocr_src")

SAVE_DIR = os.path.join(OCR_SRC_DIR, "mathwriting_model")

CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint_epoch15.pth")
STOI_PATH = os.path.join(SAVE_DIR, "stoi.json")
ITOS_PATH = os.path.join(SAVE_DIR, "itos.json")

IMG_PATH = os.path.join(BASE_DIR, "ign_testfoto/long.png")

# model ve sozluk onbellek
_cached_model = None
_cached_stoi = None
_cached_itos = None

def _apply_cv2_processing(img):
    # opencv islemleri
    # gri tona cevir
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # acik griyi beyaza it koyuyu koru
    cleaned = cv2.convertScaleAbs(gray, alpha=1.5, beta=20.0)
    
    rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)

def preprocess_image_file(image_path):
    # dosya yolundan okuma
    if not OPENCV_AVAILABLE:
        return Image.open(image_path).convert("RGB")
    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"dosya bulunamadi: {image_path}")
        
    return _apply_cv2_processing(img)

def preprocess_base64_image(base64_str):
    # base64 verisinden okuma
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
        
    image_data = base64.b64decode(base64_str)
    
    if OPENCV_AVAILABLE:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return _apply_cv2_processing(img)
            
    # opencv yoksa veya hata varsa pil
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def load_vocab():
    global _cached_stoi, _cached_itos
    if _cached_stoi is not None and _cached_itos is not None:
        return _cached_stoi, _cached_itos

    with open(STOI_PATH, "r", encoding="utf-8") as f:
        stoi = json.load(f)
    with open(ITOS_PATH, "r", encoding="utf-8") as f:
        itos = json.load(f)
        itos = {int(k): v for k, v in itos.items()}
    
    _cached_stoi = stoi
    _cached_itos = itos
    return stoi, itos

def load_model(vocab_size):
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    
    # modeli olustur ve agirliklari yukle
    model = Im2LatexModel(vocab_size=vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    
    _cached_model = model
    return model

def remove_trailing_dot(latex_str):
    """
    modelin ciktisinin sonundaki gereksiz noktalari temizler.
    """
    if not latex_str:
        return latex_str
    
    latex_str = latex_str.strip()
    
    if latex_str.endswith("." or "\cdot"):
        latex_str = latex_str[:-1].strip()
        
    return latex_str

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
        latex_output = " ".join(tokens)
        return remove_trailing_dot(latex_output)

def predict_from_base64(base64_str):
    # disaridan cagrilacak ana fonksiyon
    try:
        # goruntu hazirla
        image = preprocess_base64_image(base64_str)
        
        # model ve sozluk yukle
        stoi, itos = load_vocab()
        model = load_model(len(stoi))
        
        # tensor donusumu
        transform = transforms.Compose([
            transforms.Resize((384, 384)),   
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # tahmin
        latex = greedy_decode(model, img_tensor, stoi=stoi, itos=itos)
        return latex
    except Exception as e:
        return f"hata: {str(e)}"

def main():
    print(f"cihaz: {DEVICE}")
    
    # model ve sozluk yuklemesi testi
    try:
        stoi, itos = load_vocab()
        model = load_model(len(stoi))
        print("model ve sozluk yuklendi")
    except Exception as e:
        print(f"yukleme hatasi: {e}")
        return

    # dosya kontrolu
    try:
        image = preprocess_image_file(IMG_PATH)
    except FileNotFoundError:
        print(f"dosya bulunamadi: {IMG_PATH}")
        return

    # transform islemleri
    transform = transforms.Compose([
        transforms.Resize((384, 384)),   
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    print("test tahmini yapiliyor...")
    latex_output = greedy_decode(model, img_tensor, stoi=stoi, itos=itos)
    
    print("-" * 30)
    print("tahmin edilen latex (dosyadan):")
    print(latex_output)
    print("-" * 30)
    
    # base64 entegrasyon testi
    if os.path.exists(IMG_PATH):
        print("\nbase64 testi baslatiliyor...")
        with open(IMG_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
            
        result = predict_from_base64(b64)
        print("tahmin edilen latex (base64):")
        print(result)

if __name__ == "__main__":
    main()
