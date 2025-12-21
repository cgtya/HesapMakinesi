import torch
import json
from PIL import Image
import torchvision.transforms as transforms
from Im2LatexModel import Im2LatexModel  # Corrected import

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
SAVE_DIR = os.path.join(BASE_DIR, "..", "..", "egitim_sonuclari", "mathwriting_exp")
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint_epoch6.pth")
STOI_PATH = os.path.join(SAVE_DIR, "stoi.json")
ITOS_PATH = os.path.join(SAVE_DIR, "itos.json")
IMG_PATH = os.path.join(BASE_DIR, "test.png")

def preprocess_for_inference(image_path):
    """
    gürültülü/gölgeli görüntüler için adaptif tresholding
    """
    if not OPENCV_AVAILABLE:
        # standart PIL open
        return Image.open(image_path).convert("RGB")
    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # adaptif tresholding: 
    # blockSize=21, C=10 (ayarlanabilir)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 21, 10)
    
    # griyi rgb'ye çevir
    rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    
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
        # 1. encode image
        encoded = model.encoder(image_tensor)
        B, C, H, W = encoded.shape
        encoded = encoded.view(B, C, H * W).permute(0, 2, 1)  # [B, seq_len, C]
        encoded = model.enc_proj(encoded) # [B, seq_len, hidden_dim]
        encoded = encoded.permute(1, 0, 2) # [seq_len, B, hidden_dim] (Transformer ordering)
        
        # 2. Decode
        # Start with <sos>
        tgt_indices = [stoi["<sos>"]]
        
        for i in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(DEVICE) # [1, seq]
            
            # Forward pass through decoder components manually or use model parts if exposed?
            # The model's forward method expects full tgt_seq. 
            # We can use the model's forward path but we need to be careful about causality.
            # Im2LatexModel.forward takes (images, tgt_seq).
            # But here we want to run step-by-step or just re-run full sequence each time.
            # Re-running full sequence is easier (Greedy).
            
            logits = model(image_tensor, tgt_tensor) # [B, seq_len, vocab_size]
            
            # Get the predicted token for the LAST position
            last_token_logits = logits[0, -1, :]
            predicted_token = last_token_logits.argmax(dim=0).item()
            
            if predicted_token == stoi["<eos>"]:
                break
            
            tgt_indices.append(predicted_token)
            
        # Convert indices to string
        tokens = [itos[idx] for idx in tgt_indices[1:]] # Skip <sos>
        return " ".join(tokens)

def main():
    print(f"Using device: {DEVICE}")
    stoi, itos = load_vocab()
    
    # Load Model
    model = Im2LatexModel(vocab_size=len(stoi)).to(DEVICE)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # Prepare Image
    # Prepare Image
    try:
        # Use new preprocessing function
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
    
    # Run Inference
    print(f"Vocab size: {len(stoi)}")
    print("Running inference...")
    
    # Image Stats Debug
    extrema = image.getextrema()
    print(f"Image Extrema (RGB): {extrema}")
    
    # Check first few steps detail
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
