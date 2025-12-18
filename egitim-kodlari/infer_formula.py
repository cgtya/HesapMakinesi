import argparse
import json
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from Im2LatexModel import Im2LatexModel

def preprocess_opencv_to_pil(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 50, 50)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 10)
    if np.mean(th) < 127:
        th = 255 - th
    rgb = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)

def greedy_decode(model, image_tensor, stoi, itos, max_len=256, device="cpu"):
    sos = stoi["<sos>"]
    eos = stoi["<eos>"]
    seq = torch.tensor([[sos]], dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        logits = model(image_tensor, seq)
        next_id = int(torch.argmax(logits[0, -1], dim=-1).item())
        seq = torch.cat([seq, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == eos:
            break
    ids = seq[0].tolist()
    toks = []
    for t in ids:
        if t in (sos, eos, stoi["<pad>"]):
            continue
        toks.append(itos.get(str(t), itos.get(t, "")))
    return " ".join([x for x in toks if x])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stoi", default="stoi.json")
    ap.add_argument("--itos", default="itos.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-opencv", action="store_true")
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    with open(args.stoi, "r", encoding="utf-8") as f:
        stoi = json.load(f)
    with open(args.itos, "r", encoding="utf-8") as f:
        itos = json.load(f)

    device = args.device

    tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    pil = Image.open(args.image).convert("RGB") if args.no_opencv else preprocess_opencv_to_pil(args.image)
    x = tf(pil).unsqueeze(0).to(device)

    model = Im2LatexModel(vocab_size=len(stoi), max_len=args.max_len).to(device)
    sd = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        latex_out = greedy_decode(model, x, stoi, itos, max_len=args.max_len, device=device)

    print(latex_out)

if __name__ == "__main__":
    main()
