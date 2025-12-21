import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Im2LatexCSV(Dataset):
    def __init__(self, csv_path_or_df, image_folder, stoi, max_len=256):
        self.image_folder = image_folder
        self.stoi = stoi
        self.max_len = max_len

        if isinstance(csv_path_or_df, pd.DataFrame):
            df = csv_path_or_df
        else:
            df = pd.read_csv(csv_path_or_df)
        self.items = []
        for _, row in df.iterrows():
            latex = row["formula"].strip()
            img_name = row["image"].strip()
            img_path = os.path.join(image_folder, img_name)
            if os.path.exists(img_path):
                self.items.append((img_path, latex))

        # data augmentation
        self.tf = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomRotation(degrees=5),      # hafif dondurme
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)), # hafif kaydirma/olcekleme
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def encode(self, s):
        toks = s.split() # Veri seti zaten bosluklarla ayrilmis
        ids = [self.stoi["<sos>"]] + [self.stoi.get(t, self.stoi["<unk>"]) for t in toks] + [self.stoi["<eos>"]]
        ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, i):
        img_path, latex = self.items[i]
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)
        tgt = self.encode(latex)
        return img, tgt

    def __len__(self):
        return len(self.items)

if __name__ == "__main__":
    stoi = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
    # ds = Im2LatexCSV(CSV_PATH, IMAGE_FOLDER, stoi)
    # print("Dataset uzunluğu:", len(ds))
    pass
