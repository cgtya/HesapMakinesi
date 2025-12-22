import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Im2LatexCSV(Dataset):
    def __init__(self, csv_path_or_df, image_folder, stoi, max_len=256, augment=True):
        self.image_folder = image_folder
        self.stoi = stoi
        self.max_len = max_len
        self.augment = augment

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


        # Base transforms (always applied)
        ops = [transforms.Resize((384, 384))]
        
        if self.augment:
            # Heavy augmentations
            ops.extend([
                transforms.RandomRotation(degrees=5),   # döndürme
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)), # kaydırma, ölçeklendirme
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # parlaklık, kontrast, ton
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), # bulanıklık
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3), # perspektif
            ])
            
        ops.append(transforms.ToTensor())
        
        if self.augment:
             ops.append(AddGaussianNoise(0., 0.05))
             
        ops.append(transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)) # normalize
        
        self.tf = transforms.Compose(ops)

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
