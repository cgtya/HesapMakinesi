import os
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import re
from tqdm import tqdm

from grid_helper import add_grid


# synthetic veri icin (+400k)
SYNTHETIC = False

# islemci sayisi (islemci cekirdegi sayisi kadar)
num_workers = os.cpu_count() or 4

# rastgele seed
random.seed(42)

# kareli kagit ozelligini acip kapatmak icin degisken
ENABLE_GRID_AUGMENTATION = True
GRID_PROBABILITY = 1    # kareli kagit olma olasiligi (0.0 - 1.0)

# kalın kalem   (normal aralık 1-4px / kalın aralık 1-6px)
THICK = False


def tokenize_formula(formula):
    """
    latex formullerini tokenize eder
    ornek:
    'x^2' -> 'x ^ 2'
    '\\sin(x)' -> '\\sin ( x )'
    """
    if not isinstance(formula, str):
        return ""
        
    # pattern aciklamalari:
    # \\[a-zA-Z]+ : latex komutlari (orn. \sin, \alpha)
    # \\.         : kacis karakterleri (orn. \{, \})
    # [a-zA-Z]    : tek harfler
    # \d          : sayilar
    # .           : herhangi tek karakter
    
    # siralamanin onemi var
    # 1. komutlar
    # 2. kacis karakterleri
    # 3. diger karakterler
    
    regex = r"(\\[a-zA-Z]+)|(\\[^a-zA-Z])|(\S)"
    
    matches = re.finditer(regex, formula)
    out_tokens = []
    for m in matches:
        token = m.group(0)
        if token:
           out_tokens.append(token)
           
    return " ".join(out_tokens)

def parse_inkml(file_path):
    #inkml dosyasindan iz ve etiketleri cikarir
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # isim alanlari
    ns = {'ink': 'http://www.w3.org/2003/InkML'}
    
    # etiketi al
    annotation = root.find(".//ink:annotation[@type='normalizedLabel']", ns)
    if annotation is None:
        # normalizedlabel yoksa digerini dene
        annotation = root.find(".//ink:annotation[@type='label']", ns)
    
    label = annotation.text if annotation is not None else None
    
    # etiket varsa tokenize et
    if label:
        label = tokenize_formula(label)
    
    # izleri al
    traces = []
    for trace_tag in root.findall(".//ink:trace", ns):
        trace_text = trace_tag.text.strip()
        points = trace_text.split(',')
        trace_data = []
        for point in points:
            coords = point.strip().split()
            # x ve y genelde ilk ikisidir
            try:
                x = float(coords[0])
                y = float(coords[1])
                trace_data.append((x, y))
            except (ValueError, IndexError):
                continue
        if trace_data:
            traces.append(trace_data)
            
    return traces, label

def render_traces(traces, output_path, dpi=100):
    #izleri rastgele kalem kalinligiyla gorsellestirir
    fig, ax = plt.subplots(figsize=(4, 4)) # boyutu standarize et
    ax.set_aspect('equal')
    ax.axis('off')

    # farkli kalemleri simule etmek icin rastgele kalinlik
    linewidth = random.uniform(1.0, 4.0)
    if THICK:
        linewidth = random.uniform(1.0, 6.0)

    # Grid (Kareli kagit) eklentisi
    if ENABLE_GRID_AUGMENTATION and random.random() < GRID_PROBABILITY:
        try:
            add_grid(ax, traces)
        except Exception as e:
            print(f"Grid eklenirken hata: {e}")
    
    for trace in traces:
        if not trace:
            continue
        data = np.array(trace)
        # matplotlib icin y eksenini ters cevir
        ax.plot(data[:, 0], -data[:, 1], color='black', linewidth=linewidth, zorder=10)
        
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)


from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_file(file_path, output_img_dir):
    """
    tek bir inkml dosyasini isler
    multiprocessing icin bagimsiz fonksiyon
    """
    try:
        traces, label = parse_inkml(file_path)
        
        if not traces or label is None:
            return None
        
        # gorsel dosya adi
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        img_filename = f"{file_id}.png"
        output_img_path = os.path.join(output_img_dir, img_filename)
        
        # render et
        render_traces(traces, output_img_path)
        
        return {
            "image": img_filename,
            "formula": label
        }
        
    except Exception as e:
        # Hata durumunda basmayalim ki log kirlenmesin, return None
        return None

def process_dataset(input_dir, output_img_dir, output_csv_path, subset="test"):
    #klasordeki tum inkml dosyalarini isler
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
        
    # inkml dosyalarini ara
    search_pattern = os.path.join(input_dir, subset, "*.inkml")
    files = glob.glob(search_pattern)
    
    print(f"Toplam dosya sayisi: {len(files)} ({subset})")
    
    data = []
    

    print(f"kullanilan islemci sayisi: {num_workers}")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # cekirdeklere gorevleri dagit
        futures = {executor.submit(process_single_file, f, output_img_dir): f for f in files}
        
        for future in tqdm(as_completed(futures), total=len(files), desc=f"Isleniyor ({subset})"):
            result = future.result()
            if result:
                data.append(result)

    # csv olustur
    if data:
        df = pd.DataFrame(data)
        # sadece islenenleri kaydet
        df.to_csv(output_csv_path, index=False)
        print(f"CSV kaydedildi: {output_csv_path} ({len(data)} satir)")
    else:
        print("Kaydedilecek veri yok.")

if __name__ == "__main__":
    # script konumuna gore yollari ayarla
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # ana proje dizinine cik
    DATA_ROOT = os.path.join(BASE_DIR, "..", "..", "training_data", "mathwriting-2024")
    
    # islenecek klasorler
    SUBSETS = ["test", "train", "valid"] 
    if SYNTHETIC:
        SUBSETS.append("synthetic") # synthetic verisi icin bunu acin
    
    OUTPUT_IMG_DIR = os.path.join(DATA_ROOT, "processed_images"+("_grid" if ENABLE_GRID_AUGMENTATION else "")+("_thick" if THICK else ""))
    
    for subset in SUBSETS:
        print(f"\nIslem basliyor: {subset}")
        csv_path = os.path.join(DATA_ROOT, f"mathwriting_{subset}"+("_grid" if ENABLE_GRID_AUGMENTATION else "")+("_thick" if THICK else "") + ".csv")
        process_dataset(DATA_ROOT, OUTPUT_IMG_DIR, csv_path, subset)    
