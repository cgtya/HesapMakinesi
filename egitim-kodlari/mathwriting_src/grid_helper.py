import random
import math

def add_grid(ax, traces):
    """
    Grafige rastgele kareli kagit desenleri ekler.
    Modelin saglamligini artirmak icin kullanilir.
    """
    # Tum noktalari topla
    all_x = []
    all_y = []
    for t in traces:
        for p in t:
            all_x.append(p[0])
            all_y.append(-p[1]) # render_traces fonksiyonundaki gibi Y eksenini ters ceviriyoruz
    
    if not all_x: return

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # resmin merkezini bul
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # kare araligini belirle (resmin boyutuna gore dinamik)
    base_dim = max(width, height)
    if base_dim == 0: base_dim = 100 
    
    step = base_dim / random.uniform(5, 20) 
    
    # kare rengi
    colors = [
        '#a0a0a0', # gri 
        '#e0e0e0', # acik gri
        '#808080' # koyu gri
    ]
    
    grid_color = random.choice(colors)
    grid_linewidth = random.uniform(0.5, 1.5)
    
    # Rastgele bir aci belirle (-20 ile +20 derece arasi)
    angle_deg = random.uniform(-20, 20)
    angle_rad = math.radians(angle_deg)
    
    # Rotasyon icin genislesmis sinirlari hesapla
    # Izgara, dondugunde cerceveyi doldurmasi icin daha genis bir alanda cizilmeli
    diag = math.sqrt(width**2 + height**2)
    limit = diag * 1.5
    
    # Donusum fonksiyonu (nokta -> donmus nokta)
    def rotate_point(px, py, cx, cy, theta):
        return (
            cx + (px - cx) * math.cos(theta) - (py - cy) * math.sin(theta),
            cy + (px - cx) * math.sin(theta) + (py - cy) * math.cos(theta)
        )

    # kare cizgilerini olustur (merkezden disari dogru)
    # Dikey cizgiler
    x = 0
    while x < limit:
        # Saga dogru
        p1 = rotate_point(center_x + x, center_y - limit, center_x, center_y, angle_rad)
        p2 = rotate_point(center_x + x, center_y + limit, center_x, center_y, angle_rad)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=grid_color, linewidth=grid_linewidth, zorder=0)
        
        if x != 0:
            # Sola dogru
            p1 = rotate_point(center_x - x, center_y - limit, center_x, center_y, angle_rad)
            p2 = rotate_point(center_x - x, center_y + limit, center_x, center_y, angle_rad)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=grid_color, linewidth=grid_linewidth, zorder=0)
            
        x += step

    # Yatay cizgiler
    y = 0
    while y < limit:
        # Yukari dogru
        p1 = rotate_point(center_x - limit, center_y + y, center_x, center_y, angle_rad)
        p2 = rotate_point(center_x + limit, center_y + y, center_x, center_y, angle_rad)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=grid_color, linewidth=grid_linewidth, zorder=0)
        
        if y != 0:
            # Asagi dogru
            p1 = rotate_point(center_x - limit, center_y - y, center_x, center_y, angle_rad)
            p2 = rotate_point(center_x + limit, center_y - y, center_x, center_y, angle_rad)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=grid_color, linewidth=grid_linewidth, zorder=0)
            
        y += step

    padding_x = width * 0.1
    padding_y = height * 0.2
    
    ax.set_xlim(min_x - padding_x, max_x + padding_x)
    ax.set_ylim(min_y - padding_y, max_y + padding_y)
