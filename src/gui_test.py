import flet as ft
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import threading
import sympy as sp
import Solver_mod as solver
from numpy.matlib import empty
from Solver_mod import MathStep
from sympy import sin, cos, tan, pi
from latex2sympy2_extended import latex2sympy
import re

# Parse işlemleri (rationalize önemli)
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, \
    rationalize


# Matplotlib arka plan ayarı
matplotlib.use('Agg')

# Şeffaf 1x1 piksellik resim
BOS_RESIM = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


def main(page: ft.Page) -> None:
    page.title = "Hesap Makinesi"
    page.window.maximized = True
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = ft.Colors.GREY_900
    page.scroll = ft.ScrollMode.AUTO
    page.horizontal_alignment = ft.MainAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.padding = 20

    alt_bar = ft.NavigationBar(
        destinations=[
            ft.NavigationBarDestination(icon=ft.Icons.DASHBOARD, label="Günlük"),
            ft.NavigationBarDestination(icon=ft.Icons.SCIENCE_OUTLINED, label="Bilimsel"),
            ft.NavigationBarDestination(icon=ft.Icons.ADD_PHOTO_ALTERNATE_OUTLINED, label="Görsel"),
            ft.NavigationBarDestination(icon=ft.Icons.BOOKMARK_BORDER, selected_icon=ft.Icons.BOOKMARK, label="Geçmiş"),
        ],
        on_change=lambda e: ekran_degistir(e)
    )
    page.navigation_bar=alt_bar


    soncevap = "0"
    islemsonrasi = False
    debounce_timer = None

    # Hafızadaki son geçerli resim
    son_gecerli_resim = BOS_RESIM

    # --- 1. Resim Bölümü ---
    formul_resmi = ft.Image(
        src_base64=BOS_RESIM,
        width=400,
        height=120,
        fit=ft.ImageFit.CONTAIN,
        gapless_playback=True,
        visible=True
    )

    formul_container = ft.Container(
        content=formul_resmi,
        padding=5,
        bgcolor=ft.Colors.BLACK38,
        border_radius=10,
        alignment=ft.alignment.center,
        visible=True,
        height=130
    )

    # --- 2. Sonuç Ekranı ---
    txt_ondalik = ft.Text(value="", size=16, color=ft.Colors.GREY_400)  # Sol taraf (Silik)
    txt_sonuc = ft.Text(value="", size=20, weight="bold")  # Sağ taraf (Parlak)

    sonuc_kutusu = ft.Container(
        content=ft.Row(
            controls=[txt_ondalik, txt_sonuc],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER
        ),
        width=380,
        height=50,
        border=ft.border.all(1, "grey"),
        border_radius=5,
        padding=ft.padding.only(left=10, right=10),
        bgcolor=ft.Colors.GREY_900
    )

    # --- Fonksiyonlar ---
    def latex_to_image(latex_str):
        try:
            fig = plt.figure(figsize=(6, 1.5), dpi=120)
            fig.patch.set_alpha(0.0)
            plt.text(0.5, 0.5, f"${latex_str}$", fontsize=18, ha='center', va='center', color='white')
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        except:
            return None

    def latexe_cevir(text_val):
        import inspect

        try:
            clean_text = text_val.replace("diff", "Derivative").replace("integrate", "Integral")
            clean_text = clean_text.replace("^", "**").replace("×", "*")

            gorsel_ayarlar = {
                'x': sp.Symbol('x'),
                'Derivative': sp.Derivative,
                'Integral': sp.Integral,
                'integrate': sp.Integral,
                'sin': sp.Function(r'\sin'),
                'cos': sp.Function(r'\cos'),
                'tan': sp.Function(r'\tan'),
                'sqrt': sp.Function(r'\sqrt'),
                'log': sp.log,
                'ln': sp.log
            }

            donusumler = (standard_transformations + (implicit_multiplication_application,))
            expr = parse_expr(clean_text, local_dict=gorsel_ayarlar, transformations=donusumler, evaluate=False)

            if inspect.isclass(expr) or isinstance(expr, sp.core.function.UndefinedFunction):
                latex_code = expr.__name__
            else:
                latex_code = sp.latex(expr)

            latex_code = latex_code.replace(r"\limits", "")
        except Exception:
            latex_code=""
            pass
        return latex_code

    def gercek_guncelleme(text_val):
        nonlocal son_gecerli_resim
        if not text_val:
            formul_resmi.src_base64 = BOS_RESIM
            formul_resmi.update()
            return

        try:
            clean_text = text_val.replace("diff", "Derivative").replace("integrate", "Integral")
            clean_text = clean_text.replace("^", "**").replace("×", "*")

            gorsel_ayarlar = {
                'x': sp.Symbol('x'),
                'Derivative': sp.Derivative,
                'Integral': sp.Integral,
                'integrate': sp.Integral,
                'sin': sp.Function(r'\sin'),
                'cos': sp.Function(r'\cos'),
                'tan': sp.Function(r'\tan'),
                'sqrt': sp.Function(r'\sqrt'),
                'log': sp.log,
                'ln': sp.log
            }

            donusumler = (standard_transformations + (implicit_multiplication_application,))
            expr = parse_expr(clean_text, local_dict=gorsel_ayarlar, transformations=donusumler, evaluate=False)

            import inspect
            if inspect.isclass(expr) or isinstance(expr, sp.core.function.UndefinedFunction):
                latex_code = expr.__name__
            else:
                latex_code = sp.latex(expr)

            latex_code = latex_code.replace(r"\limits", "")
            base64_data = latex_to_image(latex_code)

            if base64_data:
                son_gecerli_resim = base64_data
                formul_resmi.src_base64 = base64_data
                formul_resmi.update()

        except Exception:
            pass

    def on_change_handler(e):
        nonlocal debounce_timer
        if debounce_timer:
            debounce_timer.cancel()
        debounce_timer = threading.Timer(0.4, gercek_guncelleme, args=[e.control.value])
        debounce_timer.start()

    girdi_ekranı = ft.TextField(
        value="",
        text_align="left",
        width=380,
        read_only=False,
        border_color="grey",
        on_change=on_change_handler,
        text_style=ft.TextStyle(size=18)
    )

    # --- Dosya İşlemleri ---
    def islem_kaydet(islem, sonuc, cozum):
        try:
            with open("islem_log.txt", "a", encoding="utf-8") as dosya:
                dosya.write(f"İşlem: {islem}\nSonuç: {sonuc}\nÇözüm: {cozum}\n\n")
        except:
            pass

    def islem_oku():
        desen = r"İşlem:\s*(.*?)\nSonuç:\s*(.*?)\nÇözüm:\s*(.*?)(?=\n\n)"
        try:
            with open("islem_log.txt", "r", encoding="utf-8") as dosya:
                return re.findall(desen, dosya.read(), re.DOTALL)
        except:
            return []

    def log_temizle(e):
        try:
            with open("islem_log.txt", "w", encoding="utf-8") as dosya:
                dosya.write("")
            islemlog_yap()
        except:
            pass

    # --- Ana İşlem Fonksiyonu (DÜZELTİLDİ) ---
    def islem(girdi_string):
        nonlocal soncevap, islemsonrasi
        duzenli = girdi_string.replace("^", "**").replace("ANS", soncevap).replace("×", "*")

        yerel_degiskenler = {
            'x': sp.Symbol('x'),
            'sin': lambda x: sin(x * pi / 180),
            'cos': lambda x: cos(x * pi / 180),
            'tan': lambda x: tan(x * pi / 180)
        }
        try:
            donusumler = (standard_transformations + (implicit_multiplication_application, rationalize))
            sonuc_obj = parse_expr(duzenli, local_dict=yerel_degiskenler, transformations=donusumler)

            # 1. Sembolik (Çözüm)
            sembolik_str = str(sonuc_obj)

            # 2. Ondalık (Sonuç)
            ondalik_str = ""
            if not sonuc_obj.free_symbols:
                try:
                    val = float(sonuc_obj)
                    if not val.is_integer():
                        ondalik_str = f"{val:.4f}"
                except:
                    pass

            # Ekrana bas
            txt_ondalik.value = ondalik_str
            txt_sonuc.value = sembolik_str
            sonuc_kutusu.update()

            # Son cevabı güncelle (Sonraki işlem için)
            soncevap = sembolik_str

            # KAYDET (3 Parametre ile)
            islem_kaydet(girdi_string, ondalik_str if ondalik_str else sembolik_str, sembolik_str)

            islemsonrasi = True
            gercek_guncelleme(girdi_string)
            page.update()
        except Exception as ex:
            txt_ondalik.value = ""
            txt_sonuc.value = "Hata"
            sonuc_kutusu.update()
            page.update()

    def sonuc_goster(islem, sonuc, cozum):
        page.clean()
        adim=ft.ElevatedButton(text="Adım Adım Çöz...",on_click=lambda e:adim_adim_ekrani(latexe_cevir(islem)))
        geri=ft.ElevatedButton(text="Geri Dön", on_click=normal_yap)
        tum_yazi=ft.TextField(value=f"İşlem: {islem}\nSonuç: {sonuc}\n\n Çözüm: {cozum}\n\n Sonuç: {sonuc}", text_align="left", width=380, read_only=False, border_color="grey",multiline=True)
        page.add(geri,adim,tum_yazi)
        page.update()

    def tus_basma(e):
        nonlocal islemsonrasi
        data = e.control.text
        if girdi_ekranı.value and islemsonrasi == True and data not in ["+", "-", "×", "/", "^"]:
            girdi_ekranı.value = ""
            islemsonrasi = False

        if data == "AC":
            girdi_ekranı.value = ""
        elif data == "∫":
            girdi_ekranı.value += "integrate("
        elif data == "f'(x)":
            girdi_ekranı.value += "diff("
        elif data == "x²":
            girdi_ekranı.value += "^2"
        elif data == "√":
            girdi_ekranı.value += "sqrt("
        elif data == "sin":
            girdi_ekranı.value += "sin("
        elif data == "cos":
            girdi_ekranı.value += "cos("
        elif data == "tan":
            girdi_ekranı.value += "tan("
        elif data == "log":
            girdi_ekranı.value += "log("
        elif data == "<--":
            girdi_ekranı.value = girdi_ekranı.value[:-1]
        elif data == "=":
            islem(girdi_ekranı.value)
            return
        else:
            girdi_ekranı.value += data
        gercek_guncelleme(girdi_ekranı.value)
        page.update()

    # --- Sayfa Düzeni ---
    row_girdi = ft.Row(controls=[girdi_ekranı], alignment=ft.MainAxisAlignment.CENTER)
    row_sonuc = ft.Row(controls=[sonuc_kutusu], alignment=ft.MainAxisAlignment.CENTER)

    kose_stili = ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))
    bilimkose_stili = ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=5))

    normal_row1 = ft.Row(
        [ft.ElevatedButton("7", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.GREY_300,
                           color="black"),
         ft.ElevatedButton("8", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.GREY_300,
                           color="black"),
         ft.ElevatedButton("9", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.GREY_300,
                           color="black"),
         ft.ElevatedButton("+", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.YELLOW_300,
                           color="black")], alignment=ft.MainAxisAlignment.CENTER)
    normal_row2 = ft.Row(
        [ft.ElevatedButton("4", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.GREY_300,
                           color="black"),
         ft.ElevatedButton("5", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.GREY_300,
                           color="black"),
         ft.ElevatedButton("6", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.GREY_300,
                           color="black"),
         ft.ElevatedButton("-", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.YELLOW_300,
                           color="black")], alignment=ft.MainAxisAlignment.CENTER)
    normal_row3 = ft.Row(
        [ft.ElevatedButton("1", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.GREY_300,
                           color="black"),
         ft.ElevatedButton("2", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.GREY_300,
                           color="black"),
         ft.ElevatedButton("3", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.GREY_300,
                           color="black"),
         ft.ElevatedButton("×", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.YELLOW_300,
                           color="black")], alignment=ft.MainAxisAlignment.CENTER)
    normal_row4 = ft.Row(
        [ft.ElevatedButton("0", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.GREY_300,
                           color="black"),
         ft.ElevatedButton(".", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.GREY_300,
                           color="black"),
         ft.ElevatedButton(",", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.GREY_300,
                           color="black"),
         ft.ElevatedButton("/", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.YELLOW_300,
                           color="black")], alignment=ft.MainAxisAlignment.CENTER)
    normal_row5 = ft.Row(
        [ft.ElevatedButton("=", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.LIME_200,
                           color="black"),
         ft.ElevatedButton("ANS", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.BLUE_200,
                           color="black"),
         ft.ElevatedButton("AC", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.DEEP_ORANGE_300,
                           color="black"),
         ft.ElevatedButton("<--", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.RED_400,
                           color="black")], alignment=ft.MainAxisAlignment.CENTER)
    bilim_row1 = ft.Row(
        [ft.ElevatedButton("x²", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200, color="black"),
         ft.ElevatedButton("x", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200, color="black"),
         ft.ElevatedButton("^", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200, color="black"),
         ft.ElevatedButton("√", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200, color="black")],
        alignment=ft.MainAxisAlignment.CENTER)
    bilim_row2 = ft.Row(
        [ft.ElevatedButton("sin", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200, color="black"),
         ft.ElevatedButton("cos", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200, color="black"),
         ft.ElevatedButton("tan", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200, color="black"),
         ft.ElevatedButton("log", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200,
                           color="black")], alignment=ft.MainAxisAlignment.CENTER)
    bilim_row3 = ft.Row(
        [ft.ElevatedButton("∫", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200, color="black"),
         ft.ElevatedButton("f'(x)", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200,
                           color="black"),
         ft.ElevatedButton("(", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200, color="black"),
         ft.ElevatedButton(")", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200, color="black")],
        alignment=ft.MainAxisAlignment.CENTER)

    tutor1 = ft.Text(value="İntegral: integrate(fonskiyon, (x,alt,üst))")
    tutor2 = ft.Text(value="Türev: diff(fonksiyon, x)")
    tutor3 = ft.Text(value="Logaritma: log(tavan,taban)")
    tutorCol= ft.Column(controls=[tutor1,tutor2,tutor3],alignment=ft.MainAxisAlignment.CENTER)
    tutorRow=ft.Row(controls=[tutorCol],alignment=ft.MainAxisAlignment.CENTER)

    def adim_kart_olustur(step_data, derinlik=0):
        # --- 1. Veriyi Güvenli Çekme ---
        if isinstance(step_data, dict):
            inp = step_data.get('input_latex', "")
            out = step_data.get('output_latex', "")
            desc = step_data.get('description', "")
            subs = step_data.get('substeps', [])
            rule = step_data.get('rule', "")
        else:
            inp = getattr(step_data, 'input_latex', "")
            out = getattr(step_data, 'output_latex', "")
            desc = getattr(step_data, 'description', "")
            subs = getattr(step_data, 'substeps', [])
            rule = getattr(step_data, 'rule', "")

        # --- 2. Görsel Ayarlar ---
        sol_bosluk = 20 if derinlik > 0 else 0

        if derinlik == 0:
            kenar_renk = ft.Colors.AMBER
            bg_renk = ft.Colors.GREY_900
            yazi_boyut = 16
        else:
            kenar_renk = ft.Colors.BLUE_GREY_400 if derinlik % 2 != 0 else ft.Colors.TEAL_400
            bg_renk = ft.Colors.with_opacity(0.5, ft.Colors.BLACK)
            yazi_boyut = 14

        # --- 3. LaTeX -> Resim Dönüştürücü Helper (GÜNCELLENDİ) ---
        def latex_widget_getir(latex_metni, on_ek=""):
            if not latex_metni:
                return ft.Container()

            base64_kod = latex_to_image(latex_metni)

            if base64_kod:
                return ft.Row([
                    ft.Text(on_ek, size=14, color="grey", italic=True) if on_ek else ft.Container(),
                    # DEĞİŞİKLİK BURADA: height=40 yerine height=70 yaptık
                    ft.Image(src_base64=base64_kod, height=110, fit=ft.ImageFit.CONTAIN)
                ], alignment=ft.MainAxisAlignment.START)
            else:
                return ft.Text(f"{on_ek} {latex_metni}", font_family="Consolas", color=ft.Colors.GREY_300)

        # --- 4. İçerik Oluşturma ---
        icerik_listesi = [
            ft.Row([
                ft.Icon(ft.Icons.SUBDIRECTORY_ARROW_RIGHT if derinlik > 0 else ft.Icons.CALCULATE,
                        size=16, color=kenar_renk),
                ft.Text(desc, weight="bold", color=ft.Colors.WHITE, size=yazi_boyut, expand=True),
                ft.Text(f"[{rule}]", color=ft.Colors.GREY_500, size=10, italic=True)
            ], alignment=ft.MainAxisAlignment.START),

            ft.Container(
                content=ft.Column([
                    latex_widget_getir(inp, "Girdi:"),
                    latex_widget_getir(out, "=")
                ]),
                padding=ft.padding.only(left=10, top=5)
            )
        ]

        # --- 5. Alt Adımlar ---
        if subs:
            alt_adim_containerlari = []
            for sub in subs:
                alt_kart = adim_kart_olustur(sub, derinlik + 1)
                alt_adim_containerlari.append(alt_kart)

            icerik_listesi.append(
                ft.Column(alt_adim_containerlari, spacing=5)
            )

        return ft.Container(
            content=ft.Column(icerik_listesi, spacing=5),
            padding=10,
            margin=ft.margin.only(left=sol_bosluk, top=5, bottom=5),
            border=ft.border.only(left=ft.BorderSide(4, kenar_renk)),
            border_radius=ft.border_radius.only(top_right=10, bottom_right=10),
            bgcolor=bg_renk
        )

    # --- ANA EKRAN FONKSİYONU ---
    def adim_adim_ekrani(latexForm):
        # Bu fonksiyon butona basılınca çalışır
        if not latexForm: return

        page.clean()

        # Geri Dön Butonu
        geri_btn = ft.ElevatedButton("Geri Dön", on_click=normal_yap, icon=ft.Icons.ARROW_BACK)

        page.add(ft.Row([geri_btn]), ft.Divider())

        try:
            cozucu = solver.MathSolver()
            # Solver'dan MathStep nesnesi alıyoruz
            ana_adim = cozucu.solve(latexForm)

            # Recursive fonksiyonu çağırıp dönen Tek Büyük Container'ı sayfaya ekliyoruz
            final_gorunum = adim_kart_olustur(ana_adim)

            # Sayfanın kaydırılabilir olduğundan emin olalım
            scroll_container = ft.Column(
                controls=[final_gorunum],
                scroll=ft.ScrollMode.AUTO,
                expand=True
            )

            page.add(scroll_container)

        except Exception as ex:
            page.add(ft.Text(f"Hata: {ex}", color="red"))

        page.update()

    def anlat_button_handler(e=None):
        islem(girdi_ekranı.value)
        adim_adim_ekrani(latexe_cevir(girdi_ekranı.value))

    ozel_pencere = ft.Container(
        width=300,
        height=300,
        bgcolor=ft.Colors.GREY_700,
        border_radius=15,
        border=ft.border.all(2, ft.Colors.LIGHT_BLUE_ACCENT_100),
        padding=20,
        visible=False,  # Başlangıçta gizli
        content=ft.Column([
            ft.Text("Fonksiyon Kullanımları:", size=20, weight="bold", color="white"),
            ft.Divider(),
            tutorCol,
            ft.Divider(),
            ft.ElevatedButton("Anladım!", on_click=lambda e: popup_kapat(e), bgcolor="red", color="white")
        ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    )

    # Fonksiyonlar
    def popup_ac(e):
        ozel_pencere.visible = True
        page.update()

    def popup_kapat(e):
        ozel_pencere.visible = False
        page.update()

    # --- NORMAL_YAP FONKSİYONUNU GÜNCELLEME ---
    def normal_yap(e=None):
        page.clean()
        alt_bar.selected_index = 0


    def normal_yap(e=None):
        page.clean()
        anlat_buton = ft.ElevatedButton(text="֍ Adım adım Çöz...", style=kose_stili, bgcolor=ft.Colors.CYAN_ACCENT_100,
                                        width=232, height=50, on_click=lambda e: anlat_button_handler(e))
        anlat_row = ft.Row(controls=[anlat_buton], alignment=ft.MainAxisAlignment.CENTER)
        alt_bar.selected_index = 0
        page.add(formul_container, row_girdi, row_sonuc, normal_row1, normal_row2, normal_row3, normal_row4,
                 normal_row5,anlat_row, alt_bar)
        page.update()

    def bilimsel_yap():
        page.clean()
        anlat_buton = ft.ElevatedButton(text="֍ Adım adım Çöz...", style=kose_stili, bgcolor=ft.Colors.CYAN_ACCENT_100,
                                        width=172, height=50, on_click=lambda e: anlat_button_handler(e))
        soru_isareti=ft.ElevatedButton(text="?",style=kose_stili,width=50,height=50,bgcolor=ft.Colors.BLUE_GREY_700,color="white",on_click=lambda e: popup_ac(e))
        anlat_row = ft.Row(controls=[anlat_buton,soru_isareti], alignment=ft.MainAxisAlignment.CENTER)
        alt_bar.selected_index = 1

        ana_icerik=ft.Column(controls=[formul_container, row_girdi, row_sonuc, bilim_row1, bilim_row2, bilim_row3, normal_row1, normal_row2,
                 normal_row3, normal_row4, normal_row5,anlat_row])

        stack=ft.Stack(controls=[ft.Container(content=ana_icerik,alignment=ft.alignment.center, expand=True),ozel_pencere],alignment=ft.alignment.center)

        page.add(stack,alt_bar)
        page.update()

    def islemlog_yap():
        page.clean()
        page.scroll="auto"
        buton_tipi=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8),color=ft.Colors.CYAN_ACCENT_100,side=ft.BorderSide(color=ft.Colors.WHITE24, width=1))
        alt_bar.selected_index = 3
        temizleme = ft.ElevatedButton(text="Geçmişi temizle", on_click=log_temizle, color="red",
                                      bgcolor=ft.Colors.GREY_900,style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8),color=ft.Colors.CYAN_ACCENT_100,side=ft.BorderSide(color=ft.Colors.WHITE24, width=1)))
        temizlemeRow=ft.Row(controls=[temizleme], spacing=5,alignment=ft.MainAxisAlignment.CENTER)
        gecmis_listesi = ft.Column(scroll=ft.ScrollMode.AUTO)
        yazi=ft.Text(value="-╣ İşlem Geçmişi ╠-",size=25)
        yaziRow=ft.Row(controls=[yazi],alignment=ft.MainAxisAlignment.CENTER)
        page.add(alt_bar,yaziRow, temizlemeRow, gecmis_listesi)
        eslesmeler = islem_oku()
        if eslesmeler:
            # eslesmeler artık bir demet (tuple) döndüğü için (İşlem, Sonuç, Çözüm)
            # bunu stringe çevirerek ekrana basmalıyız.
            for temp in reversed(eslesmeler):
                # temp[0] -> İşlem, temp[1] -> Sonuç (ondalık), temp[2] -> Çözüm
                gosterilecek_metin = f"{temp[0]} = {temp[2]}"
                if temp[1]: gosterilecek_metin += f"  ({temp[1]})"

                gecmis_listesi.controls.append(ft.Row(
                    [ft.ElevatedButton(color="grey", text=f"İşlem: {temp[0]}\nSonuç: {temp[1]}",style=buton_tipi,height=50,width=300,on_click=lambda e,islem=temp[0],sonuc=temp[1],cozum=temp[2]:sonuc_goster(islem,sonuc,cozum))], alignment=ft.MainAxisAlignment.CENTER))

                page.update()

    def gorsel_yap():
        page.clean()
        page.add(alt_bar)


    def ekran_degistir(e):
        indis = e.control.selected_index
        page.clean()
        if indis == 0:
            normal_yap()
        elif indis == 1:
            bilimsel_yap()
        elif indis== 2:
            gorsel_yap()
        elif indis == 3:
            islemlog_yap()

    normal_yap()

    print(latexe_cevir("5/2"))

ft.app(target=main)