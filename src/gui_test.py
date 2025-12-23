import flet as ft
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import threading
import sympy as sp
import Solver as solver
from Solver import MathStep
from sympy import sin, cos, tan, pi
from latex2sympy2_extended import latex2sympy
import re
from PIL import Image as PILImage
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, \
    rationalize
from inference import predict_from_base64

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

    # --- Görsel İşleme Değişkenleri ---
    secilen_resim = ft.Image(
        src_base64=BOS_RESIM,
        width=300,
        height=300,
        fit=ft.ImageFit.CONTAIN,
        visible=False
    )

    def pil_to_base64(pil_image):
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # --- KIRPMA MODÜLÜ (Geniş / Yatay Tasarım) ---
    def kirpma_penceresi_ac(e):
        dosya_yolu = secilen_resim.data
        if not dosya_yolu:
            page.snack_bar = ft.SnackBar(ft.Text("Lütfen önce bir resim seçin!"))
            page.snack_bar.open = True
            page.update()
            return

        # 1. Sliderlar (Yerel)
        yerel_slider_sol = ft.Slider(label="Sol Kesik: {value}%", min=0, max=100, divisions=100, value=0)
        yerel_slider_sag = ft.Slider(label="Sağ Kesik: {value}%", min=0, max=100, divisions=100, value=100)
        yerel_slider_ust = ft.Slider(label="Üst Kesik: {value}%", min=0, max=100, divisions=100, value=0)
        yerel_slider_alt = ft.Slider(label="Alt Kesik: {value}%", min=0, max=100, divisions=100, value=100)

        # 2. Önizleme Resmi
        yerel_onizleme = ft.Image(
            src_base64=BOS_RESIM,
            width=450, height=450,
            fit=ft.ImageFit.CONTAIN
        )


        # 3. Kırpma Mantığı
        def yerel_anlik_kirp(e=None):

            try:
                img = PILImage.open(dosya_yolu)
                w, h = img.size

                left = (yerel_slider_sol.value / 100) * w
                top = (yerel_slider_ust.value / 100) * h
                right = (yerel_slider_sag.value / 100) * w
                bottom = (yerel_slider_alt.value / 100) * h

                if left >= right: left = right - 10
                if top >= bottom: top = bottom - 10

                cropped_img = img.crop((left, top, right, bottom))

                # Önizleme (Thumbnail)
                onizleme_kopyasi = cropped_img.copy()
                onizleme_kopyasi.thumbnail((450, 450))
                yerel_onizleme.src_base64 = pil_to_base64(onizleme_kopyasi)

                if yerel_onizleme.page:
                    yerel_onizleme.update()
            except Exception as ex:
                print(f"Hata: {ex}")

        # 4. Kaydetme Mantığı
        def yerel_kaydet(e):
            try:
                img = PILImage.open(dosya_yolu)
                w, h = img.size
                left = (yerel_slider_sol.value / 100) * w
                top = (yerel_slider_ust.value / 100) * h
                right = (yerel_slider_sag.value / 100) * w
                bottom = (yerel_slider_alt.value / 100) * h

                if left >= right: left = right - 10
                if top >= bottom: top = bottom - 10

                final_crop = img.crop((left, top, right, bottom))
                secilen_resim.src_base64 = pil_to_base64(final_crop)
                secilen_resim.src = ""
                secilen_resim.update()

                dialog_penceresi.open = False
                page.update()
                page.overlay.remove(dialog_penceresi)
            except Exception as ex:
                print(f"Kaydetme Hatası: {ex}")

        def iptal_et(e):
            dialog_penceresi.open = False
            page.update()
            page.overlay.remove(dialog_penceresi)

        yerel_slider_sol.on_change = yerel_anlik_kirp
        yerel_slider_sag.on_change = yerel_anlik_kirp
        yerel_slider_ust.on_change = yerel_anlik_kirp
        yerel_slider_alt.on_change = yerel_anlik_kirp

        yerel_anlik_kirp()

        # Tasarım (Sol - Orta - Sağ)
        sol_panel = ft.Column(
            [ft.Text("↔️ Yatay", weight="bold"), ft.Text("Sol:"), yerel_slider_sol, ft.Divider(), ft.Text("Sağ:"),
             yerel_slider_sag], width=200, alignment=ft.MainAxisAlignment.CENTER)
        sag_panel = ft.Column(
            [ft.Text("↕️ Dikey", weight="bold"), ft.Text("Üst:"), yerel_slider_ust, ft.Divider(), ft.Text("Alt:"),
             yerel_slider_alt], width=200, alignment=ft.MainAxisAlignment.CENTER)
        orta_panel = ft.Container(content=yerel_onizleme, alignment=ft.alignment.center,
                                  border=ft.border.all(1, "grey"), border_radius=10, padding=5, expand=True)

        ana_icerik = ft.Row([sol_panel, ft.VerticalDivider(width=1, color="grey"), orta_panel,
                             ft.VerticalDivider(width=1, color="grey"), sag_panel], expand=True,
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        dialog_penceresi = ft.AlertDialog(
            title=ft.Text("Gelişmiş Kırpma Aracı"),
            modal=True,
            content=ft.Container(content=ana_icerik, width=900, height=500),
            actions=[ft.TextButton("İptal", on_click=iptal_et),
                     ft.ElevatedButton("Kaydet", on_click=yerel_kaydet, bgcolor="green", color="white")]
        )

        page.overlay.append(dialog_penceresi)
        dialog_penceresi.open = True
        page.update()


    def gorsel_isleme(e=None):
        gorsel=ft.Image(src_base64=secilen_resim.src_base64)
        container=ft.Container(content=gorsel)

        if model_switch.value==True:
            print("Hazır modele gönderildi.")


        else:
            print("Özgün modele gönderildi.")

            adim_adim_ekrani(predict_from_base64(secilen_resim.src_base64))

    # --- Navigasyon Barı ---
    alt_bar = ft.NavigationBar(
        destinations=[
            ft.NavigationBarDestination(icon=ft.Icons.DASHBOARD, label="Günlük"),
            ft.NavigationBarDestination(icon=ft.Icons.SCIENCE_OUTLINED, label="Bilimsel"),
            ft.NavigationBarDestination(icon=ft.Icons.ADD_PHOTO_ALTERNATE_OUTLINED, label="Görsel"),
            ft.NavigationBarDestination(icon=ft.Icons.BOOKMARK_BORDER, selected_icon=ft.Icons.BOOKMARK, label="Geçmiş"),
        ],
        on_change=lambda e: ekran_degistir(e)
    )
    page.navigation_bar = alt_bar

    # --- Hesap Makinesi Değişkenleri ---
    soncevap = "0"
    islemsonrasi = False
    debounce_timer = None
    son_gecerli_resim = BOS_RESIM

    # --- 1. Formül Resmi ---
    formul_resmi = ft.Image(src_base64=BOS_RESIM, width=400, height=120, fit=ft.ImageFit.CONTAIN, gapless_playback=True,
                            visible=True)
    formul_container = ft.Container(content=formul_resmi, padding=5, bgcolor=ft.Colors.BLACK38, border_radius=10,
                                    alignment=ft.alignment.center, visible=True, height=130)

    # --- 2. Sonuç Ekranı ---
    txt_ondalik = ft.Text(value="", size=16, color=ft.Colors.GREY_400)
    txt_sonuc = ft.Text(value="", size=20, weight="bold")
    sonuc_kutusu = ft.Container(content=ft.Row([txt_ondalik, txt_sonuc], alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                               vertical_alignment=ft.CrossAxisAlignment.CENTER), width=380, height=50,
                                border=ft.border.all(1, "grey"), border_radius=5,
                                padding=ft.padding.only(left=10, right=10), bgcolor=ft.Colors.GREY_900)

    # --- Helper Fonksiyonlar (Latex vb) ---
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
            clean_text = text_val.replace("diff", "Derivative").replace("integrate", "Integral").replace("^",
                                                                                                         "**").replace(
                "×", "*")
            gorsel_ayarlar = {'x': sp.Symbol('x'), 'Derivative': sp.Derivative, 'Integral': sp.Integral,
                              'integrate': sp.Integral, 'sin': sp.Function(r'\sin'), 'cos': sp.Function(r'\cos'),
                              'tan': sp.Function(r'\tan'), 'sqrt': sp.Function(r'\sqrt'), 'log': sp.log, 'ln': sp.log}
            donusumler = (standard_transformations + (implicit_multiplication_application,))
            expr = parse_expr(clean_text, local_dict=gorsel_ayarlar, transformations=donusumler, evaluate=False)

            if inspect.isclass(expr) or isinstance(expr, sp.core.function.UndefinedFunction):
                latex_code = expr.__name__
            else:
                latex_code = sp.latex(expr)
            latex_code = latex_code.replace(r"\limits", "")
        except Exception:
            latex_code = ""
        return latex_code

    def gercek_guncelleme(text_val):
        nonlocal son_gecerli_resim
        if not text_val:
            formul_resmi.src_base64 = BOS_RESIM
            formul_resmi.update()
            return
        base64_data = latex_to_image(latexe_cevir(text_val))
        if base64_data:
            son_gecerli_resim = base64_data
            formul_resmi.src_base64 = base64_data
            formul_resmi.update()

    def on_change_handler(e):
        nonlocal debounce_timer
        if debounce_timer: debounce_timer.cancel()
        debounce_timer = threading.Timer(0.4, gercek_guncelleme, args=[e.control.value])
        debounce_timer.start()

    girdi_ekranı = ft.TextField(value="", text_align="left", width=380, read_only=False, border_color="grey",
                                on_change=on_change_handler, text_style=ft.TextStyle(size=18))

    # --- Dosya Kayıt ---
    def islem_kaydet(islem, sonuc, cozum):
        try:
            with open("islem_log.txt", "a", encoding="utf-8") as dosya:
                dosya.write(f"İşlem: {islem}\nSonuç: {sonuc}\nÇözüm: {cozum}\n\n")
        except:
            pass

    def islem_oku():
        try:
            with open("islem_log.txt", "r", encoding="utf-8") as dosya:
                return re.findall(r"İşlem:\s*(.*?)\nSonuç:\s*(.*?)\nÇözüm:\s*(.*?)(?=\n\n)", dosya.read(), re.DOTALL)
        except:
            return []

    def log_temizle(e):
        try:
            with open("islem_log.txt", "w", encoding="utf-8") as dosya:
                dosya.write("")
            islemlog_yap()
        except:
            pass

    # --- Ana Hesaplama Mantığı ---
    def islem(girdi_string):
        nonlocal soncevap, islemsonrasi
        duzenli = girdi_string.replace("^", "**").replace("ANS", soncevap).replace("×", "*")
        yerel_degiskenler = {'x': sp.Symbol('x'), 'sin': lambda x: sin(x * pi / 180),
                             'cos': lambda x: cos(x * pi / 180), 'tan': lambda x: tan(x * pi / 180)}
        try:
            donusumler = (standard_transformations + (implicit_multiplication_application, rationalize))
            sonuc_obj = parse_expr(duzenli, local_dict=yerel_degiskenler, transformations=donusumler)

            raw_sembolik = str(sonuc_obj)
            sembolik_str = raw_sembolik.replace("**", "^")
            ondalik_str = ""
            if not sonuc_obj.free_symbols:
                try:
                    val = float(sonuc_obj)
                    if not val.is_integer(): ondalik_str = f"{val:.4f}"
                except:
                    pass

            txt_ondalik.value = ondalik_str
            txt_sonuc.value = sembolik_str
            sonuc_kutusu.update()

            soncevap = raw_sembolik
            islem_kaydet(girdi_string, ondalik_str if ondalik_str else sembolik_str, sembolik_str)
            islemsonrasi = True
            gercek_guncelleme(girdi_string)
            page.update()
        except Exception:
            txt_ondalik.value = ""
            txt_sonuc.value = "Hata"
            sonuc_kutusu.update()
            page.update()

    def sonuc_goster(islem, sonuc, cozum):
        page.clean()
        adim = ft.ElevatedButton(text="֍ Adım Adım Çöz...",style=kose_stili, bgcolor=ft.Colors.CYAN_ACCENT_100,
                                        width=172, height=50, on_click=lambda e: adim_adim_ekrani(latexe_cevir(islem)))
        geri = ft.ElevatedButton(text="Geri Dön", on_click=normal_yap)
        tum_yazi = ft.TextField(value=f"İşlem: {islem}\n\n Çözüm: {cozum}\n\n Sonuç: {sonuc}",
                                text_align="left", width=380, read_only=True, border_color="grey", multiline=True)
        tus_row = ft.Row(controls=[adim,geri],alignment=ft.MainAxisAlignment.CENTER)
        yazi_row=ft.Row(controls=[tum_yazi],alignment=ft.MainAxisAlignment.CENTER)
        ana_column=ft.Column(controls=[tus_row,yazi_row],alignment=ft.MainAxisAlignment.CENTER)
        page.add(ana_column)
        page.update()

    def tus_basma(e):
        nonlocal islemsonrasi
        data = e.control.text
        if girdi_ekranı.value and islemsonrasi and data not in ["+", "-", "×", "/", "^"]:
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
            islem(girdi_ekranı.value); return
        else:
            girdi_ekranı.value += data

        gercek_guncelleme(girdi_ekranı.value)
        page.update()

    # --- UI Tanımları ---
    row_girdi = ft.Row([girdi_ekranı], alignment=ft.MainAxisAlignment.CENTER)
    row_sonuc = ft.Row([sonuc_kutusu], alignment=ft.MainAxisAlignment.CENTER)
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
         ft.ElevatedButton("AC", on_click=tus_basma, style=kose_stili, width=50, height=50,
                           bgcolor=ft.Colors.DEEP_ORANGE_300, color="black"),
         ft.ElevatedButton("<--", on_click=tus_basma, style=kose_stili, width=50, height=50, bgcolor=ft.Colors.RED_400,
                           color="black")], alignment=ft.MainAxisAlignment.CENTER)
    bilim_row1 = ft.Row(
        [ft.ElevatedButton("x²", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200,
                           color="black"),
         ft.ElevatedButton("x", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200,
                           color="black"),
         ft.ElevatedButton("^", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200,
                           color="black"),
         ft.ElevatedButton("√", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200,
                           color="black")], alignment=ft.MainAxisAlignment.CENTER)
    bilim_row2 = ft.Row(
        [ft.ElevatedButton("sin", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200,
                           color="black"),
         ft.ElevatedButton("cos", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200,
                           color="black"),
         ft.ElevatedButton("tan", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200,
                           color="black"),
         ft.ElevatedButton("log", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200,
                           color="black")], alignment=ft.MainAxisAlignment.CENTER)
    bilim_row3 = ft.Row(
        [ft.ElevatedButton("∫", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200,
                           color="black"),
         ft.ElevatedButton("f'(x)", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200,
                           color="black"),
         ft.ElevatedButton("(", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200,
                           color="black"),
         ft.ElevatedButton(")", on_click=tus_basma, style=bilimkose_stili, width=50, bgcolor=ft.Colors.ORANGE_200,
                           color="black")], alignment=ft.MainAxisAlignment.CENTER)

    tutorCol = ft.Column([ft.Text("İntegral: integrate(fonskiyon, (x,alt,üst))"), ft.Text("Türev: diff(fonksiyon, x)"),
                          ft.Text("Logaritma: log(tavan,taban)")], alignment=ft.MainAxisAlignment.CENTER)

    # --- Adım Adım Çözüm Gösterimi ---
    def adim_kart_olustur(step_data, derinlik=0):
        if isinstance(step_data, dict):
            inp, out, desc, subs, rule = step_data.get('input_latex', ""), step_data.get('output_latex',
                                                                                         ""), step_data.get(
                'description', ""), step_data.get('substeps', []), step_data.get('rule', "")
        else:
            inp, out, desc, subs, rule = getattr(step_data, 'input_latex', ""), getattr(step_data, 'output_latex',
                                                                                        ""), getattr(step_data,
                                                                                                     'description',
                                                                                                     ""), getattr(
                step_data, 'substeps', []), getattr(step_data, 'rule', "")

        sol_bosluk = 20 if derinlik > 0 else 0
        kenar_renk, bg_renk, yazi_boyut = (ft.Colors.AMBER, ft.Colors.GREY_900, 16) if derinlik == 0 else (
            ft.Colors.BLUE_GREY_400 if derinlik % 2 != 0 else ft.Colors.TEAL_400,
            ft.Colors.with_opacity(0.5, ft.Colors.BLACK), 14)

        def latex_widget_getir(latex_metni, on_ek=""):
            if not latex_metni: return ft.Container()
            base64_kod = latex_to_image(latex_metni)
            if base64_kod:
                return ft.Row([ft.Text(on_ek, size=14, color="grey", italic=True) if on_ek else ft.Container(),
                               ft.Image(src_base64=base64_kod, height=110, fit=ft.ImageFit.CONTAIN)],
                              alignment=ft.MainAxisAlignment.START)
            return ft.Text(f"{on_ek} {latex_metni}", font_family="Consolas", color=ft.Colors.GREY_300)

        icerik_listesi = [ft.Row(
            [ft.Icon(ft.Icons.SUBDIRECTORY_ARROW_RIGHT if derinlik > 0 else ft.Icons.CALCULATE, size=16,
                     color=kenar_renk),
             ft.Text(desc, weight="bold", color=ft.Colors.WHITE, size=yazi_boyut, expand=True),
             ft.Text(f"[{rule}]", color=ft.Colors.GREY_500, size=10, italic=True)],
            alignment=ft.MainAxisAlignment.START), ft.Container(content=latex_widget_getir(inp, "Girdi:"),
                                                                padding=ft.padding.only(left=10, top=5))]
        if subs: icerik_listesi.append(ft.Column([adim_kart_olustur(sub, derinlik + 1) for sub in subs], spacing=5))
        icerik_listesi.append(
            ft.Container(content=latex_widget_getir(out, "="), padding=ft.padding.only(left=10, top=5)))

        return ft.Container(content=ft.Column(icerik_listesi, spacing=5), padding=10,
                            margin=ft.margin.only(left=sol_bosluk, top=5, bottom=5),
                            border=ft.border.only(left=ft.BorderSide(4, kenar_renk)),
                            border_radius=ft.border_radius.only(top_right=10, bottom_right=10), bgcolor=bg_renk)

    def adim_adim_ekrani(latexForm):
        if not latexForm: return
        page.clean()
        page.add(ft.Row([ft.ElevatedButton("Geri Dön", on_click=normal_yap, icon=ft.Icons.ARROW_BACK)]), ft.Divider())
        try:
            cozucu = solver.MathSolver()
            ana_adim = cozucu.solve(latexForm)
            page.add(ft.Column(controls=[adim_kart_olustur(ana_adim)], scroll=ft.ScrollMode.AUTO, expand=True))
        except Exception as ex:
            page.add(ft.Text(f"Hata: {ex}", color="red"))
        page.update()

    def anlat_button_handler(e=None):
        islem(girdi_ekranı.value)
        adim_adim_ekrani(latexe_cevir(girdi_ekranı.value))

    # --- Popup ve Pencere Mantığı ---
    ozel_pencere = ft.Container(width=300, height=300, bgcolor=ft.Colors.GREY_700, border_radius=15,
                                border=ft.border.all(2, ft.Colors.LIGHT_BLUE_ACCENT_100), padding=20, visible=False,
                                content=ft.Column(
                                    [ft.Text("Fonksiyon Kullanımları:", size=20, weight="bold", color="white"),
                                     ft.Divider(), tutorCol, ft.Divider(),
                                     ft.ElevatedButton("Anladım!", on_click=lambda e: popup_kapat(e), bgcolor="red",
                                                       color="white")], alignment=ft.MainAxisAlignment.CENTER,
                                    horizontal_alignment=ft.CrossAxisAlignment.CENTER))

    def popup_ac(e):
        ozel_pencere.visible = True; page.update()

    def popup_kapat(e):
        ozel_pencere.visible = False; page.update()

    # --- SAYFA FONKSİYONLARI ---
    def normal_yap(e=None):
        page.clean()
        page.scroll = "Auto"
        anlat_buton = ft.ElevatedButton(text="֍ Adım adım Çöz...", style=kose_stili, bgcolor=ft.Colors.CYAN_ACCENT_100,
                                        width=232, height=50, on_click=lambda e: anlat_button_handler(e))
        alt_bar.selected_index = 0
        page.add(formul_container, row_girdi, row_sonuc, normal_row1, normal_row2, normal_row3, normal_row4,
                 normal_row5, ft.Row([anlat_buton], alignment=ft.MainAxisAlignment.CENTER), alt_bar)
        page.update()

    def bilimsel_yap():
        page.clean()
        page.scroll = "Auto"
        anlat_buton = ft.ElevatedButton(text="֍ Adım adım Çöz...", style=kose_stili, bgcolor=ft.Colors.CYAN_ACCENT_100,
                                        width=172, height=50, on_click=lambda e: anlat_button_handler(e))
        soru_isareti = ft.ElevatedButton(text="?", style=kose_stili, width=50, height=50,
                                         bgcolor=ft.Colors.BLUE_GREY_700, color="white", on_click=lambda e: popup_ac(e))
        alt_bar.selected_index = 1
        ana_icerik = ft.Column(
            controls=[formul_container, row_girdi, row_sonuc, bilim_row1, bilim_row2, bilim_row3, normal_row1,
                      normal_row2, normal_row3, normal_row4, normal_row5,
                      ft.Row([anlat_buton, soru_isareti], alignment=ft.MainAxisAlignment.CENTER)])
        stack = ft.Stack(
            controls=[ft.Container(content=ana_icerik, alignment=ft.alignment.center, expand=True), ozel_pencere],
            alignment=ft.alignment.center)
        page.add(stack, alt_bar)
        page.update()

    def islemlog_yap():
        page.clean()
        page.scroll = "auto"
        buton_tipi = ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8), color=ft.Colors.CYAN_ACCENT_100,
                                    side=ft.BorderSide(color=ft.Colors.WHITE24, width=1))
        alt_bar.selected_index = 3
        temizleme = ft.ElevatedButton(text="Geçmişi temizle", on_click=log_temizle, color="red",
                                      bgcolor=ft.Colors.GREY_900, style=buton_tipi)
        gecmis_listesi = ft.Column(scroll=ft.ScrollMode.AUTO)
        page.add(alt_bar,
                 ft.Row([ft.Text(value="-╣ İşlem Geçmişi ╠-", size=25)], alignment=ft.MainAxisAlignment.CENTER),
                 ft.Row([temizleme], spacing=5, alignment=ft.MainAxisAlignment.CENTER), gecmis_listesi)
        for temp in reversed(islem_oku()):
            gosterilecek_metin = f"{temp[0]} = {temp[2]}" + (f"  ({temp[1]})" if temp[1] else "")
            gecmis_listesi.controls.append(ft.Row(
                [ft.ElevatedButton(color="grey", text=f"İşlem: {temp[0]}\nSonuç: {temp[1]}", style=buton_tipi,
                                   height=50, width=300,
                                   on_click=lambda e, i=temp[0], s=temp[1], c=temp[2]: sonuc_goster(i, s, c))],
                alignment=ft.MainAxisAlignment.CENTER))
        page.update()

    # --- GÖRSEL İŞLEME UI ---
    resim_placeholder = ft.Container(content=ft.Text("Görseliniz burada görünecek...", color="grey"), width=300,
                                     height=300, border=ft.border.all(2, ft.Colors.BLUE_200), border_radius=10,
                                     alignment=ft.alignment.center, visible=True)
    kirp_butonu = ft.IconButton(icon=ft.Icons.CROP, icon_color="white", bgcolor=ft.Colors.GREY_800,
                                tooltip="Resmi Kırp", disabled=True, on_click=kirpma_penceresi_ac)

    def dosya_secildi(e: ft.FilePickerResultEvent):

        if e.files and len(e.files) > 0:
            yol = e.files[0].path
            secilen_resim.data = yol  # Yol Hafızaya
            try:
                with open(yol, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                secilen_resim.src_base64 = encoded_string
                secilen_resim.src = ""
                secilen_resim.visible = True
                resim_placeholder.visible = False
                kirp_butonu.disabled = False
                kirp_butonu.update()
                secilen_resim.update()
                resim_placeholder.update()
                if secilen_resim.src_base64 != BOS_RESIM:
                    anlat_container.visible = True
                else:
                    anlat_container.visible = False
                page.update()
            except Exception as ex:
                print(f"Hata: {ex}")

    file_picker = ft.FilePicker(on_result=dosya_secildi)
    page.overlay.append(file_picker)

    model_switch=ft.Switch(value=True, active_color=ft.Colors.CYAN_200)

    anlat_buton_gorsel = ft.ElevatedButton(text="֍ Adım adım Çöz...", style=kose_stili, bgcolor=ft.Colors.CYAN_ACCENT_100,
                                    width=172, height=50, on_click=lambda e: gorsel_isleme(e))

    anlat_container = ft.Container(content=anlat_buton_gorsel, visible=False)

    def gorsel_yap():
        page.clean()
        if secilen_resim.src_base64!=BOS_RESIM:
            anlat_container.visible = True
        else:
            anlat_container.visible = False
        page.scroll = None
        yazi_row = ft.Row([ft.Text(value="֍ Görüntü İşleme", size=30, color=ft.Colors.CYAN_ACCENT_100)],
                          alignment=ft.MainAxisAlignment.CENTER)
        switch_row = ft.Row(
            [ft.Text("Özgün Model"), model_switch, ft.Text("Hazır Model")],
            alignment=ft.MainAxisAlignment.CENTER)
        gorsel_button = ft.ElevatedButton(icon=ft.Icons.ADD_A_PHOTO, text="Görsel ekleyin",
                                          bgcolor=ft.Colors.CYAN_ACCENT_700, color=ft.Colors.BLACK, style=kose_stili,
                                          on_click=lambda _: file_picker.pick_files(allow_multiple=False,
                                                                                    allowed_extensions=["png", "jpg",
                                                                                                        "jpeg"]))
        resim_alani = ft.Stack([resim_placeholder, secilen_resim], alignment=ft.alignment.center)
        ana_column = ft.Column(
            [yazi_row, switch_row, ft.Row([gorsel_button, kirp_butonu], alignment=ft.MainAxisAlignment.CENTER),
             ft.Row([resim_alani], alignment=ft.MainAxisAlignment.CENTER),anlat_container], alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True, spacing=20)
        page.navigation_bar = alt_bar
        page.add(ana_column)

    def ekran_degistir(e):
        indis = e.control.selected_index
        page.clean()
        if indis == 0:
            normal_yap()
        elif indis == 1:
            bilimsel_yap()
        elif indis == 2:
            gorsel_yap()
        elif indis == 3:
            islemlog_yap()

    normal_yap()
    #naber
    print(latexe_cevir("5/2"))


ft.app(target=main)