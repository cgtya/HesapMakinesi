import json
from dataclasses import dataclass, field
from typing import List, Optional, Any

from sympy import (
    Symbol, latex, Integral, Derivative, Add, Mul, Pow,
    diff, Basic, sympify
)
from latex2sympy2_extended import latex2sympy

# Manuel integral kurallari
from sympy.integrals.manualintegrate import (
    manualintegrate, integral_steps, AtomicRule,
    AddRule, ConstantTimesRule, ConstantRule, PowerRule, NestedPowRule, URule,
    PartsRule, AlternativeRule, RewriteRule, PiecewiseRule, HyperbolicRule,
    TrigRule, ExpRule, ReciprocalRule, DontKnowRule
)


# --- 1. VERI MODELI ---
@dataclass
class MathStep:
    type: str  # "Integral", "Derivative", "Simplification"
    rule: str  # kural Adi
    input_latex: str  # islem oncesi
    output_latex: str  # islem sonrasi
    description: str  # aciklama
    substeps: List['MathStep'] = field(default_factory=list)  # alt adimlar

    def to_dict(self):
        return {
            "type": self.type,
            "rule": self.rule,
            "input": self.input_latex,
            "output": self.output_latex,
            "description": self.description,
            "substeps": [s.to_dict() for s in self.substeps]
        }


# --- 2. INTEGRAL KURAL CEVIRICISI ---
class IntegralConverter:
    """
    SymPy kural nesnelerini MathStep nesnesine cevirir.
    """

    def convert(self, rule: Any, current_expr: Basic = None) -> MathStep:
        rule_name = rule.__class__.__name__.replace("Rule", "")

        # Varsayilan Adim Yapisi
        step = MathStep(
            type="Integral",
            rule=rule_name,
            input_latex=fr"\int {latex(rule.integrand)} d{latex(rule.variable)}",
            output_latex="",  # Simdilik bos, asagida doldurulacak
            description=f"{rule_name} uygulaniyor.",
            substeps=[]
        )

        # --- A. YAPISAL KURALLAR ---

        if isinstance(rule, ConstantTimesRule):
            step.description = "Sabit Katsayı Kuralı: Sabit sayı integralin dışına alınır."
            const_latex = latex(rule.constant)

            # 1. Alt integrali coz (Recursive)
            sub_step_obj = self.convert(rule.substep, None)
            step.substeps.append(sub_step_obj)

            # 2. Nihai sonucu hesapla (Evaluating)
            # Bu kisim sorununuzu cozen yer: rule.eval() sonucu hesaplar.
            final_res_latex = latex(rule.eval())
            step.output_latex = final_res_latex

            # 3. Carpma islemini gosteren ara adim ekle
            # Eger alt adimin sonucu varsa ve katsayi varsa, carpimi gosterelim
            sub_res_latex = sub_step_obj.output_latex
            if sub_res_latex and sub_res_latex != "0":
                step.substeps.append(MathStep(
                    type="Simplification",
                    rule="Multiplication",
                    input_latex=fr"{const_latex} \cdot \left( {sub_res_latex} \right)",
                    output_latex=final_res_latex,
                    description="Katsayı ile integral sonucu çarpılır ve sadeleştirilir.",
                    substeps=[]
                ))

        elif isinstance(rule, AddRule):
            step.description = "Toplam Kuralı: Terimlerin ayrı ayrı integrali alınır."

            # Alt adimlari ekle
            for sub in rule.substeps:
                step.substeps.append(self.convert(sub))

            # Nihai sonucu hesapla (rule.eval tum toplami hesaplar)
            step.output_latex = latex(rule.eval())


        elif isinstance(rule, PartsRule):
            step.description = "Kısmi İntegrasyon (Integration by Parts)"
            u_latex = latex(rule.u)
            dv_latex = latex(rule.dv)

            # Formül gösterimi
            step.output_latex = latex(rule.eval())  # Sonucu direkt yaz

            # Detay adimlari
            step.substeps.append(MathStep(
                "Info", "Setup", "", "",
                f"Seçimler: u = {u_latex}, dv = {dv_latex}"
            ))

            if rule.v_step:
                v_step_obj = self.convert(rule.v_step, None)
                v_step_obj.description = "dv'den v'yi bulma:"
                step.substeps.append(v_step_obj)

            if rule.second_step:
                sub_step_obj = self.convert(rule.second_step, None)
                step.substeps.append(sub_step_obj)


        elif isinstance(rule, URule):
            step.description = "Değişken Değiştirme (U-Substitution)"
            u_latex = latex(rule.u_func)
            step.output_latex = latex(rule.eval())

            step.substeps.append(MathStep(
                "Info", "Setup", "", "", f"u = {u_latex} dönüşümü yapılır."
            ))

            if rule.substep:
                step.substeps.append(self.convert(rule.substep, None))


        elif isinstance(rule, RewriteRule):
            step.description = "Yeniden Yazma (Rewrite)"
            rewritten_latex = latex(rule.rewritten)
            step.output_latex = latex(rule.eval())

            step.substeps.append(MathStep(
                "Simplification", "Rewrite",
                latex(rule.integrand), rewritten_latex,
                "İfade daha kolay integral alınabilir hale getirilir."
            ))

            if rule.substep:
                step.substeps.append(self.convert(rule.substep, getattr(rule, 'rewritten', None)))


        # --- B. TEMEL KURALLAR (ATOMİK) ---
        elif isinstance(rule, AtomicRule):
            # Basit kurallarin sonucunu direkt hesapla
            step.output_latex = latex(rule.eval())

            if isinstance(rule, ConstantRule):
                step.description = "Sabit Kuralı: Sabit sayının yanına x eklenir."
            elif isinstance(rule, PowerRule):
                step.description = "Üs Kuralı: Üs bir arttırılır, ifade yeni üsse bölünür."
            elif isinstance(rule, ExpRule):
                step.description = "Üstel Fonksiyon Kuralı"
            elif isinstance(rule, TrigRule):
                step.description = "Trigonometrik İntegral Kuralı"
            elif isinstance(rule, ReciprocalRule):
                step.description = "1/x İntegrali (ln|x|)"
            else:
                step.description = f"Temel Kural: {rule_name}"

        # --- C. DIĞER ---
        else:
            step.description = f"Kural: {rule_name}"
            try:
                step.output_latex = latex(rule.eval())
            except:
                step.output_latex = ""

            if hasattr(rule, 'substep') and rule.substep:
                step.substeps.append(self.convert(rule.substep, None))

        return step


# --- 3. TUREV CONVERTER (KISA) ---
class DerivativeConverter:
    def derive(self, expr: Basic, var: Symbol) -> MathStep:
        # Basit türev adımı (Detaylı ağaç yerine işlem sonucu)
        res = diff(expr, var)
        return MathStep(
            "Derivative", "Diff",
            latex(expr), latex(res),
            f"{var} değişkenine göre türev alınır.",
            []
        )


# --- 4. ANA COZUCU SINIF ---
class MathSolver:
    def __init__(self):
        self.converter = IntegralConverter()

    def solve(self, latex_input: str) -> MathStep:
        try:
            # 1. Latex Parse
            expr = latex2sympy(latex_input)

            # 2. Cozum Agacini Gez
            result_step = self._traverse_expr(expr)

            return result_step

        except Exception as e:
            return MathStep(
                type="Error",
                rule="Exception",
                input_latex=latex_input,
                output_latex="Hata",
                description=f"Çözüm hatası: {str(e)}",
                substeps=[]
            )

    def _traverse_expr(self, expr: Basic) -> MathStep:
        # --- A. INTEGRAL ---
        if isinstance(expr, Integral):
            function = expr.function
            limits = expr.limits
            var = limits[0][0]

            try:
                # SymPy manualintegrate adimlarini al
                rule_tree = integral_steps(function, var)
                step = self.converter.convert(rule_tree, function)

                # Belirli integral ise (Sinirlar varsa)
                if len(limits[0]) == 3:
                    lower, upper = limits[0][1], limits[0][2]
                    indef_result = manualintegrate(function, var)
                    val_upper = indef_result.subs(var, upper)
                    val_lower = indef_result.subs(var, lower)

                    final_val = val_upper - val_lower

                    bound_step = MathStep(
                        "DefiniteIntegral", "Limits",
                        latex(indef_result),
                        latex(final_val),
                        f"Sınırlar yerine konulur: F({upper}) - F({lower})"
                    )
                    step.substeps.append(bound_step)
                    step.output_latex = latex(final_val)

                return step
            except Exception as e:
                # Fallback: Eger manualintegrate yapamazsa standart integrate dene
                return MathStep("Integral", "Standard", latex(expr), latex(expr.doit()), "İntegral hesaplanıyor.")

        # --- B. TUREV ---
        elif isinstance(expr, Derivative):
            var = expr.variable_count[0][0]
            func = expr.expr
            converter = DerivativeConverter()
            return converter.derive(func, var)

        # --- C. ISLEM (ADD, MUL, POW) ---
        elif isinstance(expr, (Add, Mul, Pow)):
            # Özyinelemeli olarak alt parçaları çöz
            substeps = []
            has_calculus = False

            for arg in expr.args:
                child_step = self._traverse_expr(arg)
                # Sadece içinde türev/integral geçenleri alt adım yap
                if child_step.type in ["Integral", "Derivative", "Combination"]:
                    substeps.append(child_step)
                    has_calculus = True

            final_res = expr.doit()

            if has_calculus:
                return MathStep(
                    "Combination", "Operation",
                    latex(expr), latex(final_res),
                    "İfade içindeki işlemler çözülüyor:",
                    substeps
                )
            else:
                return MathStep("Simplification", "Basic", latex(expr), latex(final_res), "", [])

        # --- D. TEMEL ELEMAN (x, 5, sin(x)) ---
        return MathStep("Atom", "Identity", latex(expr), latex(expr), "", [])


if __name__ == "__main__":
    solver = MathSolver()
    # Test
    res = solver.solve(r"\int 5x^4 dx")
    print(res.output_latex)  # x^5 olmalı