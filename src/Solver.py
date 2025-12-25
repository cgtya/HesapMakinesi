from dataclasses import dataclass, field
from typing import List, Any
import re
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

# turev kurallari
@dataclass
class DerivRule:
    expr: Any
    variable: Any

@dataclass
class ConstantDiffRule(DerivRule):
    pass

@dataclass
class IdentityDiffRule(DerivRule):
    pass

@dataclass
class PowerDiffRule(DerivRule):
    base: Any
    exp: Any
    
@dataclass
class AddDiffRule(DerivRule):
    terms: List[DerivRule]

@dataclass
class ProductDiffRule(DerivRule):
    terms: List[Any]  # f, g
    derivs: List[DerivRule] # f', g'

@dataclass
class ChainDiffRule(DerivRule):
    outer_func: Any
    inner_func: Any
    inner_deriv: DerivRule

@dataclass
class TrigDiffRule(DerivRule):
    func_name: str
    arg: Any
    
@dataclass
class ExpDiffRule(DerivRule):
    base: Any
    arg: Any

@dataclass
class LogDiffRule(DerivRule):
    arg: Any
    base: Any

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


# --- 2. KURAL CEVIRICILERI ---
class DerivativeConverter:

    """
    Verilen ifadeyi manuel turev kurallarina gore analiz eder ve MathStep olusturur.
    sympyda 'manualintegrate' gibi bir metot olmadigi icin bunu biz yapiyoruz.
    """
    
    def derive(self, expr: Basic, var: Symbol) -> MathStep:
        rule = self._find_rule(expr, var)
        return self._convert_rule_to_step(rule)

    def _find_rule(self, expr: Basic, var: Symbol) -> DerivRule:
        # 1. Sabit (Constant)
        if not expr.has(var):
            return ConstantDiffRule(expr, var)
            
        # 1.5 Degiskenin kendisi (Identity)
        if expr == var:
            return IdentityDiffRule(expr, var)
        
        # 2. Toplama (Add)
        if isinstance(expr, Add):
            sub_rules = [self._find_rule(arg, var) for arg in expr.args]
            return AddDiffRule(expr, var, sub_rules)
            
        # 3. Carpma (Mul) - Product Rule
        if isinstance(expr, Mul):
            # Basitlestirme: Ikili carpim gibi dusunelim veya zincirleme
            # Sympy Mul argumanlarini duzlestirir, ornek x*y*z -> (x, y*z) seklinde recursively cozebiliriz
            # Ya da hepsini listeye alip genel carpim kurali
            
            # Katsayi varsa ayiralim mi? Ornek 5*x^2 -> ConstantTimes aslinda Add icinde handle edilebilir ama Mul buraya duser.
            # Eger argumanlarin sadece biri degiskene bagliysa -> ConstantMultiple
            dep_args = [arg for arg in expr.args if arg.has(var)]
            if len(dep_args) == 0:
                return ConstantDiffRule(expr, var)
            elif len(dep_args) == 1:
                # Constant multiple: c * f(x)
                # Bunu da ProductRule gibi gosterebiliriz veya ayri constant rule
                # Tutarlilik icin ProductRule dondurelim, 0 turev gosterilir.
                pass 
            
            # Genel Carpim Kurali
            # f * g * h ...
            return ProductDiffRule(expr, var, expr.args, [self._find_rule(arg, var) for arg in expr.args])

        # 4. Us (Pow)
        if isinstance(expr, Pow):
            base, exp = expr.args
            
            # x^n (n sabit) -> Power Rule
            if base == var and not exp.has(var):
                return PowerDiffRule(expr, var, base, exp)
                
            # u(x)^n (Zincir kurali ile Power Rule)
            if not exp.has(var): # base degiskenli
                # Chain Rule ozel durumu: PowerChain
                # Outer: u^n, Inner: u
                inner_rule = self._find_rule(base, var)
                return ChainDiffRule(expr, var, "Pow", base, inner_rule)

            # a^x (Ustel)
            if not base.has(var) and exp.has(var):
                return ExpDiffRule(expr, var, base, exp)
                
            # x^x (Logaritmik turev gerekir, simdilik Chain gibi bakalim veya Exp(x*ln(x)))
            # Sympy buna genellikle exp(b * ln(a)) donusumu yapar.
            # Biz direkt ChainRule dondurelim.
            pass

        # 5. Fonksiyonlar (Trig, Exp, Log vb.)
        # func(u(x)) -> Zincir Kurali
        if expr.is_Function:
            func_name = expr.func.__name__
            arg = expr.args[0]
            
            # Eger arguman sadece 'var' ise -> Basic Rule
            if arg == var:
                if func_name in ['sin', 'cos', 'tan', 'cot', 'sec', 'csc']:
                    return TrigDiffRule(expr, var, func_name, arg)
                if func_name == 'exp':
                    return ExpDiffRule(expr, var, Symbol('e'), arg)
                if func_name == 'log' or func_name == 'ln':
                     return LogDiffRule(expr, var, arg, Symbol('e'))
            
            # Degilse Chain Rule
            inner_rule = self._find_rule(arg, var)
            return ChainDiffRule(expr, var, expr.func, arg, inner_rule)

        # Fallback
        return ConstantDiffRule(expr, var) # Aslinda bilinmeyen, ama 0 donsun simdilik

    def _convert_rule_to_step(self, rule: DerivRule) -> MathStep:
        step = MathStep("Derivative", "Unknown", latex(rule.expr), "", "", [])
        var = rule.variable
        
        if isinstance(rule, ConstantDiffRule):
            step.rule = "ConstantRule"
            step.description = "Sabit sayının türevi 0'dır."
            step.output_latex = "0"
            
        elif isinstance(rule, IdentityDiffRule):
            step.rule = "IdentityRule"
            step.description = "Değişkenin kendisine göre türevi 1'dir."
            step.output_latex = "1"
            
        elif isinstance(rule, AddDiffRule):
            step.rule = "SumRule"
            step.description = "Toplam Türevi: Her terimin ayrı ayrı türevi alınır."
            terms_out = []
            for r in rule.terms:
                s = self._convert_rule_to_step(r)
                step.substeps.append(s)
                # Sadece 0 olmayanlari ekle (veya tek terim ise ekle)
                if s.output_latex != "0":
                    terms_out.append(s.output_latex)
            
            if not terms_out:
                step.output_latex = "0"
            else:
                step.output_latex = " + ".join(terms_out).replace("+ -", "- ")
            
        elif isinstance(rule, PowerDiffRule):
            step.rule = "PowerRule"
            n = rule.exp
            # n*x^(n-1)
            step.description = "Üs Kuralı: Üs başa çarpan olarak gelir, üs bir azaltılır."
            from sympy import Number
            new_exp = n - 1
            if new_exp == 0:
                 step.output_latex = f"{latex(n)}"
            elif new_exp == 1:
                 step.output_latex = f"{latex(n)}{latex(rule.base)}"
            else:
                 step.output_latex = f"{latex(n)}{latex(rule.base)}^{{{latex(new_exp)}}}"

        elif isinstance(rule, ProductDiffRule):
            step.rule = "ProductRule"
            step.description = "Çarpım Kuralı: Birincinin türevi çarpı ikinci + ikincinin türevi çarpı birinci."
            
            total_sum = []
            terms = rule.terms
            derivs = rule.derivs # sub steps
            
            for i in range(len(terms)):
                # Term i'nin turevi
                d_step = self._convert_rule_to_step(derivs[i])
                step.substeps.append(d_step)
                
                # Eger turev 0 ise bu terim duser
                if d_step.output_latex == "0":
                    continue
                    
                # Formula parcasi: f' * (digerleri)
                others = [latex(t) for k, t in enumerate(terms) if k != i]
                
                part = ""
                # Eger turev 1 ise ve baska carpanlar varsa direk diger carpanlari yaz
                if d_step.output_latex == "1" and others:
                    part = " \\cdot ".join(others)
                else:
                    part = d_step.output_latex
                    if others:
                        part += " \\cdot " + " \\cdot ".join(others)
                        
                total_sum.append(part)
            
            if not total_sum:
                step.output_latex = "0"
            else:
                step.output_latex = " + ".join(total_sum)
            
        elif isinstance(rule, ChainDiffRule):
            step.rule = "ChainRule"
            step.description = "Zincir Kuralı: Dış fonksiyonun türevi (iç aynen kalır) çarpı iç fonksiyonun türevi."
            
            inner_step = self._convert_rule_to_step(rule.inner_deriv)
            step.substeps.append(inner_step)
            
            # Dis turevi hesapla (Sympy ile)
            # f(u) -> f'(u)
            # Burada sembolik bir trick yapacagiz: diff(f(dummy)).subs(dummy, u)
            from sympy import Dummy, Function
            u_dummy = Dummy('u')
            
            if isinstance(rule.outer_func, str) and rule.outer_func == "Pow":
                # u^n -> n*u^(n-1)
                base = rule.inner_func # Original inner
                 # Aslinda PowerDiffRule daki exp lazim burada ama ChainDiffRule'a tasimadik.
                 # HACK: expr'den cekelim
                if isinstance(rule.expr, Pow):
                    exp = rule.expr.args[1]
                    outer_deriv = exp * Pow(u_dummy, exp-1)
                else:
                    outer_deriv = sympify(1) # Fail safe
            else:
                # sin(u), exp(u) vs
                func_cls = rule.outer_func # sin class
                outer_deriv = diff(func_cls(u_dummy), u_dummy)
            
            # Substitute back
            outer_res = outer_deriv.subs(u_dummy, rule.inner_func)
            
            if inner_step.output_latex == "1":
                 step.output_latex = f"{latex(outer_res)}"
            else:
                 step.output_latex = f"({latex(outer_res)}) \\cdot ({inner_step.output_latex})"
            
        elif isinstance(rule, TrigDiffRule):
             step.rule = "TrigRule"
             step.description = f"{rule.func_name} fonksiyonunun türevi."
             res = diff(rule.expr, rule.variable)
             step.output_latex = latex(res)
             
        elif isinstance(rule, ExpDiffRule):
             step.rule = "ExpRule"
             step.description = "Üstel fonksiyon türevi."
             res = diff(rule.expr, rule.variable)
             step.output_latex = latex(res)
        
        # Son hesaplanan latex'i sympy ile basitlestirebilirdik ama
        # adim adim gostermek istedigimiz icin ham birakmak daha egitici olabilir.
        
        return step
    
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

        elif isinstance(rule, AlternativeRule):
            step.description = "Alternatif Yontemler"
            # AlternativeRule -> alternatives (List)
            if rule.alternatives:
                # Sadece ilkini al
                first_alt = rule.alternatives[0]
                step.substeps.append(self.convert(first_alt, None))

        elif isinstance(rule, PiecewiseRule):
            step.description = "Parçalı Fonksiyon: Her aralık ayrı hesaplanır."
            # PiecewiseRule -> subfunctions (List)
            if hasattr(rule, 'subfunctions'):
                for sub in rule.subfunctions:
                    ctx = getattr(sub, 'context', None)
                    step.substeps.append(self.convert(sub, ctx))
            try:
                step.output_latex = latex(rule.eval())
            except:
                pass
        
        elif isinstance(rule, DontKnowRule):
            step.description = "Otomatik çözüm bulunamadı."


        # --- B. TEMEL KURALLAR (ATOMİK) ---
        elif isinstance(rule, AtomicRule):
            # Basit kurallarin sonucunu direkt hesapla
            step.output_latex = latex(rule.eval())

            if isinstance(rule, ConstantRule):
                step.description = "Sabit Kuralı: Sabit sayının yanına x eklenir."
            elif isinstance(rule, PowerRule):
                step.description = "Üs Kuralı: Üs bir arttırılır, ifade yeni üsse bölünür."
            elif isinstance(rule, NestedPowRule): 
                step.description = "Ussun ussu kurali"

            elif isinstance(rule, ExpRule):
                step.description = "Üstel Fonksiyon Kuralı"
            elif isinstance(rule, TrigRule):
                step.description = "Trigonometrik İntegral Kuralı"
            elif isinstance(rule, ReciprocalRule):
                step.description = "1/x İntegrali (ln|x|)"
            elif isinstance(rule, HyperbolicRule):
                step.description = "Hiperbolik Fonksiyon İntegrali"
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



# --- 4. ANA COZUCU SINIF ---
class MathSolver:
    def __init__(self):
        self.converter = IntegralConverter()

    def solve(self, latex_input: str) -> MathStep:
        try:
            # 0. Preprocessing: Fragmented Functions (c o s -> \cos)
            latex_input = self.fix_fragmented_latex(latex_input)
            
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
                description=f"Hata oluştu: {str(e)}",
                substeps=[]
            )

    def fix_fragmented_latex(self, latex_input: str) -> str:
        """
        parcalanmis fonksiyon isimlerini duzeltir.
        ornek: "s i n" -> "\\sin", "c*o*s" -> "\\cos"
        """
        # duzeltilecek fonksiyonlar
        funcs = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'ln', 'log', 'exp', 'arcsin', 'arccos', 'arctan']
        
        # islem yapilan string
        text = latex_input
        
        for func in funcs:
            # harf harf pattern olustur
            # her harf arasinda bosluk, * veya \cdot olabilir
            chars = list(func)
            pattern_parts = []
            for i, char in enumerate(chars):
                pattern_parts.append(re.escape(char))
                if i < len(chars) - 1:
                    # ayirici: bosluk, *, \cdot
                    pattern_parts.append(r'(?:\s|\*|\\cdot)*')
            
            pattern = "".join(pattern_parts)
            
            # regex ile degistir
            # eslesme varsa \func seklinde degistir
            regex = re.compile(pattern, re.IGNORECASE)
            
            text = regex.sub(lambda m: f"\\{func}", text)
            
        # cift backslash olusursa temizle (\\sin -> \sin)
        text = text.replace(r"\\", "\\")
        
        return text

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
                return MathStep("Simplification", "Basic", latex(expr), latex(final_res), "Sadeleştirme", [])

        # --- D. TEMEL ELEMAN (x, 5, sin(x)) ---
        return MathStep("Atom", "Identity", latex(expr), latex(expr), "", [])


if __name__ == "__main__":
    solver = MathSolver()
    # Test
    res = solver.solve(r"\int 5x^4 dx")
    print(res.output_latex)  # x^5 olmalı