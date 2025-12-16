import json
from dataclasses import dataclass, field
from typing import List, Optional, Any

from sympy import (
    Symbol, latex, Integral, Derivative, Add, Mul, Pow, 
    diff, Basic, sympify
)
from latex2sympy2_extended import latex2sympy

# SADECE manualintegrate.py icinde kesin var olan temel kurallari import ediyoruz
from sympy.integrals.manualintegrate import (
    manualintegrate, integral_steps, AtomicRule,
    AddRule, ConstantTimesRule, ConstantRule, PowerRule, NestedPowRule, URule,
    PartsRule, AlternativeRule, RewriteRule, PiecewiseRule, HyperbolicRule,
    TrigRule, ExpRule, ReciprocalRule, DontKnowRule
)

# --- 1. VERI MODELI ---
@dataclass
class MathStep:
    type: str             # "Integral", "Derivative", "Simplification"
    rule: str             # kural Adi (Orn: PowerRule)
    input_latex: str      # islem oncesi ifade
    output_latex: str     # islem sonrasi ifade
    description: str      # aciklama
    substeps: List['MathStep'] = field(default_factory=list)    # alt adimlar

    def to_dict(self):
        return {
            "type": self.type,
            "rule": self.rule,
            "input": self.input_latex,
            "output": self.output_latex,
            "description": self.description,
            "substeps": [s.to_dict() for s in self.substeps]
        }


# --- 1.5. TUREV KURAL SINIFLARI (MANUEL) ---
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

# --- 2. TUREV VE INTEGRAL KURAL CEVIRICILERI ---

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
            step.description = "Sabit sayinin turevi 0'dir."
            step.output_latex = "0"
            
        elif isinstance(rule, IdentityDiffRule):
            step.rule = "IdentityRule"
            step.description = "Degiskenin kendisine gore turevi 1'dir."
            step.output_latex = "1"
            
        elif isinstance(rule, AddDiffRule):
            step.rule = "SumRule"
            step.description = "Toplam Turevi: Her terimin ayri ayri turevi alinir."
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
            step.description = "Us Kurali: Us basa carpan olarak gelir, us bir azaltilir."
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
            step.description = "Carpim Kurali: Birincinin turevi x ikinci + ikincinin turevi x birinci"
            
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
            step.description = "Zincir Kurali: Dis fonksiyonun turevi (ic aynen) x ic fonksiyonun turevi"
            
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
             step.description = f"{rule.func_name} fonksiyonunun turevi."
             res = diff(rule.expr, rule.variable)
             step.output_latex = latex(res)
             
        elif isinstance(rule, ExpDiffRule):
             step.rule = "ExpRule"
             step.description = "Ustel fonksiyon turevi."
             res = diff(rule.expr, rule.variable)
             step.output_latex = latex(res)
        
        # Son hesaplanan latex'i sympy ile basitlestirebilirdik ama
        # adim adim gostermek istedigimiz icin ham birakmak daha egitici olabilir.
        
        return step

# --- 2. INTEGRAL KURAL CEVIRICISI (SAFE CONVERTER) ---
class IntegralConverter:
    """
    sympy kural nesnelerini MathStep nesnesine cevirir.
    hata cikarabilecek karmasik rule importlari yerine yapisal kontrol yapar.
    """
    
    def convert(self, rule: Any, current_expr: Basic = None) -> MathStep:
        # kural ismini al (Orn: 'AddRule' -> 'Add')
        rule_name = rule.__class__.__name__.replace("Rule", "")
        
        # Varsayilan (bos) adim olustur
        step = MathStep(
            type="Integral",
            rule=rule_name,
            input_latex = fr"\int {latex(rule.integrand)} d{latex(rule.variable)}",
            output_latex="",
            description=f"{rule_name} uygulaniyor.",
            substeps=[]
        )

        # --- A. YAPISAL KURALLAR (RECURSIVE) ---
        
        if isinstance(rule, AddRule):
            step.description = "Toplam Kurali: Terimlerin ayri ayri integrali alinir."
            
            # AddRule -> substeps (List)
            for iter, sub in enumerate(rule.substeps):
                
                # latex out kısmını doldurur
                step.output_latex = step.output_latex + Fr"\int {latex(sub.integrand)} d{sub.variable} "
                
                # her integral teriminin arasına '+' koyar
                if (iter != len(rule.substeps)-1):
                    step.output_latex += "+ "
                
                step.substeps.append(self.convert(sub))



        # TODO bunu tam anlamadım
        elif isinstance(rule, PiecewiseRule):
            step.description = "Parcali Fonksiyon: Her aralik ayri hesaplanir."
            # PiecewiseRule -> subfunctions (List)
            if hasattr(rule, 'subfunctions'):
                for sub in rule.subfunctions:
                    ctx = getattr(sub, 'context', None)
                    step.substeps.append(self.convert(sub, ctx))



        elif isinstance(rule, PartsRule):
            step.description = "Kismi Integrasyon (Integration by Parts): Kurala gore u ve dv secilir ve formul uygulanir"
            # PartsRule -> u, dv, second_step (Rule)
            u_latex = latex(rule.u)
            dv_latex = latex(rule.dv)
            step.output_latex = f"u={u_latex}, \\quad dv={dv_latex},\\quad uv-\\int vdu"
            
            # u dv yi bulduktan sonra dv den vye dönüşüm için integral
            if rule.v_step:
                 v_step_obj = self.convert(rule.v_step, None)
                 v_step_obj.description = f"dv'den v'yi bulma adimi: " + v_step_obj.description
                 step.substeps.append(v_step_obj)
            
            # kalan integral adimi (second_step)
            if rule.second_step:
                # second_step bir Rule nesnesidir, bunu recursive cevir
                sub_step_obj = self.convert(rule.second_step, None)
                step.substeps.append(sub_step_obj)
                
                

        # TODO u nun turevini de yazdirma isi
        elif isinstance(rule, URule):
            step.description = "Degisken Degistirme (U-Substitution)"
            u_latex = latex(rule.u_func)
            step.output_latex = f"u = {u_latex}, \\quad \\int {latex(rule.substep.integrand)} du"
            
            # URule -> substep 
            if rule.substep:
                step.substeps.append(self.convert(rule.substep, None))



        elif isinstance(rule, RewriteRule):
            step.description = "Yeniden Yazma (Rewrite)"
            # RewriteRule -> rewritten (Expr), substep (Rule)
            rewritten_latex = latex(rule.rewritten)
            step.output_latex = f"\\rightarrow {rewritten_latex}"
            if rule.substep:
                step.substeps.append(self.convert(rule.substep, getattr(rule, 'rewritten', None)))
        

        elif isinstance(rule, AlternativeRule):
            step.description = "Alternatif Yontemler"
            # AlternativeRule -> alternatives (List)
            if rule.alternatives:
                # Sadece ilkini al
                first_alt = rule.alternatives[0]
                step.substeps.append(self.convert(first_alt, None))
        


        elif isinstance(rule, ConstantTimesRule):
            step.description = "Sabit Katsayi Kurali: Sabit sayi integralin disina alinir."
            const = getattr(rule, 'constant', '')
            step.output_latex = fr"{const} \int {latex(rule.substep.integrand)}d{rule.variable}"
            if rule.substep:
                step.substeps.append(self.convert(rule.substep, None))

        # --- B. TEMEL KURALLAR ---
        


        elif isinstance(rule, AtomicRule):

            if isinstance(rule, ConstantRule): step.description = "Sabit Kurali: Sabit sayinin integrali, sayi ile integral degiskeninin carpimidir."
            elif isinstance(rule, PowerRule): step.description = "Us Kurali (Power Rule): Us bir arttirilir, ifade olusan yeni usse bolunur."
            elif isinstance(rule, ReciprocalRule): step.description = "Sonucu ln(x) gelen integral"
            elif isinstance(rule, ExpRule): step.description = "Ustel Fonksiyon Integrali: Ustel fonksiyon aynen yazilir, ln(taban)'a bolunur (a^x/ln(a))"
            elif isinstance(rule, TrigRule): step.description = "Trigonometrik Fonksiyon Integrali"
            elif isinstance(rule, NestedPowRule): step.description = "Ussun ussu kurali"
            elif isinstance(rule, HyperbolicRule): step.description = "Hiperobolik Fonksiyon Integrali"

            # eger bulunan temel kural implement edilmemisse diye fallback
            else: step.description = f"Kural: {rule_name}"

            step.output_latex = latex(rule.eval())


        # --- C. FALLBACK (DIGER TUM DURUMLAR) ---
        
        elif isinstance(rule, DontKnowRule):
            step.description = "Otomatik cozum bulunamadi."

        else:
            # TrigSubstitutionRule vb. import edilmeyen ama donebilen kurallar icin
            step.description = f"Kural: {rule_name}"
            # Eger substep varsa onlar da eklenir
            if hasattr(rule, 'substep') and rule.substep:
                 step.substeps.append(self.convert(rule.substep, None))

        return step



# --- 3. ANA COZUCU SINIF ---
class MathSolver:
    def __init__(self):
        self.converter = IntegralConverter()

    def solve(self, latex_input: str) -> dict:
        try:
            # 1. Parsing
            expr = latex2sympy(latex_input)
            
            # 2. Agaci Gezme
            result_step = self._traverse_expr(expr)
            
            # 3. Nihai Sonuc
            final_res = expr.doit()
            
            return {
                "input_latex": latex_input,
                "result_latex": latex(final_res),
                "steps": result_step.to_dict()
            }
        except Exception as e:
            return {"error": str(e), "details": "Parse veya cozum hatasi"}

    def _traverse_expr(self, expr: Basic) -> MathStep:
        """
        Ifade agacini gez:
            Integral ise SymPy manualintegrate kullan.
            Turev ise manuel turev kullan.
            Islem ise alt dallara in.
        """
        
        # --- DURUM 1: INTEGRAL ---
        if isinstance(expr, Integral):
            function = expr.function
            limits = expr.limits
            var = limits[0][0] # Ilk degisken (dx)
            
            # Adimlari bul
            try:
                rule_tree = integral_steps(function, var)
                step = self.converter.convert(rule_tree, function)
                
                # Belirli integral mi? (Sinirlar var mi?)
                if len(limits[0]) == 3:
                    lower, upper = limits[0][1], limits[0][2]
                    # Manuel integral sonucunu hesapla
                    indef_result = manualintegrate(function, var)
                    
                    # Sinirlari koyma adimi
                    val_upper = indef_result.subs(var, upper)
                    val_lower = indef_result.subs(var, lower)
                    
                    bound_step = MathStep(
                        "DefiniteIntegral", "Limits",
                        latex(indef_result),
                        latex(val_upper - val_lower),
                        f"Sinirlar yerine konulur: F({upper}) - F({lower})"
                    )
                    step.substeps.append(bound_step)
                    
                return step
            except Exception as e:
                return MathStep("Error", "IntegralError", latex(expr), "", str(e))

        # --- DURUM 2: TUREV ---
        elif isinstance(expr, Derivative):
            # Turev icin manuel adimlayici
            var = expr.variable_count[0][0]
            func = expr.expr
            
            converter = DerivativeConverter()
            try:
                step = converter.derive(func, var)
                # Nihai sonuc kontrolu
                final_res = diff(func, var)
                # Eger adim adim sonuc ile sympy sonucu cok farkliysa buraya bir check koyulabilir
                # ama simdilik guveniyoruz.
                return step
            except Exception as e:
                # Fallback
                res = diff(func, var)
                return MathStep(
                    "Derivative", "Result", 
                    latex(expr), latex(res), 
                    f"{var} degiskenine gore turev alindi. (Detayli adim hatasi: {e})"
                )

        # --- DURUM 3: DORT ISLEM (Recursive) ---
        elif isinstance(expr, (Add, Mul, Pow)):
            substeps = []
            has_action = False
            
            for arg in expr.args:
                child_step = self._traverse_expr(arg)
                # Eger alt dalda bir integral/turev islemi varsa alt adim olarak ekle
                if child_step.type in ["Integral", "Derivative", "Combination"]:
                    substeps.append(child_step)
                    has_action = True
            
            if has_action:
                return MathStep(
                    "Combination", "Operation", 
                    latex(expr), latex(expr.doit()), 
                    "Islem icindeki terimler cozuluyor.", 
                    substeps
                )

        # --- DURUM 4: PASIF TERIM (x, 5, sin(x)) ---
        return MathStep("Atom", "Identity", latex(expr), "", "", [])

# --- TEST ---
if __name__ == "__main__":
    solver = MathSolver()
    
    """
    print("--- Test 1: Piecewise / Abs ---")
    res1 = solver.solve(r"\int |x| dx")
    print(json.dumps(res1, indent=2))

    # SENARYO 2: Kismi Integrasyon (PartsRule)
    print("\n--- Test 2: Parts ---")
    res2 = solver.solve(r"\int x e^x dx")
    print(json.dumps(res2, indent=2))
    
    
    # SENARYO 3: Belirli Integral + Sabit
    print("\n--- Test 3: Definite Integral ---")
    res3 = solver.solve(r"\int_0^1 x^2 dx + 5")
    print(json.dumps(res3, indent=2))
    
    
    
    
    print("\n--- Test 4 ---")
    res3 = solver.solve(r"\int 2x \cos(x^2)dx")
    print(json.dumps(res3, indent=2))
    

    print("\n--- Test 5 ---")
    res3 = solver.solve(r"5\int_{6}^{7}8x^3+5x^2+4x-2dx")
    print(json.dumps(res3, indent=2))
    
    

    print("\n--- Test 6 ---")
    res3 = solver.solve(r"3+5-5/2")
    print(json.dumps(res3, indent=2))
    
    
    # Test 1: Nested Chain Rule
    # d/dx(sin(cos(x^2)))
    # Outer: sin(u) -> cos(u) * u'
    # Inner u = cos(v) -> -sin(v) * v'
    # Inner v = x^2 -> 2x

    print("\n--- Test ---")
    res3 = solver.solve(r"\frac{d}{dx}(\sin(\cos(x^2)))")
    print(json.dumps(res3, indent=2))
    
    print("\n--- Test ---")
    res3 = solver.solve(r"\frac{d}{dx}(x^2 e^{3x})")
    print(json.dumps(res3, indent=2))

    print("\n--- Test ---")
    res3 = solver.solve(r"\frac{d}{dx}(\frac{x}{x+1})")
    print(json.dumps(res3, indent=2))

    print("\n--- Test ---")
    res3 = solver.solve(r"\frac{d}{dx}(\ln(x^2 + 1))")
    print(json.dumps(res3, indent=2))

    print("\n--- Test ---")
    res3 = solver.solve(r"\frac{d}{dx}((3x^2 + 1)^5)")
    print(json.dumps(res3, indent=2))

    print("\n--- Test ---")
    res3 = solver.solve(r"\frac{d}{dx}(\tan(\sin(x)))")
    print(json.dumps(res3, indent=2))


    print("\n--- Test ---")
    res3 = solver.solve(r"\int_{}^{}\frac{2}{x^2+4}dx")
    print(json.dumps(res3, indent=2))
    """