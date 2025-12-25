import os
import sys

from cx_Freeze import Executable, setup

sys.setrecursionlimit(5000)


build_options = {
    "packages": ["torch", "torchvision"],
    "excludes": [],
    "include_files": [("ocr_src", os.path.join("lib", "ocr_src"))],
    "optimize": 2,
}

base = "gui"

executables = [
    Executable(
        os.path.join("src", "gui_test.py"), base=base, target_name="HesapMakinesi"
    )
]

setup(
    name="HesapMakinesi",
    version="1.0",
    description="Verilen görseldeki matematiksel ifadeyi tanımlayıp adım adım çözümünü sunabilen hesap makinesi uygulaması.",
    options={"build_exe": build_options},
    executables=executables,
)
