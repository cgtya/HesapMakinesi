import sys
from cx_Freeze import setup, Executable
sys.setrecursionlimit(5000)

# Dependencies are automatically detected, but it might need
# fine tuning.
build_options = {
    'packages': ['torch','torchvision'], 
    'excludes': [],
    'include_files': [('ocr_src')],
}

base = 'gui'

executables = [
    Executable('src/gui_test.py', base=base, target_name="HesapMakinesi")
]

setup(name='HesapMakinesi',
      version = '1.0',
      description = 'Verilen görseldeki matematiksel ifadeyi tanımlayıp adım adım çözümünü sunabilen hesap makinesi uygulaması.',
      options = {'build_exe': build_options},
      executables = executables)
