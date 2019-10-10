"""
printing modules in numpy, scipy and pandas
"""

import pkgutil as pu
import pydoc
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl

print("NumPy version", np.__version__)
print("SciPy version", sp.__version__)
print("pandas version", pd.__version__)
print("Matplotlib version", mpl.__version__)

def clean(astr):
   s = astr
   # remove multiple spaces
   s = ' '.join(s.split())
   s = s.replace('=', '')
   return s

def print_desc(prefix, pkg_path):
    """
    pu.iter_modules : path의 모든 서브 모듈에 대한 ModuleInfo
    pydoc : 자동으로 설명서를 생성
    """
    for pkg in pu.iter_modules(path=pkg_path):
        name = prefix + "." + pkg[1]

        if pkg[2] == True:
            try:
                docstr = pydoc.plain(pydoc.render_doc(name))
                docstr = clean(docstr)
                start = docstr.find("DESCRIPTION")
                docstr = docstr[start: start + 140]
                print(name, docstr)
            except:
                continue

print("\n")
print_desc("numpy", np.__path__)
print("\n")
print_desc("scipy", sp.__path__)
print("\n")
print_desc("pandas", pd.__path__)
print("\n")
print_desc("matplotlib", mpl.__path__)



