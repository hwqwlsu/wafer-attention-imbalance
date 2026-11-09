import os
# -*- coding: utf-8 -*-
import re, sys

ROOT = os.environ.get(
    "WAFER_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
p = (sys.argv[1] if len(sys.argv) > 1
     else os.path.join(ROOT, "manuscript", "manuscript.tex"))
t = open(p, encoding="utf-8").read()
print("begin/end:", t.count("\\begin{"), t.count("\\end{"))
if t.count("\\begin{") != t.count("\\end{"):
    from collections import Counter
    b = Counter(re.findall(r"\\begin\{([^}]+)\}", t))
    e = Counter(re.findall(r"\\end\{([^}]+)\}", t))
    print("unbalanced envs:", {k: (b[k], e[k]) for k in set(b) | set(e) if b[k] != e[k]})
print("brace balance:", t.count("{") - t.count("}"))
labels = set(re.findall(r"\\label\{([^}]+)\}", t))
refs = set(re.findall(r"\\ref\{([^}]+)\}", t))
print("undefined refs:", sorted(refs - labels))
body = t.split("\\section{Introduction}", 1)[1]
print("body words:", len(re.sub(r"\\[a-zA-Z]+", " ", body).split()))
print("citations used:", len(set(re.findall(r"\\cite\{([^}]+)\}", t))))
