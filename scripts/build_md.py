# -*- coding: utf-8 -*-
"""Build manuscript.md from manuscript.tex (single source of truth).
Mirrors build_docx.py logic: fig/tab numbering, cite map, table parsing."""
import os, re

ROOT = os.environ.get(
    "WAFER_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.join(ROOT, "manuscript")
TABS = os.path.join(BASE, "tables")
OUT = os.path.join(BASE, "manuscript.md")

src = open(os.path.join(BASE, "manuscript.tex"), encoding="utf-8").read()
src = re.sub(r"(?<!\\)%.*$", "", src, flags=re.M)          # strip comments
src = src.replace("\r\n", "\n")

# ---------- helpers ----------
def balanced(s, i):
    """Return (inner, end) of {...} group whose '{' is at index i."""
    d, j = 1, i + 1
    while j < len(s) and d:
        if s[j] == "{": d += 1
        elif s[j] == "}": d -= 1
        j += 1
    return s[i + 1:j - 1], j

def take_cmd(s, cmd):
    m = re.search(r"\\" + cmd + r"\{", s)
    if not m: return ""
    return balanced(s, m.end() - 1)

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]

# ---------- bibliography ----------
bib = re.search(r"\\begin\{thebibliography\}.*?\n(.*?)\\end\{thebibliography\}",
                src, re.S).group(1)
parts = re.split(r"\\bibitem\{([^}]*)\}", bib)
CITE, BIB = {}, []
for i in range(1, len(parts), 2):
    key, txt = parts[i], parts[i + 1]
    CITE[key] = (i + 1) // 2
    BIB.append(txt.strip())

# ---------- title / author / abstract ----------
title = take_cmd(src, "title")[0].replace("\\\\", " ")
author = take_cmd(src, "author")[0]
author = re.sub(r"\\thanks\{", "\x00", author)
if "\x00" in author:
    pre, rest = author.split("\x00", 1)
    _, j = balanced(rest, 0)
    author = pre + rest[j:]
author = re.sub(r"\\IEEEauthorblock[na]\{[^}]*\}", "", author)
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S).group(1).strip()
body = src.split(r"\end{abstract}", 1)[1].split(r"\begin{thebibliography}")[0]

# ---------- pre-scan figure order ----------
FIGNO, FIGCAP, FIGIMG = {}, {}, {}
for blk in re.findall(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", body, re.S):
    lab = re.search(r"\\label\{(fig:[^}]*)\}", blk)
    img = re.search(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", blk)
    cap = re.search(r"\\caption\{", blk)
    if lab and img:
        FIGNO[lab.group(1)] = len(FIGNO) + 1
        FIGIMG[lab.group(1)] = img.group(1).replace(".pdf", ".png")
        FIGCAP[lab.group(1)] = balanced(blk, cap.end() - 1)[0] if cap else ""

# ---------- pre-scan table order ----------
TABNO = {}
for tf in re.findall(r"\\input\{(tables/[^}]+)\}", body):
    if tf not in TABNO:
        TABNO[tf] = ROMAN[len(TABNO)]

# ---------- pre-scan section numbers ----------
SECNUM = {}
sec, sub = 0, 0
for m in re.finditer(r"\\(sub)?section\{", body):
    t, j = balanced(body, m.end() - 1)
    lab = re.match(r"\s*\\label\{(sec:[^}]*)\}", body[j:j + 60])
    if m.group(1):
        sub += 1; num = f"{sec}.{sub}"
    else:
        sec += 1; sub = 0; num = str(sec)
    if lab: SECNUM[lab.group(1)] = num

# section labels carry no text; drop them now that SECNUM is built
body = re.sub(r"\\label\{sec:[^}]*\}", "", body)

# label -> roman numeral, read from each table file
TABLABEL = {}
for tf, rn in TABNO.items():
    s = open(os.path.join(BASE, tf + ".tex"), encoding="utf-8").read()
    m = re.search(r"\\label\{(tab:[^}]*)\}", s)
    if m: TABLABEL[m.group(1)] = rn

# ---------- text cleaning ----------
def clean(t):
    t = re.sub(r"\\ref\{(fig:[^}]*)\}", lambda m: str(FIGNO.get(m.group(1), "?")), t)
    t = re.sub(r"\\ref\{(tab:[^}]*)\}", lambda m: TABLABEL.get(m.group(1), "?"), t)
    t = re.sub(r"\\ref\{(sec:[^}]*)\}", lambda m: SECNUM.get(m.group(1), "?"), t)
    t = re.sub(r"\\cite\{([^}]*)\}",
               lambda m: "[" + ", ".join(str(CITE.get(k.strip(), "?")) for k in m.group(1).split(",")) + "]", t)
    t = re.sub(r"\\emph\{", "\x01", t)
    t = re.sub(r"\\textbf\{", "\x02", t)
    for mark, rep in [("\x01", "*"), ("\x02", "**")]:
        while mark in t:
            i = t.index(mark)
            inner, j = balanced(t, i)
            t = t[:i] + rep + inner + rep + t[j:]
    t = re.sub(r"\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
               r"(\1)/(\2)", t)
    t = re.sub(r"\\sum_?\{?([a-zA-Z])?\}?", lambda m: "Σ" + ("_" + m.group(1) if m.group(1) else ""), t)
    t = re.sub(r"\\text\{([^{}]*)\}", r"\1", t)
    t = t.replace(r"\;", " ").replace(r"\,", " ").replace(r"\qquad", "   ")
    t = re.sub(r"\\'([aeiouAEIOU])",
               lambda m: {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú",
                          "A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú"}[m.group(1)], t)
    t = re.sub(r"\\hat\{(.)\}", lambda m: m.group(1) + "̂", t)
    t = re.sub(r"\\(odot|oplus|sigma|rho|theta|cos|sin)(?![a-zA-Z])",
               lambda m: {"odot": "⊙", "oplus": "⊕", "sigma": "σ", "rho": "ρ",
                          "theta": "θ", "cos": "cos", "sin": "sin"}[m.group(1)], t)
    t = t.replace(r"\!", "")
    t = t.replace(r"\dagger", "†")
    t = t.replace("``", "\u201c").replace("''", "\u201d")
    t = t.replace("`", "\u2018").replace("'", "\u2019")
    t = t.replace("---", "\u2014").replace("--", "\u2013")
    t = t.replace("~", " ").replace("\\%", "%").replace("\\&", "&")
    t = t.replace(r"^{\circ}", "°").replace(r"^\circ", "°")
    t = re.sub(r"\\(pm|times|Delta|beta|alpha|leq|geq|infty|circ)(?![a-zA-Z])",
               lambda m: {"pm": "\u00b1", "times": "\u00d7", "Delta": "\u0394",
                          "beta": "\u03b2", "alpha": "\u03b1", "leq": "\u2264",
                          "geq": "\u2265", "infty": "\u221e", "circ": "\u00b0"}[m.group(1)], t)
    t = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^]]*\])?", "", t)     # drop remaining cmds
    t = t.replace("\\ ", " ").replace("\\\n", " ")          # explicit spacing cmds
    t = t.replace("{", "").replace("}", "")
    t = re.sub(r"\^(?=[*†])", "", t)                       # footnote markers
    t = re.sub(r"[ \t]+\n", "\n", t)
    return t

def flatten(t):
    return re.sub(r"\s*\n\s*", " ", clean(t)).strip()

# ---------- table -> markdown ----------
def parse_table(tf):
    s = open(os.path.join(BASE, tf + ".tex"), encoding="utf-8").read()
    s = re.sub(r"(?<!\\)%.*$", "", s, flags=re.M)
    cap = flatten(take_cmd(s, "caption")[0])
    m = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", s, re.S)
    blk = m.group(1)
    blk = re.sub(r"\\(toprule|bottomrule|addlinespace\[[^\]]*\]|addlinespace|small|centering)", "", blk)
    blk = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", blk)
    head, _, bod = blk.partition(r"\midrule")

    def rows(x):
        out = []
        for chunk in x.split(r"\\"):
            chunk = chunk.strip()
            if not chunk: continue
            cells = []
            for c in chunk.split("&"):
                c = c.strip()
                mm = re.match(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{", c)
                if mm:
                    inner, _ = balanced(c, mm.end() - 1)
                    cells.append(flatten(inner)); cells += [""] * (int(mm.group(1)) - 1)
                else:
                    cells.append(flatten(c))
            out.append(cells)
        return out

    hdr, bodyrows = rows(head), rows(bod)
    ncol = max(len(r) for r in hdr + bodyrows)
    if len(hdr) > 1:                                   # group row + name row
        filled, last = [], ""
        for g in hdr[0]:
            if g: last = g
            filled.append(last)
        merged = []
        for ci in range(ncol):
            name = hdr[-1][ci] if ci < len(hdr[-1]) else ""
            g = filled[ci] if ci < len(filled) else ""
            merged.append((g + " " + name).strip())
    else:
        merged = (hdr[0] + [""] * ncol)[:ncol] if hdr else [""] * ncol
    md = ["| " + " | ".join(merged) + " |",
          "|" + "---|" * ncol]
    for r in bodyrows:
        r += [""] * (ncol - len(r))
        md.append("| " + " | ".join(r) + " |")
    return cap, "\n".join(md)

# ---------- main walk ----------
out = [f"# {flatten(title)}", "", flatten(author), "",
       f"**Abstract\u2014**{flatten(abstract)}", ""]

tok = re.compile(r"\\section\{|\\subsection\{|\\begin\{figure\*?\}|"
                 r"\\input\{tables/[^}]+\}|\\begin\{itemize\}|\\begin\{enumerate\}|"
                 r"\\end\{itemize\}|\\end\{enumerate\}|\\item(?:\[[^\]]*\])?|"
                 r"\\begin\{equation\}|\\end\{equation\}|\\begin\{IEEEkeywords\}|\\end\{IEEEkeywords\}")
pos, sec, sub, para, listno = 0, 0, 0, [], 0

def flush():
    txt = " ".join(para)
    para.clear()
    # keep the blank-line paragraph breaks of the .tex source
    for blk in re.split(r"\n\s*\n", txt):
        t = flatten(blk)
        if t:
            out.extend([t, ""])

for m in tok.finditer(body):
    para.append(body[pos:m.start()]); pos = m.end()
    t = m.group(0)
    if t == r"\section{":
        flush(); sec += 1; sub = 0
        inner, j = balanced(body, m.end() - 1); pos = j
        out.extend([f"## {sec}. {flatten(inner)}", ""])
    elif t == r"\subsection{":
        flush(); sub += 1
        inner, j = balanced(body, m.end() - 1); pos = j
        out.extend([f"### {sec}.{sub} {flatten(inner)}", ""])
    elif t.startswith(r"\begin{figure"):
        flush()
        endm = re.search(r"\\end\{figure\*?\}", body[m.end():])
        blk = body[m.start():m.end() + endm.end()]
        pos = m.end() + endm.end()
        lab = re.search(r"\\label\{(fig:[^}]*)\}", blk)
        if lab and lab.group(1) in FIGNO:
            L = lab.group(1)
            out.extend([f"![Fig. {FIGNO[L]}. {flatten(FIGCAP[L])}](figs/{FIGIMG[L]})",
                        "", f"*Fig. {FIGNO[L]}. {flatten(FIGCAP[L])}*", ""])
    elif t.startswith(r"\input{"):
        flush()
        tf = re.search(r"\{(tables/[^}]+)\}", t).group(1)
        cap, md = parse_table(tf)
        out.extend([f"**Table {TABNO[tf]}.** {cap}", "", md, ""])
    elif t in (r"\begin{itemize}", r"\begin{enumerate}"):
        flush(); listno = 0
    elif t in (r"\end{itemize}", r"\end{enumerate}"):
        flush()
    elif t.startswith(r"\item"):
        flush(); listno += 1
        para.append(f"- ")
    elif t == r"\begin{equation}":
        flush(); out.append("$$")
    elif t == r"\end{equation}":
        flush()
        if out and out[-1] == "": out.pop()
        out.extend(["$$", ""])
    elif t == r"\begin{IEEEkeywords}":
        flush(); para.append("**Keywords:** ")

flush()

out.extend(["## References", ""])
for i, b in enumerate(BIB, 1):
    out.append(f"[{i}] {flatten(b)}")
    out.append("")

open(OUT, "w", encoding="utf-8").write("\n".join(out))
print("saved", OUT)
print("figs:", FIGNO)
print("tabs:", TABNO)
print("refs:", len(BIB))
