# -*- coding: utf-8 -*-
"""Build a reviewable Word version of manuscript.tex with python-docx."""
import os, re
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

ROOT = os.environ.get(
    "WAFER_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.join(ROOT, "manuscript")
TEX = os.path.join(BASE, "manuscript.tex")
OUT = os.path.join(BASE, "manuscript_review.docx")
FIGD = os.path.join(BASE, "figs")

src = open(TEX, encoding="utf-8").read()
src = re.sub(r"(?<!\\)%.*$", "", src, flags=re.M)


def balanced(s, i):
    """s[i] == '{' -> (inner, index after matching '}')."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def take_cmd(s, name):
    """Extract first argument of \\name{...}; return (arg, s_without_cmd)."""
    pat = "\\" + name + "{"
    k = s.find(pat)
    if k < 0:
        return None, s
    arg, j = balanced(s, k + len(pat) - 1)
    return arg, s[:k] + s[j:]


def strip_cmd(s, name):
    while True:
        arg, s2 = take_cmd(s, name)
        if arg is None:
            return s
        s = s2


title, src = take_cmd(src, "title")
author, src = take_cmd(src, "author")
src = strip_cmd(src, "maketitle")
src = re.sub(r"\\markboth\{.*?\}%\s*\n\{.*?\}", "", src, flags=re.S)

# ---------------- bibliography ----------------
bib_src = src.split("\\begin{thebibliography}")[1].split("\\end{thebibliography}")[0]
items = [x for x in re.split(r"\\bibitem\{", bib_src)[1:]]
bib_keys, bib_text = [], []
for it in items:
    key, rest = it.split("}", 1)
    bib_keys.append(key.strip())
    bib_text.append(rest.strip())
CITE = {k: i + 1 for i, k in enumerate(bib_keys)}

body = src.split("\\begin{abstract}")[0].split("\\begin{document}")[-1]
body = body.split("\\begin{document}")[-1]
abstract_src = src.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
kw_src = src.split("\\begin{IEEEkeywords}")[1].split("\\end{IEEEkeywords}")[0]
body = src.split("\\end{IEEEkeywords}")[1].split("\\begin{thebibliography}")[0]

# ---------------- numbering maps ----------------
fig_order = re.findall(r"\\begin\{figure\*?\}[\s\S]*?\\label\{(fig:\w+)\}", body)
FIGNO = {lab: i + 1 for i, lab in enumerate(fig_order)}
SECNO = {"sec:why": 5, "sec:results-perclass": "4.4"}
# section labels carry no text once SECNO is known
body = re.sub(r"\\label\{sec:[^}]*\}", "", body)

def parse_rows(block):
    """Parse a booktabs row block into a list of cell lists."""
    block = re.sub(r"\\(toprule|midrule|bottomrule|addlinespace|centering|small|centering)", "", block)
    block = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", block)
    rows = []
    for chunk in block.split("\\\\"):
        chunk = chunk.strip()
        if not chunk:
            continue
        cells = []
        for c in chunk.split("&"):
            c = c.strip()
            m = re.match(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{", c)
            if m:
                n = int(m.group(1))
                inner, _ = balanced(c, m.end() - 1)
                cells.append(inner)
                cells.extend([""] * (n - 1))
            else:
                cells.append(c)
        rows.append(cells)
    return rows


tab_files = re.findall(r"\\input\{tables/(tab\w+)\}", body)
TABNO, tab_caps, tab_rows, tab_hdr = {}, {}, {}, {}
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
for n, tf in enumerate(tab_files, 1):
    t = open(os.path.join(BASE, "tables", tf + ".tex"), encoding="utf-8").read()
    lab = re.search(r"\\label\{(tab:\w+)\}", t).group(1)
    TABNO[lab] = ROMAN[n - 1]
    cap, _ = balanced(t, t.index("{", t.index("\\caption")))
    tab_caps[tf] = cap.strip()
    tab = t.split("\\begin{tabular}")[1]
    tab = tab[tab.index("}") + 1:].split("\\end{tabular}")[0]
    if "\\midrule" in tab:
        head, bodypart = tab.split("\\midrule", 1)
    else:
        rows0 = parse_rows(tab)
        head, bodypart = "", tab
        head = " \\\\ ".join(" & ".join(r) for r in rows0[:1]) if rows0 else ""
    tab_hdr[tf] = parse_rows(head)
    tab_rows[tf] = parse_rows(bodypart)

SYM = [(r"\\pm", "±"), (r"\\Delta", "Δ"), (r"\\times", "×"), (r"\\approx", "≈"),
       (r"\\leq", "≤"), (r"\\geq", "≥"), (r"\\beta", "β"), (r"\\alpha", "α"),
       (r"\\rho", "ρ"), (r"\\theta", "θ"), (r"\\sigma", "σ"), (r"\\odot", "⊙"),
       (r"\\oplus", "⊕"), (r"\\sum", "Σ"), (r"\\in", "∈"), (r"\\rightarrow", "→"),
       (r"\\qquad", "  "), (r"\\quad", "  "), (r"\\;", " "), (r"\\,", " "),
       (r"\\!", ""), (r"\\%", "%"), (r"\\&", "&"), (r"\\_", "_"), (r"\\#", "#"),
       (r"\\ldots", "..."), (r"\\dots", "..."), (r"\\mathrm\{df\}", "df"),
       (r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2"), (r"\{,\}", ","),
       (r"\\texttt\{([^{}]*)\}", r"\1"), (r"\\text\{([^{}]*)\}", r"\1"),
       (r"\\MakeLowercase\{([^{}]*)\}", r"\1"), (r"\\textit\{([^{}]*)\}", r"\1"),
       (r"\\emph\{([^{}]*)\}", r"\1"), (r"\\textbf\{([^{}]*)\}", r"\1"),
       (r"\\IEEEPARstart\{([^{}]*)\}\{([^{}]*)\}", r"\1\2"),
       (r"``", "“"), (r"''", "”"), (r"~", " "), (r"\$", "$"),
       (r"\^\\circ", "°"), (r"---", "—"), (r"--", "–"),
       (r"_\{([^{}]*)\}", r"\1"), (r"\^\{([^{}]*)\}", r"^\1"),
       (r"\\dagger", "†"), (r"\^\\?dagger", "†"), (r"\^†", "†")]
SYM_DIC = dict(SYM)


def clean(t):
    t = re.sub(r"\\ref\{(fig:[^}]+)\}", lambda m: str(FIGNO.get(m.group(1), "?")), t)
    t = re.sub(r"\\ref\{(tab:[^}]+)\}", lambda m: TABNO.get(m.group(1), "?"), t)
    t = re.sub(r"\\ref\{(sec:[^}]+)\}", lambda m: str(SECNO.get(m.group(1), "?")), t)
    t = re.sub(r"\\cite\{([^}]*)\}", lambda m: "["
               + ", ".join(str(CITE.get(k.strip(), "?")) for k in m.group(1).split(","))
               + "]", t)
    for pat, rep in SYM:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\\([a-zA-Z]+)", " ", t)
    t = re.sub(r"\^\*", "", t)
    t = re.sub(r"\^ ", " ", t)
    t = t.replace("\\", "")
    t = t.replace("{", "").replace("}", "").replace("$", "").replace("%", "")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


# ---------------- document ----------------
doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Inches(0.9)
sec.top_margin = sec.bottom_margin = Inches(0.9)

st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(10.5)
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15


def para(text, size=10.5, bold=False, italic=False, align="just", space_after=6):
    p = doc.add_paragraph()
    p.alignment = {"just": WD_ALIGN_PARAGRAPH.JUSTIFY, "c": WD_ALIGN_PARAGRAPH.CENTER,
                   "l": WD_ALIGN_PARAGRAPH.LEFT}[align]
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    p.paragraph_format.space_after = Pt(space_after)
    return p


para(clean(title or "Manuscript"), size=16, bold=True, align="c", space_after=4)
author = strip_cmd(author or "", "thanks")
para(clean(author), size=10, align="c", space_after=2)
para("Manuscript draft for internal review", size=9, italic=True, align="c", space_after=10)
para("Abstract", size=10.5, bold=True, space_after=2)
para(clean(abstract_src), size=9.5)
para("Keywords: " + clean(kw_src), size=9.5, italic=True, space_after=12)

lines = body.split("\n")
buf, i, in_bul = [], 0, False
sec_n, sub_n = 0, 0


def flush():
    if buf:
        txt = clean(" ".join(buf))
        if txt:
            para(txt)
        buf.clear()


while i < len(lines):
    ln = lines[i].strip()
    if not ln:
        flush(); i += 1; continue

    if ln.startswith("\\section{"):
        flush()
        name, _ = balanced(ln, ln.index("{"))
        sec_n += 1; sub_n = 0
        para("%d. %s" % (sec_n, clean(name)), size=12.5, bold=True, space_after=6)
        i += 1; continue

    if ln.startswith("\\subsection{"):
        flush()
        name, _ = balanced(ln, ln.index("{"))
        sub_n += 1
        para("%d.%d %s" % (sec_n, sub_n, clean(name)), size=11, bold=True,
             italic=True, space_after=4)
        i += 1; continue

    if ln.startswith("\\input{tables/"):
        flush()
        tf = re.search(r"tables/(tab\w+)", ln).group(1)
        cap = clean(tab_caps[tf])
        rn = TABNO.get("tab:" + tf.split("_", 1)[1].rstrip("0123456789"), "")
        para("Table %s. %s" % (rn, cap), size=9, italic=False, align="l", space_after=3)
        hdr = tab_hdr[tf]
        rows = hdr + tab_rows[tf]
        ncol = max(len(r) for r in rows)
        tb = doc.add_table(rows=0, cols=ncol)
        tb.style = "Table Grid"
        tb.alignment = WD_TABLE_ALIGNMENT.CENTER
        for ri, r in enumerate(rows):
            cells = tb.add_row().cells
            for ci in range(ncol):
                txt = clean(r[ci]) if ci < len(r) else ""
                c = cells[ci]
                c.text = ""
                p = c.paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                run = p.add_run(txt)
                run.font.size = Pt(8)
                run.bold = (ri < len(hdr))
                if ci > 0 or ri < len(hdr):
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para("", size=6, space_after=6)
        i += 1; continue

    if ln.startswith("\\begin{figure"):
        flush()
        j = i
        while "\\end{figure" not in lines[j]:
            j += 1
        blk = "\n".join(lines[i:j + 1])
        m = re.search(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", blk)
        cap = ""
        k = blk.find("\\caption{")
        if k >= 0:
            cap, _ = balanced(blk, k + len("\\caption"))
        lab = re.search(r"\\label\{(fig:\w+)\}", blk)
        num = FIGNO.get(lab.group(1), "?") if lab else "?"
        if m:
            img = m.group(1).replace(".pdf", ".png")
            path = os.path.join(FIGD, img)
            if os.path.exists(path):
                w = Inches(6.3) if "figure*" in ln else Inches(3.3)
                doc.add_picture(path, width=w)
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        para("Fig. %s. %s" % (num, clean(cap)), size=9, align="l", space_after=10)
        i = j + 1; continue

    if ln.startswith("\\begin{itemize}") or ln.startswith("\\begin{enumerate}"):
        flush(); in_bul = True; i += 1; continue
    if ln.startswith("\\end{itemize}") or ln.startswith("\\end{enumerate}"):
        flush(); in_bul = False; i += 1; continue
    if ln.startswith("\\item"):
        flush()
        buf.append(ln[5:].strip())
        i += 1; continue
    if in_bul and not ln.startswith("\\item"):
        buf.append(ln)
        i += 1
        continue
    if ln.startswith("\\begin{equation}") or ln.startswith("\\end{equation}"):
        i += 1; continue

    buf.append(ln)
    i += 1

flush()

# ---------------- references ----------------
para("References", size=12.5, bold=True, space_after=4)
for n, (k, t) in enumerate(zip(bib_keys, bib_text), 1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    r = p.add_run("[%d] %s" % (n, clean(t)))
    r.font.size = Pt(9)

doc.save(OUT)
print("saved", OUT)
print("paragraphs:", len(doc.paragraphs), "tables:", len(doc.tables),
      "figures:", len(doc.inline_shapes))
print("figure order:", fig_order)
print("table map:", TABNO)
