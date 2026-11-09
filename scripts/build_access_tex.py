# -*- coding: utf-8 -*-
"""Derive manuscript_access.tex (ieeeaccess.cls) from manuscript.tex (IEEEtran).

Only the preamble and the front matter are rewritten; body, figures, tables and
the bibliography are carried over verbatim so that no number or claim changes.
"""
import os, re

ROOT = os.environ.get(
    "WAFER_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.join(ROOT, "manuscript")
SRC = os.path.join(BASE, "manuscript.tex")
OUT = os.path.join(BASE, "manuscript_access.tex")

src = open(SRC, encoding="utf-8").read().replace("\r\n", "\n")

# ---- split: everything before \begin{abstract} is replaced -----------------
head, rest = src.split(r"\begin{abstract}", 1)

TITLE = ("Is Class Imbalance the Bottleneck? A Controlled Re-evaluation of\n"
         "Attention, Rebalancing, and Robustness for Wafer Map Defect Detection")
SHORT = "Controlled Re-evaluation of Attention, Rebalancing, and Robustness"

new_head = r"""%=====================================================================
%  IEEE Access version (official template ACCESS_latex_template_20260513)
%  Build: pdflatex manuscript_access.tex   (run twice for \ref resolution)
%  Body text, figures, tables and bibliography are identical to
%  manuscript.tex; only the preamble and front matter differ.
%=====================================================================
\documentclass{ieeeaccess}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{url}
\usepackage{bm}
\makeatletter
\AtBeginDocument{\DeclareMathVersion{bold}
\SetSymbolFont{operators}{bold}{T1}{times}{b}{n}
\SetSymbolFont{NewLetters}{bold}{T1}{times}{b}{it}
\SetMathAlphabet{\mathrm}{bold}{T1}{times}{b}{n}
\SetMathAlphabet{\mathit}{bold}{T1}{times}{b}{it}
\SetMathAlphabet{\mathbf}{bold}{T1}{times}{b}{n}
\SetMathAlphabet{\mathtt}{bold}{OT1}{pcr}{b}{n}
\SetSymbolFont{symbols}{bold}{OMS}{cmsy}{b}{n}
\renewcommand\boldmath{\@nomath\boldmath\mathversion{bold}}}
\makeatother

\graphicspath{{figs/}}

\begin{document}
\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}
\doi{10.1109/ACCESS.2026.XXXXXXX}

\title{__TITLE__}

\author{\uppercase{First A. Author}\authorrefmark{1},
\IEEEmembership{Member, IEEE},
\uppercase{Second B. Author}\authorrefmark{2},
AND \uppercase{Third C. Author}\authorrefmark{3}}

\address[1]{Department, University, City, Country (e-mail: author@example.edu)}
\address[2]{Department, University, City, Country}
\address[3]{Department, University, City, Country}

\tfootnote{This work was supported by [funding agency and grant number, to be
completed].}

\markboth{Author \headeretal: __SHORT__}{Author \headeretal: __SHORT__}

\corresp{Corresponding author: First A. Author (e-mail: author@example.edu).}

\begin{abstract}""".replace("__TITLE__", TITLE).replace("__SHORT__", SHORT)

body = rest

# ---- IEEEtran-only constructs ---------------------------------------------
body = body.replace(r"\begin{IEEEkeywords}", r"\begin{keywords}")
body = body.replace(r"\end{IEEEkeywords}", r"\end{keywords}")
body = body.replace(r"\end{keywords}",
                    "\\end{keywords}\n\n\\titlepgskip=-21pt\n\n\\maketitle", 1)
body = re.sub(r"\\IEEEPARstart\{([^}]*)\}\{([^}]*)\}", r"\\PARstart{\1}{\2}", body)

BIO = r"""
%=====================================================================
%  Author biographies (required by IEEE Access). Replace the placeholder
%  text below with the real biographies. Once the photographs are available,
%  switch each entry from the no-photo environment to the photo environment,
%  using \includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{authorN.png}
%  as the optional argument.
%=====================================================================
\begin{IEEEbiographynophoto}{First A. Author}
[Biography to be completed: degrees, institutions, current position, research
interests, and IEEE membership grade.]
\end{IEEEbiographynophoto}

\begin{IEEEbiographynophoto}{Second B. Author}
[Biography to be completed.]
\end{IEEEbiographynophoto}

\begin{IEEEbiographynophoto}{Third C. Author}
[Biography to be completed.]
\end{IEEEbiographynophoto}

\EOD

\end{document}
"""

body = body.replace("\n\\end{document}\n", "\n")
body = body.rstrip() + "\n"

out = new_head + body + "\n" + BIO
open(OUT, "w", encoding="utf-8", newline="\n").write(out)

# ---- self-check ------------------------------------------------------------
checks = {
    "ieeeaccess class": r"\documentclass{ieeeaccess}" in out,
    "history": r"\history{" in out,
    "corresp": r"\corresp{" in out,
    "keywords block": r"\begin{keywords}" in out and r"\end{keywords}" in out,
    "no IEEEPARstart": r"\IEEEPARstart" not in out,
    "no IEEEkeywords": "IEEEkeywords" not in out,
    "no \\thanks": r"\thanks" not in out,
    "EOD": r"\EOD" in out,
    "single end{document}": out.count(r"\end{document}") == 1,
    "abstract": out.count(r"\begin{abstract}") == 1,
    "bibitems preserved": out.count(r"\bibitem") == src.count(r"\bibitem"),
    "figures preserved": out.count(r"\begin{figure") == src.count(r"\begin{figure"),
    "tables preserved": out.count(r"\input{tables/") == src.count(r"\input{tables/"),
}
for k, v in checks.items():
    print(("OK   " if v else "FAIL ") + k)
print("bibitems:", out.count(r"\bibitem"), "chars:", len(out))
