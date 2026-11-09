# -*- coding: utf-8 -*-
"""Render paper/cover_letter_ieee_access.md to a submission-ready .docx."""
import os, re
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = os.environ.get(
    "WAFER_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.join(ROOT, "manuscript")
SRC  = os.path.join(BASE, "cover_letter_ieee_access.md")
OUT  = os.path.join(BASE, "cover_letter_ieee_access.docx")

doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(11)
for s in doc.sections:
    s.top_margin = s.bottom_margin = Cm(2.2)
    s.left_margin = s.right_margin = Cm(2.4)

BOLD = re.compile(r"\*\*(.+?)\*\*")

def add_runs(p, text):
    pos = 0
    for m in BOLD.finditer(text):
        if m.start() > pos:
            p.add_run(text[pos:m.start()])
        p.add_run(m.group(1)).bold = True
        pos = m.end()
    if pos < len(text):
        p.add_run(text[pos:])

lines = open(SRC, encoding="utf-8").read().splitlines()
for raw in lines:
    line = raw.rstrip()
    if not line or line == "---":
        continue
    if line.startswith("# "):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(line[2:])
        r.bold = True
        r.font.size = Pt(14)
        p.paragraph_format.space_after = Pt(12)
    elif line.startswith("## "):
        p = doc.add_paragraph()
        r = p.add_run(line[3:])
        r.bold = True
        r.font.size = Pt(12)
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
    elif line.startswith("- "):
        p = doc.add_paragraph(style="List Bullet")
        add_runs(p, line[2:])
        p.paragraph_format.space_after = Pt(4)
    else:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(8)
        add_runs(p, line)

doc.save(OUT)
print("saved", OUT, "paragraphs:", len(doc.paragraphs))
