from pathlib import Path
import json
from copy import deepcopy
from lxml import etree
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
O=Path(__file__).parent
DATA=json.loads((O/'equation_definitions.json').read_text(encoding='utf-8'))
XSL=etree.XSLT(etree.parse('C:/Program Files/Microsoft Office/root/Office16/MML2OMML.XSL'))
def run_props():
 rp=OxmlElement('w:rPr')
 fonts=OxmlElement('w:rFonts')
 for key in ['ascii','hAnsi','cs']:fonts.set(qn('w:'+key),'Cambria Math')
 rp.append(fonts)
 for key in ['sz','szCs']:
  el=OxmlElement('w:'+key);el.set(qn('w:val'),'24');rp.append(el)
 return rp
def append_equation(doc,key):
 p=doc.add_paragraph()
 p.alignment=WD_ALIGN_PARAGRAPH.CENTER
 p.paragraph_format.space_before=Pt(8);p.paragraph_format.space_after=Pt(8)
 ppr=p._p.get_or_add_pPr();ppr.append(run_props())
 converted=XSL(etree.fromstring(DATA[key]['mathml'].encode()))
 root=deepcopy(converted.getroot())
 math=root if root.tag==qn('m:oMath') else root.find('.//'+qn('m:oMath'))
 assert math is not None
 for run in math.iter(qn('m:r')):
  for old in list(run):
   if old.tag==qn('w:rPr'):run.remove(old)
  rpr=run.find(qn('m:rPr'))
  run.insert(1 if rpr is not None else 0,run_props())
 # Set the control font for fractions, sums, integrals and scripts.
 for pr in list(math.iter()):
  if pr.tag.startswith('{http://schemas.openxmlformats.org/officeDocument/2006/math}') and pr.tag.endswith('Pr') and pr.tag not in [qn('m:rPr'),qn('m:ctrlPr')]:
   ctrl=pr.find(qn('m:ctrlPr'))
   if ctrl is None:ctrl=OxmlElement('m:ctrlPr');pr.append(ctrl)
   for old in list(ctrl):ctrl.remove(old)
   ctrl.append(run_props())
 para=OxmlElement('m:oMathPara');pr=OxmlElement('m:oMathParaPr')
 jc=OxmlElement('m:jc');jc.set(qn('m:val'),'center');pr.append(jc);para.append(pr);para.append(math)
 p._p.append(para)

