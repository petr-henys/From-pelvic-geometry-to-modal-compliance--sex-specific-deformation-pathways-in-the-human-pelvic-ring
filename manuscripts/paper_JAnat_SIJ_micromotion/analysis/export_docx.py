"""Export the manuscript and Supporting Information with embedded figures and native math.

Requires pandoc and pdftoppm; compiles no FE model. Run after the LaTeX build.
"""
from pathlib import Path
import argparse,re,subprocess,tempfile,struct,zipfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parents[1]

def export(source,aux):
 base=source.parent
 with tempfile.TemporaryDirectory(prefix='janat_docx_') as tmpname:
  tmp=Path(tmpname);s=source.read_text()
  s=re.sub(r'\\newcommand\{\\(?:Std|Supp)TableInput\}.*?(?=\\title)', '',s,flags=re.S)
  def expand(text):
   def read(m):
    path=base/m[1]
    if not path.suffix:path=path.with_suffix('.tex')
    return expand(path.read_text())
   return re.sub(r'\\(?:input|StdTableInput|SuppTableInput)\{([^}]+)\}',read,text)
  s=expand(s)
  refs=dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}',aux.read_text()))
  def reference(m):
   if m[2] not in refs:raise ValueError('Unresolved reference: '+m[2])
   return '('+refs[m[2]]+')' if m[1]=='eqref' else refs[m[2]]
  s=re.sub(r'\\(eqref|ref)\{([^}]+)\}',reference,s)
  for kind,prefix in [('figure','Figure'),('table','Table')]:
   def env(m):
    body=m[1];lab=re.search(r'\\label\{([^}]+)\}',body)
    if not lab:raise ValueError('Unlabeled main-text float')
    n=refs[lab[1]];body=body.replace('\\caption{',r'\caption{'+prefix+' '+n+'. ',1)
    if kind=='figure':return body.replace('\\caption{','\n\n{',1)
    return '\\begin{table}\n'+body+'\n\\end{table}'
   s=re.sub(r'\\begin\{'+kind+r'\}(?:\[[^]]*\])?(.*?)\\end\{'+kind+r'\}',env,s,flags=re.S)
  def img(m):
   src=(base/m[1]).resolve()
   if src.suffix.lower()=='.pdf':
    dest=tmp/src.stem
    subprocess.run(['pdftoppm','-singlefile','-scale-to','2400','-png',str(src),str(dest)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    src=dest.with_suffix('.png')
   w,h=struct.unpack('>II',src.read_bytes()[16:24]);width=min(6.2,7.5*w/h)
   return r'\includegraphics[width='+str(width)+'in]{'+str(src)+'}'
  s=re.sub(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',img,s)
  s=s.replace(r'\begin{abstract}',r'\section*{Abstract}').replace(r'\end{abstract}','')
  s=s.replace(r'\bibliography{bib/refs}',r'\section*{References}'+'\n'+r'\bibliography{bib/refs}')
  # Keep section cross-references meaningful in Word, including Appendix A.
  state={'section':0,'subsection':0,'appendix':False}
  def heading(m):
   if m[0]==r'\appendix':state.update(section=0,subsection=0,appendix=True);return ''
   kind,star,title=m[1],m[2],m[3]
   if star:return m[0]
   if kind=='section':
    state['section']+=1;state['subsection']=0
    number=chr(64+state['section']) if state['appendix'] else str(state['section'])
    prefix=('Appendix '+number+'. ') if state['appendix'] else number+'. '
   else:
    state['subsection']+=1
    number=chr(64+state['section']) if state['appendix'] else str(state['section'])
    prefix=number+'.'+str(state['subsection'])+'. '
   return '\\'+kind+'{'+prefix+title+'}'
  s=re.sub(r'\\appendix|\\(section|subsection)(\*)?\{([^{}]+)\}',heading,s)
  expanded=tmp/'expanded.tex';expanded.write_text(s);out=source.with_suffix('.docx')
  subprocess.run(['pandoc',str(expanded),'-f','latex','-t','docx','--standalone','--citeproc','--bibliography='+str(P/'bib/refs.bib'),'-o',str(out)],check=True,cwd=base)
  with zipfile.ZipFile(out) as z:data={n:z.read(n) for n in z.namelist()}
  ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'};tag=lambda n:'{'+ns['w']+'}'+n
  root=ET.fromstring(data['word/document.xml']);sect=root.find('.//w:sectPr',ns)
  size=ET.SubElement(sect,tag('pgSz'));size.set(tag('w'),'11906');size.set(tag('h'),'16838')
  margin=ET.SubElement(sect,tag('pgMar'))
  for side in ['top','bottom','left','right']:margin.set(tag(side),'1417')
  # Keep native Word tables within the available text width.
  for table in root.findall('.//w:tbl',ns):
   pr=table.find('w:tblPr',ns)
   if pr is None:pr=ET.SubElement(table,tag('tblPr'))
   width=pr.find('w:tblW',ns)
   if width is None:width=ET.SubElement(pr,tag('tblW'))
   width.set(tag('type'),'pct');width.set(tag('w'),'5000')
   grid=table.find('w:tblGrid',ns)
   if grid is not None:
    cols=list(grid);total=sum(int(c.get(tag('w'),'0')) for c in cols)
    if total>9072:
     for c in cols:c.set(tag('w'),str(round(int(c.get(tag('w')))*9072/total)))
   for run in table.findall('.//w:r',ns):
    rp=run.find('w:rPr',ns)
    if rp is None:rp=ET.SubElement(run,tag('rPr'))
    sz=rp.find('w:sz',ns)
    if sz is None:sz=ET.SubElement(rp,tag('sz'))
    sz.set(tag('val'),'18')
  data['word/document.xml']=ET.tostring(root,encoding='utf-8',xml_declaration=True)
  with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
   for n,b in data.items():z.writestr(n,b)
  print(out)

def main():
 p=argparse.ArgumentParser();p.add_argument('--main-aux',type=Path,default=P/'main.aux');p.add_argument('--supp-aux',type=Path,default=P/'supplementary/supplement.aux');a=p.parse_args()
 export(P/'main.tex',a.main_aux);export(P/'supplementary/supplement.tex',a.supp_aux)
if __name__=='__main__':main()
