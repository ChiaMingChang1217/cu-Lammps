from pathlib import Path
import json
from xml.sax.saxutils import escape
O=Path(__file__).parent
def tag(n,*children,**attrs):
 a=''.join(' '+k+'="'+v+'"' for k,v in attrs.items())
 return '<'+n+a+'>'+''.join(children)+'</'+n+'>'
def mi(s):return tag('mi',escape(s))
def mn(s):return tag('mn',str(s))
def mo(s):return tag('mo',escape(s))
def row(*s):return tag('mrow',*s)
def sub(a,b):return tag('msub',a,b)
def sup(a,b):return tag('msup',a,b)
def frac(a,b):return tag('mfrac',a,b)
def par(a):return row(mo('('),a,mo(')'))
def summ(lower,upper=None):return tag('munderover',mo('∑'),lower,upper) if upper else tag('munder',mo('∑'),lower)
def app(f,a):return row(f,par(a))
def eq(a,b):return row(a,mo('='),b)
def comma():return row(mo(','),tag('mspace',width='1em'))
i,j=mi('i'),mi('j')
rij=sub(mi('r'),mi('ij'));rho=sub(mi('ρ'),i)
u=row(summ(i,mi('N')),app(mi('F'),rho),mo('+'),frac(mn(1),mn(2)),summ(i,mi('N')),summ(row(j,mo('≠'),i)),app(mi('ψ'),rij))
e={}
e['eam']=(eq(mi('U'),u),r'U=\sum_{i=1}^{N}F(\rho_i)+\frac{1}{2}\sum_{i=1}^{N}\sum_{j\ne i}\psi(r_{ij})')
# Make lower bounds agree with the TeX counterpart.
e['eam']=(e['eam'][0].replace('<mi>i</mi></munderover>','<mi>i</mi></munderover>'),e['eam'][1])
e['density']=(eq(rho,row(summ(row(j,mo('≠'),i)),app(mi('f'),rij))),r'\rho_i=\sum_{j\ne i}f(r_{ij})')
e['force']=(eq(sub(mi('F'),i),row(mo('−'),sub(mi('∇'),i),mi('U'))),r'\mathbf{F}_i=-\nabla_i U')
eps=mi('ε');lx=sub(mi('L'),mi('x'));l0=sub(mi('L'),mi('x,0'))
e['strain']=(row(eq(app(eps,mi('t')),frac(row(app(lx,mi('t')),mo('−'),l0),l0)),comma(),eq(app(lx,mi('t')),row(l0,par(row(mn(1),mo('+'),mi('ė'),mi('t')))))),r'\varepsilon(t)=\frac{L_x(t)-L_{x,0}}{L_{x,0}},\qquad L_x(t)=L_{x,0}(1+\dot{\varepsilon}t)')
kin=row(summ(i,mi('N')),sub(mi('m'),i),sup(sub(mi('v'),mi('ix')),mn(2)),mo('+'),sub(mi('W'),mi('xx')))
e['virial']=(row(sub(mi('σ'),mi('xx')),mo('='),mo('−'),sub(mi('P'),mi('xx')),mo('='),mo('−'),frac(kin,mi('V'))),r'\sigma_{xx}=-P_{xx}=-\frac{\sum_{i=1}^{N}m_i v_{ix}^{2}+W_{xx}}{V}')
low=row(mn('.001'),mo('≤'),eps,mo('≤'),mn('.2995'))
e['peak']=(eq(mi('S'),row(tag('munder',mi('max'),low),app(sub(mi('σ'),mi('xx')),eps))),r'S=\max_{0.001\leq\varepsilon\leq0.2995}\sigma_{xx}(\varepsilon)')
e['integral']=(eq(mi('I'),row(tag('msubsup',mo('∫'),mn('0.001'),mn('0.2995')),app(sub(mi('σ'),mi('xx')),eps),mi('d'),eps)),r'I=\int_{0.001}^{0.2995}\sigma_{xx}(\varepsilon)\,d\varepsilon')
e['retention']=(eq(mi('R'),frac(app(sub(mi('σ'),mi('xx')),mn('0.2995')),mi('S'))),r'R=\frac{\sigma_{xx}(0.2995)}{S}')
x=mi('x');b=mi('b');k=mi('k')
pred=row(sub(b,mn(0)),mo('+'),summ(row(i,mo('='),mn(1)),mn(4)),sub(b,i),sub(x,i),mo('+'),summ(row(i,mo('='),mn(1)),mn(4)),sub(b,mi('ii')),sup(sub(x,i),mn(2)),mo('+'),summ(row(i,mo('<'),j)),sub(b,mi('ij')),sub(x,i),sub(x,j))
e['quadratic']=(eq(tag('mover',mi('ŷ'),mo('')),pred),r'\hat{y}=b_0+\sum_{i=1}^{4}b_i x_i+\sum_{i=1}^{4}b_{ii}x_i^2+\sum_{i<j}b_{ij}x_i x_j')
# Correct lower limits and use genuine mathematical accents.
for key in ['eam','virial']:
 m,t=e[key];m=m.replace('<munderover><mo>∑</mo><mi>i</mi>','<munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow>');e[key]=(m,t)
e['quadratic']=(e['quadratic'][0].replace('<mover><mi>ŷ</mi><mo></mo></mover>','<mover accent="true"><mi>y</mi><mo>^</mo></mover>'),e['quadratic'][1])
e['strain']=(e['strain'][0].replace('<mi>ė</mi>','<mover accent="true"><mi>ε</mi><mo>˙</mo></mover>'),e['strain'][1])
e['force']=(e['force'][0].replace('<mi>F</mi>','<mi mathvariant="bold">F</mi>'),e['force'][1])
data={name:{'mathml':'<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">'+m+'</math>','latex':t} for name,(m,t) in e.items()}
(O/'equation_definitions.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
# Render the companion PDF at a fixed physical 12 pt, without width normalization.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset']='stix'
D=O/'figures';D.mkdir(exist_ok=True)
for key,d in data.items():
 fig=plt.figure(figsize=(7,.75),dpi=400)
 fig.text(.5,.5,'$'+d['latex']+'$',fontsize=12,ha='center',va='center')
 fig.savefig(D/('eq_'+key+'.png'),dpi=400,bbox_inches='tight',pad_inches=.03)
 plt.close(fig)
print('Prepared',len(data),'equations at 12 pt.')

