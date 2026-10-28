from pathlib import Path
import numpy as np,pandas as pd,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
O=Path(__file__).parent;F=O/'figures';a=pd.read_json(O/'ovito_metrics.json')
cases=[f'prod_pbc_r{r}0_p15_T300_sr{rate}_s87287' for r in [3,9] for rate in ['2p5e10','2e11']]
plt.rcParams.update({'font.size':11,'savefig.dpi':300})
for kind,name in [('disp','Fig09_ovito_displacement'),('ptm','Fig10_ovito_structure')]:
 fig,axs=plt.subplots(4,4,figsize=(11,9.5))
 for i,case in enumerate(cases):
  for j,frame in enumerate([0,10,20,30]):
   ax=axs[i,j];ax.imshow(plt.imread(F/'ovito'/f'{case}_{kind}_{frame}.png'));ax.set_xticks([]);ax.set_yticks([])
   for sp in ax.spines.values():sp.set_visible(False)
   if i==0:ax.set_title(f'Strain = {frame/100:.2f}')
   if j==0:ax.set_ylabel(f'r = {3 if i<2 else 9} Å\n'+('2.5×10¹⁰ s⁻¹' if i%2==0 else '2×10¹¹ s⁻¹'))
 fig.subplots_adjust(wspace=.01,hspace=.01,left=.09,right=.97,top=.95,bottom=.1)
 if kind=='disp':
  cbax=fig.add_axes([.3,.055,.42,.015]);fig.colorbar(ScalarMappable(norm=Normalize(0,8),cmap='viridis'),cax=cbax,orientation='horizontal',label='Affine-corrected displacement (Å)')
 else:
  fig.legend(handles=[Patch(color=c,label=l) for c,l in [('#66ff66','FCC'),('#ff6666','HCP'),('#6666ff','BCC'),('#f3c75c','ICO'),('#f3f3f3','Other')]],loc='lower center',ncol=5,frameon=False)
 fig.savefig(F/(name+'.png'),bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(11,3.5),constrained_layout=True)
for case,col in zip(cases,['#0072b2','#56b4e9','#d55e00','#e69f00']):
 g=a[a.case_id==case];label=f'r={3 if "r30" in case else 9} Å; '+('low rate' if '2p5' in case else 'high rate')
 for ax,v in zip(axs,['fcc_fraction','hcp_fraction','displacement_p95_A']):ax.plot(g.strain,g[v],color=col,label=label)
for ax,l in zip(axs,['FCC fraction','HCP fraction','95th percentile displacement (Å)']):ax.set(xlabel='Engineering strain',ylabel=l)
axs[0].legend(fontsize=8);fig.savefig(F/'Fig11_ovito_evolution.png',bbox_inches='tight');fig.savefig(F/'Fig11_ovito_evolution.pdf',bbox_inches='tight');plt.close(fig)
rows=[]
ROOT=Path(__file__).resolve().parents[1]/'data'/'results'
for case in cases:
 old=pd.read_csv(ROOT/case/'stress_strain.csv');new=pd.read_csv(O/'trajectories'/case/'stress_strain.csv');grid=np.linspace(.001,.2995,1000)
 y1=np.interp(grid,old.strain,old.stress_gpa);y2=np.interp(grid,new.strain,new.stress_gpa)
 rows.append({'case_id':case,'original_peak':old.stress_gpa.max(),'rerun_peak':new.stress_gpa.max(),'peak_change_pct':100*(new.stress_gpa.max()/old.stress_gpa.max()-1),'curve_rmse_GPa':float(np.sqrt(np.mean((y2-y1)**2)))})
pd.DataFrame(rows).to_csv(O/'rerun_comparison.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
for nm,eq in [('descriptors',r'$S=\max_{0.001\leq\varepsilon\leq0.2995}\sigma_{xx}(\varepsilon),\quad I=\int_{0.001}^{0.2995}\sigma_{xx}(\varepsilon)\,d\varepsilon,\quad R=\frac{\sigma_{xx}(0.2995)}{S}$'),('quadratic',r'$\widehat{y}=b_0+\sum_{i=1}^{4}b_i x_i+\sum_{i=1}^{4}b_{ii}x_i^2+\sum_{i<j}b_{ij}x_ix_j,\quad\mathbf{x}=(p,r,t,q)$')]:
 fig=plt.figure(figsize=(10,1));fig.text(.5,.5,eq,ha='center',va='center',fontsize=16);fig.savefig(F/('eq_'+nm+'.png'),bbox_inches='tight',dpi=350);plt.close(fig)
