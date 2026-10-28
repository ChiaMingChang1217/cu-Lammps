from pathlib import Path
import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]/'data'
OUT=Path(__file__).parent; FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
d=pd.concat([pd.read_csv(p) for p in sorted(ROOT.glob('ml_dataset_production*.csv'))],ignore_index=True)
assert len(d)==512
rows=[]
for _,r in d.iterrows():
 c=pd.read_csv(ROOT/'results'/r.case_id/'stress_strain.csv').sort_values('strain')
 x=c.strain.to_numpy();y=c.stress_gpa.to_numpy()
 assert np.all(np.diff(x)>0) and x[-1]>=.2995
 row={'case_id':r.case_id}
 for w in [.1,.2,.2995]:
  # Shared observed lower bound avoids inventing the missing zero-strain stress.
  lo=.001;xx=np.r_[lo,x[(x>lo)&(x<w)],w];yy=np.interp(xx,x,y)
  row[f'W{w}']=float(np.trapz(yy,xx));row[f'S{w}']=float(yy.max())
 row['late_share']=(row['W0.2995']-row['W0.1'])/row['W0.2995']
 row['peak_gain']=row['S0.2995']/row['S0.1']
 row['retention']=float(np.interp(.2995,x,y)/row['S0.2995'])
 rows.append(row)
d=d.merge(pd.DataFrame(rows),on='case_id')
d.to_csv(OUT/'extended_descriptors.csv',index=False)
summary={'n':len(d),'peak_after_010':int((d.uts_strain>.1).sum()),'peak_near_end':int((d.uts_strain>.29).sum()),'late_share_mean':d.late_share.mean(),'late_share_range':[d.late_share.min(),d.late_share.max()],'peak_gain_mean':d.peak_gain.mean(),'retention_range':[d.retention.min(),d.retention.max()]}
for fac in ['pore_radius_angstrom','strain_rate_s','temperature_k','target_porosity']:
 summary[fac]=d.groupby(fac)[['ultimate_tensile_strength_gpa','toughness_gpa','late_share','peak_gain','retention']].mean().reset_index().to_dict('records')
(OUT/'extension_summary.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':320})
def save(fig,name):
 fig.savefig(FIG/(name+'.png'),bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,6));ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
boxes=[(0.3,4.4,'Factorial design','4 radii × 4 porosities\n4 temperatures × 4 rates × 2 seeds'),(5.3,4.4,'Periodic multi-void Cu','24³ fcc cells; 55,296 atoms\nMinimum-image pore placement'),(5.3,2.4,'Tensile MD to strain 0.30','EAM Cu; [100] loading\n512 production cases'),(.3,2.4,'Window and trajectory analysis','Peak stress, integrated response\n4 additional OVITO trajectories'),(.3,.4,'Grouped surrogate validation','Hold geometry–temperature groups\nEmpirical quadratic relations'),(5.3,.4,'Mechanically interpreted maps','3D surfaces; window dependence\nNon-affine motion and local order')]
for x,y,title,body in boxes:
 ax.add_patch(FancyBboxPatch((x,y),4.4,1.2,boxstyle='round,pad=.10',facecolor='#edf3f7',edgecolor='#284d63',linewidth=1.4))
 ax.text(x+2.2,y+.85,title,ha='center',weight='bold',fontsize=11)
 ax.text(x+2.2,y+.35,body,ha='center',va='center',fontsize=10)
for a,b in [((4.8,5),(5.1,5)),((7.5,4.25),(7.5,3.8)),((5.1,3), (4.8,3)),((2.5,2.25),(2.5,1.8)),((4.8,1),(5.1,1))]:
 ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.8,'color':'#284d63'})
save(fig,'Fig01_workflow')
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,rate,tag in zip(axs,[2.5e10,2e11],['2p5e10','2e11']):
 for radius,color in zip([3,5,7,9],['#0072B2','#009E73','#E69F00','#CC79A7']):
  curves=[];grid=np.linspace(.001,.2995,600)
  for seed in [19331,87287]:
   c=pd.read_csv(ROOT/'results'/f'prod_pbc_r{radius}0_p15_T300_sr{tag}_s{seed}'/'stress_strain.csv')
   curves.append(np.interp(grid,c.strain,c.stress_gpa))
  curves=np.array(curves);ax.plot(grid,curves.mean(0),color=color,label=f'{radius} Å');ax.fill_between(grid,curves.min(0),curves.max(0),color=color,alpha=.15)
 ax.axvline(.1,color='gray',ls='--',lw=1);ax.set(xlabel='Engineering strain',ylabel='Axial virial stress (GPa)',title=f'Rate = {rate:.2g} s⁻¹');ax.legend(title='Void radius')
save(fig,'Fig02_stress_evolution')
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for r,g in d.groupby('pore_radius_angstrom'):
 m=g.groupby('strain_rate_s')[['peak_gain','late_share']].mean()
 axs[0].plot(np.log2(m.index/5e10),m.peak_gain,'o-',label=f'{r:g} Å')
 axs[1].plot(np.log2(m.index/5e10),m.late_share*100,'o-',label=f'{r:g} Å')
for ax in axs: ax.set_xlabel('log₂(rate / 5×10¹⁰ s⁻¹)');ax.legend(title='Void radius')
axs[0].set_ylabel('Peak ratio S₀.₂₉₉₅ / S₀.₁');axs[1].set_ylabel('Integrated response after strain 0.10 (%)')
save(fig,'Fig03_window_dependence')
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,col in zip(axs,['retention','late_share']):
 table=d.groupby(['pore_radius_angstrom','strain_rate_s'])[col].mean().unstack()
 im=ax.imshow(table.values,origin='lower',aspect='auto',cmap='viridis')
 ax.set_xticks(range(4),['2.5','5','10','20']);ax.set_yticks(range(4),['3','5','7','9']);ax.set(xlabel='Strain rate (10¹⁰ s⁻¹)',ylabel='Void radius (Å)',title='Residual-to-peak stress ratio' if col=='retention' else 'Post-0.10 integrated fraction')
 for i in range(4):
  for j in range(4): ax.text(j,i,f'{table.values[i,j]:.2f}',ha='center',va='center',color='white' if table.values[i,j]<np.median(table.values) else 'black')
 fig.colorbar(im,ax=ax,shrink=.8)
save(fig,'Fig04_response_atlas')
print(json.dumps(summary,indent=2))
