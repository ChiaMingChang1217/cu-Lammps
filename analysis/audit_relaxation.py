from pathlib import Path
import re,json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).parent;ROOT=Path(__file__).resolve().parents[1]/'data'/'results'
d=pd.read_csv(O/'extended_descriptors.csv');lengths=[]
for case in d.case_id:
 s=(ROOT/case/'log.lammps').read_text(errors='replace')
 m=re.search(r'^variable L0 equal ([0-9.eE+-]+)\s*$',s,re.M)
 if not m:raise ValueError(case)
 lengths.append(float(m[1]))
d['equilibrated_L_A']=lengths;d['volume_change_pct']=100*((d.equilibrated_L_A/86.76)**3-1)
d[['case_id','pore_radius_angstrom','target_porosity','temperature_k','equilibrated_L_A','volume_change_pct']].to_csv(O/'relaxation_audit.csv',index=False)
print(d.groupby('pore_radius_angstrom').volume_change_pct.agg(['mean','min','max']).to_string())
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,T in zip(axs,[300,700]):
 g=d[np.isclose(d.strain_rate_s,2.5e10)&(d.temperature_k==T)]
 for p,gg in g.groupby('target_porosity'):
  v=gg.groupby('pore_radius_angstrom').volume_change_pct.agg(['mean','min','max']);ax.plot(v.index,v['mean'],'o-',label=f'{p:.0%}');ax.fill_between(v.index,v['min'],v['max'],alpha=.12)
 ax.axhline(0,color='gray',lw=.8);ax.set(xlabel='Nominal radius (Å)',ylabel='Cell volume change after equilibration (%)',title=f'T = {T} K');ax.legend(title='Target porosity')
fig.savefig(O/'figures'/'Fig12_equilibration.png',dpi=320,bbox_inches='tight');fig.savefig(O/'figures'/'Fig12_equilibration.pdf',bbox_inches='tight')
