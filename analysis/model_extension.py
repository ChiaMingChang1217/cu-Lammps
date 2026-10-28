from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.model_selection import GroupKFold,cross_val_predict
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.base import clone
O=Path(__file__).parent; F=O/'figures'; d=pd.read_csv(O/'extended_descriptors.csv')
X=np.column_stack([d.actual_porosity/.1,d.pore_radius_angstrom/6,(d.temperature_k-400)/300,np.log2(d.strain_rate_s/5e10)])
groups=d[['target_porosity','pore_radius_angstrom','temperature_k']].astype(str).agg('|'.join,axis=1)
geometry=d[['target_porosity','pore_radius_angstrom']].astype(str).agg('|'.join,axis=1)
cv=list(GroupKFold(5).split(X,groups=groups)); gc=list(GroupKFold(4).split(X,groups=geometry))
pd.DataFrame({'case_id':d.case_id,'group':groups,'fold':np.select([np.isin(np.arange(len(d)),te) for _,te in cv],range(1,6))}).to_csv(O/'fold_assignments.csv',index=False)
models={'Linear':LinearRegression(),'Quadratic':make_pipeline(PolynomialFeatures(2,include_bias=False),LinearRegression()),'Extra Trees':ExtraTreesRegressor(n_estimators=400,min_samples_leaf=2,random_state=42,n_jobs=2),'Gradient Boosting':GradientBoostingRegressor(n_estimators=300,learning_rate=.035,max_depth=3,random_state=42)}
rows=[]; coefs={}; summary={}
plt.rcParams.update({'font.size':10,'savefig.dpi':320,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 fig.savefig(F/(name+'.png'),bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
for target,name,label in [('S0.2995','S','Peak axial stress (GPa)'),('W0.2995','I','Integrated response (GPa)'),('retention','R','Stress retention ratio')]:
 y=d[target].values;fig,axs=plt.subplots(1,3,figsize=(11,3.6),constrained_layout=True)
 ai=0
 for mn,m in models.items():
  pred=cross_val_predict(m,X,y,cv=cv)
  rows.append({'target':name,'validation':'geometry-temperature groups','model':mn,'R2':r2_score(y,pred),'MAE':mean_absolute_error(y,pred),'RMSE':np.sqrt(mean_squared_error(y,pred))})
  pd.DataFrame({'case_id':d.case_id,'observed':y,'prediction':pred}).to_csv(O/f'oof_{name}_{mn.replace(" ","_")}.csv',index=False)
  if mn!='Linear':
   ax=axs[ai];ai+=1;sc=ax.scatter(y,pred,c=X[:,3],s=12,cmap='viridis',alpha=.7);ax.plot([y.min(),y.max()],[y.min(),y.max()],'k--',lw=1);ax.set(xlabel='MD '+label,ylabel='Held-out prediction',title=f'{mn}\nR²={r2_score(y,pred):.3f}  MAE={mean_absolute_error(y,pred):.3f}')
  if mn in ['Quadratic','Gradient Boosting']:
   pr=cross_val_predict(m,X,y,cv=gc);rows.append({'target':name,'validation':'geometry only groups','model':mn,'R2':r2_score(y,pr),'MAE':mean_absolute_error(y,pr),'RMSE':np.sqrt(mean_squared_error(y,pr))})
 fig.colorbar(sc,ax=axs,shrink=.7,label='log₂(rate / 5×10¹⁰ s⁻¹)');save(fig,'Fig05_parity_'+name)
 m=clone(models['Quadratic']).fit(X,y);coefs[name]={'intercept':float(m[-1].intercept_),**dict(zip(m[0].get_feature_names_out(['p','r','t','q']),map(float,m[-1].coef_)))}
 no_r=cross_val_predict(clone(models['Quadratic']),X[:,[0,2,3]],y,cv=cv)
 rows.append({'target':name,'validation':'geometry-temperature groups','model':'Quadratic without radius','R2':r2_score(y,no_r),'MAE':mean_absolute_error(y,no_r),'RMSE':np.sqrt(mean_squared_error(y,no_r))})
 for rate in sorted(d.strain_rate_s.unique()):
  te=np.isclose(d.strain_rate_s,rate);pr=clone(models['Quadratic']).fit(X[~te],y[~te]).predict(X[te]);rows.append({'target':name,'validation':f'leave rate {rate:.3g}','model':'Quadratic','R2':r2_score(y[te],pr),'MAE':mean_absolute_error(y[te],pr),'RMSE':np.sqrt(mean_squared_error(y[te],pr))})
 if name in ['S','I']:
  fig=plt.figure(figsize=(11,4.2));lo=float(y.min());hi=float(y.max())
  for i,temp in enumerate([100,300,700]):
   ax=fig.add_subplot(1,3,i+1,projection='3d');p,q=np.meshgrid(np.linspace(.05,.20,35),np.linspace(-1,2,35));xx=np.column_stack([p.ravel()/.1,np.full(p.size,5/6),np.full(p.size,(temp-400)/300),q.ravel()]);z=m.predict(xx).reshape(p.shape)
   ax.plot_surface(p*100,q,z,cmap='viridis',vmin=lo,vmax=hi,alpha=.9);ax.set(xlabel='Porosity (%)',ylabel='log₂(rate / 5×10¹⁰)',zlabel=name+' (GPa)',title=f'T = {temp} K; r = 5 Å',zlim=(lo,hi));ax.view_init(25,-55)
  save(fig,'Fig06_surface_'+name)
pd.DataFrame(rows).to_csv(O/'model_metrics.csv',index=False)
(O/'equations.json').write_text(json.dumps(coefs,indent=2))
print(pd.DataFrame(rows).to_string(index=False))
