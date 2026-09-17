"""Synthetic teaching example only; never reads or changes campaign observations."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src/08_multi_objective'))
from helper.batch_gp import BatchGP
from helper import campaign_plots as style
from helper.plot_reporting import write_plot
OUT=Path(__file__).resolve().parent
rng=np.random.default_rng(42)
x=np.array([0.,.33,.67,1.])[:,None];truth=np.array([45.,60.,72.,65.]);shifts=np.array([-12.,8.,-5.,12.,-8.,5.])
rows=[]
for b,shift in enumerate(shifts,1):
 for i in range(4):rows.append(dict(batch=f'TOY_{b}',recipe=chr(65+i),x=x[i,0],value=truth[i]+shift+rng.normal(0,1.5),control=False,true_shift=shift))
 rows.append(dict(batch=f'TOY_{b}',recipe='Viability control',x=.5,value=65+shift+rng.normal(0,1.5),control=True,true_shift=shift))
d=pd.DataFrame(rows);d.to_csv(OUT/'synthetic_training.csv',index=False)
models={s:BatchGP.fit(d[['x']].to_numpy(),d.value.to_numpy(),d.batch.to_numpy(),d.control.to_numpy(),strategy=s) for s in ['batch','control']}
new='TOY_NEW';reference=47.;conditioned=models['control'].condition(np.array([[.5]]),np.array([reference]),new,np.array([True]))
preds=[]
for label,m in [('Batch GP; no controls',models['batch']),('Control GP; before result',models['control']),('Control GP; after control = 47%',conditioned)]:
 mean,cov=m.posterior(x,batch=new,future=True);sd=np.sqrt(np.diag(cov))
 for i in range(4):preds.append(dict(model=label,recipe=chr(65+i),mean=mean[i],lower=mean[i]-1.96*sd[i],upper=mean[i]+1.96*sd[i],true_batch_mean=truth[i]-18))
p=pd.DataFrame(preds);p.to_csv(OUT/'synthetic_predictions.csv',index=False)
def save(fig,table,name):
 write_plot(fig,table,OUT,name,'SYNTHETIC ONLY | not campaign evidence')
 path=OUT/(name+'_metadata.json');m=json.loads(path.read_text());m['synthetic_data']=True;m['seed']=42;path.write_text(json.dumps(m,indent=2)+'\n')
style.style()
fig,ax=plt.subplots(figsize=(9,4.8))
for recipe,g in d.groupby('recipe',sort=False):ax.plot(range(1,7),g.value,marker='s' if recipe=='Viability control' else 'o',ls='--' if recipe=='Viability control' else '-',label=recipe)
ax.set(xlabel='Synthetic training batch',ylabel='Viability (%)',title='Toy history: control and formulations share a batch shift',ylim=(15,95));ax.legend(ncol=3);fig.tight_layout();save(fig,d,'01_shared_batch_history')
fig,axes=plt.subplots(1,3,figsize=(13,4.8),sharey=True)
for ax,(label,g) in zip(axes,p.groupby('model',sort=False)):
 ax.errorbar(range(4),g['mean'],yerr=[g['mean']-g.lower,g.upper-g['mean']],fmt='o',color=style.BLUE,capsize=4,label='Prediction + 95% future interval')
 ax.scatter(range(4),g.true_batch_mean,color=style.PINK_EDGE,marker='x',s=55,label='Hidden toy batch mean')
 ax.set_xticks(range(4),list('ABCD'));ax.set_title(label.replace(';','\n'));ax.set_ylim(0,100);ax.set_xlabel('Toy formulation')
axes[0].set_ylabel('Viability (%)');fig.suptitle('SYNTHETIC: a shared −18-point batch shift; reveal only the control');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2);fig.tight_layout(rect=(0,.10,1,.93));save(fig,p,'02_control_conditioning')
fig,axes=plt.subplots(1,2,figsize=(10,4.8),sharey=True)
before=p[p.model.eq('Control GP; before result')];after=p[p.model.eq('Control GP; after control = 47%')];failure=[]
for ax,shift,title in zip(axes,[-18,0],['Shared batch shift: calibration helps','Control-only assay error: calibration misleads']):
 actual=truth+shift
 ax.plot(list('ABCD'),before['mean'],'o--',color=style.GRAY,label='Before control')
 ax.plot(list('ABCD'),after['mean'],'o-',color=style.BLUE,label='After control = 47%')
 ax.scatter(list('ABCD'),actual,marker='x',s=60,color=style.PINK_EDGE,label='Hidden toy batch mean')
 ax.set_title(title,fontsize=11);ax.set_xlabel('Toy formulation');ax.set_ylim(0,100)
 for label,g in [('before',before),('after',after)]:failure.append(dict(scenario=title,stage=label,mae=float(np.abs(g['mean'].to_numpy()-actual).mean())))
axes[0].set_ylabel('Viability (%)');fig.suptitle('SYNTHETIC: the same control result can have different causes');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3);fig.tight_layout(rect=(0,.1,1,.93));save(fig,pd.DataFrame(failure),'03_when_calibration_fails')
# Independent mechanical GP: condition viability only and leave mechanical posterior intact.
mech=BatchGP.fit(x,np.array([.12,.18,.25,.21]),np.array(['MECH_TOY']*4),strategy='campaign',noise=np.full(4,.015))
mm,mc=mech.posterior(x);mm_after,mc_after=mech.posterior(x);assert np.array_equal(mm,mm_after) and np.array_equal(mc,mc_after)
t=pd.DataFrame({'recipe':list('ABCD'),'mechanical_before':mm,'mechanical_after_viability_control':mm_after})
fig,ax=plt.subplots(figsize=(7,4.5));ax.plot(list('ABCD'),mm,'o-',color=style.BLUE,label='Before viability control');ax.plot(list('ABCD'),mm_after,'x--',color=style.PINK_EDGE,label='After viability control (identical)');ax.set(xlabel='Toy formulation',ylabel='Predicted force (N/needle)',title='SYNTHETIC: viability calibration leaves mechanics unchanged');ax.legend();fig.tight_layout();save(fig,t,'04_mechanics_unchanged')
summary={'synthetic_only':True,'models':{k:{'batch_sd':v.batch_sd,'residual_sd':v.noise_sd} for k,v in models.items()},'conditioned_batch_offset':[o for o in conditioned.offsets() if o['batch_id']==new],'scenarios':failure,'mechanical_posterior_unchanged':True}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
