"""Offline control monitoring. No production normalization."""
import pandas as pd

def control_diagnostics(observations):
    if 'experimental_role' not in observations: return pd.DataFrame()
    c=observations.loc[observations.experimental_role.eq('campaign_control') & observations.endpoint.eq('viability_percent')].copy()
    if c.empty: return pd.DataFrame()
    t=c.groupby(['formulation_id','batch_id','endpoint']).value.agg(['mean','std','count']).reset_index()
    t['conditional_shift']=float('nan'); t['reference_mean']=float('nan')
    for _,g in t.groupby(['formulation_id','endpoint']):
        history=[]
        for i,row in g.sort_values('batch_id').iterrows():
            if history:
                reference=sum(history)/len(history)
                t.loc[i,'reference_mean']=reference; t.loc[i,'conditional_shift']=row['mean']-reference
            history.append(row['mean'])
    t['interpretation']='post-control diagnostic only; not a pre-bench forecast or normalization'
    return t
