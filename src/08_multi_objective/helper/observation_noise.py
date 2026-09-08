"""Offline only: replicate-aware estimates; never imported by production models."""
import numpy as np
import pandas as pd

def estimate_noise(observations, floor_sd=1., fallback_sd=5., prior_degrees=4.):
    if min(floor_sd,fallback_sd,prior_degrees)<=0: raise ValueError('Noise settings must be positive')
    rows=[]
    if observations.empty: return pd.DataFrame()
    obs=observations.copy()
    if 'protocol_id' not in obs: obs['protocol_id']='unknown'
    obs['protocol_id']=obs.protocol_id.fillna('unknown')
    keys=['formulation_id','batch_id','endpoint','protocol_id']
    for key,g in obs.groupby(keys,dropna=False):
        values=pd.to_numeric(g.value,errors='coerce').dropna()
        known='preparation_id' in g and g.preparation_id.fillna('').ne('').all()
        if known:
            # Give specimens equal weight inside preparations, then preparations equal weight.
            unit_keys=['preparation_id']
            if 'specimen_id' in g and g.specimen_id.fillna('').ne('').all(): unit_keys+=['specimen_id']
            units=g.assign(value=pd.to_numeric(g.value,errors='coerce')).groupby(unit_keys).value.mean()
            if len(unit_keys)>1: units=units.groupby(level=0).mean()
        else: units=values
        n=len(units) if known else 1
        variance=float(units.var(ddof=1)) if known and n>1 else np.nan
        rows.append(dict(zip(keys,key),mean=float(units.mean()),n_readouts=len(values),n_independent=n,
                         sample_variance=variance,independence_known=known))
    table=pd.DataFrame(rows)
    for _,g in table.groupby(['endpoint','protocol_id'],dropna=False):
        valid=g.sample_variance.notna(); df=(g.loc[valid,'n_independent']-1)
        pooled=float((g.loc[valid,'sample_variance']*df).sum()/df.sum()) if df.sum()>0 else fallback_sd**2
        for i,row in g.iterrows():
            n=row.n_independent
            if pd.isna(row.sample_variance): var=max(pooled,fallback_sd**2); reason='pooled_conservative_fallback'
            else: var=((n-1)*row.sample_variance+prior_degrees*pooled)/(n-1+prior_degrees); reason='replicate_shrinkage'
            table.loc[i,'mean_variance']=max(floor_sd**2,var/n)
            table.loc[i,'noise_source']=reason
    table['mean_sd']=np.sqrt(table.mean_variance); table['policy_version']='offline_replicate_noise_v1'
    return table
