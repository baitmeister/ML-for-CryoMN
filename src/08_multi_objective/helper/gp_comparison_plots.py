"""Saved-evidence-only GP comparisons in the production publication theme."""
from pathlib import Path
import json
import textwrap
import numpy as np
import pandas as pd
from .batch_gp import NAMES
from . import campaign_plots as style
from .plot_reporting import write_plot
import matplotlib.pyplot as plt


def metrics(table):
    rows=[]
    if 'actual' not in table:return pd.DataFrame()
    t=table.loc[table.actual.notna()&~table.is_control.astype(bool)]
    for keys,g in t.groupby(['evidence','strategy','stage','evaluation_subset','batch_id']):
        e=g.predicted-g.actual
        r=g.actual.corr(g.predicted,method='spearman') if len(g)>1 and g.actual.nunique()>1 and g.predicted.nunique()>1 else np.nan
        rows.append(dict(zip(['evidence','strategy','stage','evaluation_subset','batch_id'],keys),n=len(g),mae=abs(e).mean(),bias=e.mean(),
            rmse=float(np.sqrt(np.mean(e**2))),r2=(1-float(np.sum(e**2))/float(np.sum((g.actual-g.actual.mean())**2))) if g.actual.nunique()>1 else np.nan,rank_correlation=r,future_coverage=((g.actual>=g.future_lower)&(g.actual<=g.future_upper)).mean(),future_width=(g.future_upper-g.future_lower).mean()))
    return pd.DataFrame(rows)


def empty(ax,message):
    ax.set_axis_off();ax.text(.5,.5,textwrap.fill(message,70),transform=ax.transAxes,ha='center',va='center',color=style.GRAY)


def canvas(title,n=1):
    style.style();fig,axes=plt.subplots(1,n,figsize=(max(10,n*3.1),5.8),squeeze=False)
    fig.suptitle(title,x=.06,ha='left',fontsize=14)
    fig.subplots_adjust(left=.30,right=.96,top=.82,bottom=.22,wspace=.55)
    return fig,axes[0]


def render(directory):
    directory=Path(directory);table=pd.read_csv(directory/'predictions.csv')
    if 'evaluation_subset' not in table:table['evaluation_subset']='full_slate'
    if 'actual' not in table:table['actual']=np.nan
    summary=metrics(table);summary.to_csv(directory/'metrics.csv',index=False)
    context='; '.join(sorted(table.evidence.unique()))+' | percentage points; not an automatic model decision'
    paths=[]
    fig,axes=canvas('GP strategy summary | before-batch evidence',4)
    s=summary.loc[(summary.stage=='before_batch')&(summary.evaluation_subset=='full_slate')] if not summary.empty else summary
    if s.empty:
        for ax in axes:empty(ax,'Outcomes unavailable. Forecasts are frozen; predictive accuracy cannot yet be evaluated.')
    else:
        # Equal-batch summaries, not an independence assumption about rows.
        grouped=s.groupby('strategy',sort=False).mean(numeric_only=True)
        labels=['\n'.join(textwrap.wrap(NAMES[n],31)) for n in grouped.index]
        for ax,col,title,color in zip(axes,['mae','bias','future_coverage','future_width'],['Average error','Signed bias','Future coverage','Future interval width'],[style.PINK_EDGE,style.PINK_EDGE,style.GREEN,style.BLUE]):
            y=np.arange(len(grouped));ax.scatter(grouped[col],y,color=color)
            ax.set_yticks(y,labels if ax is axes[0] else ['']*len(y));ax.invert_yaxis();ax.set_title(title,fontsize=10)
            if col=='bias':ax.axvline(0,color=style.GRAY,lw=.8)
            if col=='future_coverage':ax.axvline(.95,color=style.GRAY,ls='--');ax.set_xlim(0,1.03)
        fig.text(.06,.08,'Equal weight per evaluated Group. Detailed CSV includes within-batch ranking, sample counts and Group IDs.',fontsize=9)
    fig.text(.06,.025,context,fontsize=8)
    paths+=write_plot(fig,summary,directory,'gp_strategy_summary',context)

    fig,axes=canvas('GP strategies by Group | errors and within-batch ranking',2)
    if s.empty:
        for ax in axes:empty(ax,'No evaluated outcomes yet')
    else:
        for i,(name,g) in enumerate(s.groupby('strategy',sort=False)):
            for ax,col in zip(axes,['mae','rank_correlation']):
                ax.plot(g.batch_id.str.replace('ROUND_',''),g[col],marker=['o','s','^','D','v'][i%5],ls=['-','--',':','-.','-'][i%5],label=NAMES[name])
        axes[0].set_ylabel('Mean absolute error (percentage points)');axes[1].set_ylabel('Within-batch rank correlation');axes[1].set_ylim(-1.05,1.05)
        for ax in axes:ax.set_xlabel('Group')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',fontsize=7,ncol=2)
    paths+=write_plot(fig,s,directory,'gp_strategy_by_batch',context)

    matched=table.loc[table.evaluation_subset.eq('matched_nonreference')&table.actual.notna()]
    strategies=list(matched.strategy.unique()) if not matched.empty else ['batch']
    style.style()
    fig,grid=plt.subplots(len(strategies),2,figsize=(12,4*len(strategies)+1.5),squeeze=False)
    fig.subplots_adjust(left=.10,right=.95,top=.85,bottom=.14,hspace=.75,wspace=.4)
    fig.suptitle('Reference conditioning | same non-reference outcomes',x=.08,ha='left',fontsize=14)
    for rowidx,name in enumerate(strategies):
        axes=grid[rowidx]
        model_rows=matched.loc[matched.strategy.eq(name)]
        if model_rows.empty:
            for ax in axes:empty(ax,'No eligible reference-conditioned evaluation. Conditioning observations are not test outcomes.')
            continue
        k=['strategy','batch_id','formulation_id']
        pairs=model_rows.loc[model_rows.stage.eq('before_batch')].merge(
            model_rows.loc[model_rows.stage.eq('reference_conditioned')],on=k,suffixes=('_before','_after'),validate='one_to_one')
        for _,r in pairs.iterrows():
            axes[0].plot([0,1],[abs(r.predicted_before-r.actual_before),abs(r.predicted_after-r.actual_after)],color=style.PINK_EDGE,alpha=.35,marker='o',ms=3)
            axes[1].plot([0,1],[r.future_upper_before-r.future_lower_before,r.future_upper_after-r.future_lower_after],color=style.BLUE,alpha=.35,marker='o',ms=3)
        for ax in axes:
            ax.set_xticks([0,1],['Before batch','Reference-\nconditioned'])
            ax.set_xlim(-.15,1.15)
        axes[0].set_title('\n'.join(textwrap.wrap(NAMES[name],48)),fontsize=10,loc='left')
        axes[0].set_ylabel('Absolute error (percentage points)');axes[1].set_ylabel('Future interval width (percentage points)')
    fig.text(.08,.04,'Each line compares the same recipe under the same GP. Correlated within Groups; not independent replications.',fontsize=9)
    paths+=write_plot(fig,matched,directory,'reference_conditioning_comparison',context)

    fig,axes=canvas('Model-estimated shared batch effects | not measured cell health')
    try:offsets=pd.read_csv(directory/'offsets.csv')
    except (FileNotFoundError,pd.errors.EmptyDataError):offsets=pd.DataFrame()
    ax=axes[0]
    shown=offsets.loc[offsets.strategy.isin(['batch','control'])].copy() if not offsets.empty else offsets
    if shown.empty:empty(ax,'Batch-shift evidence unavailable')
    else:
        y=np.arange(len(shown));ax.errorbar(shown.offset,y,xerr=1.96*shown.sd,fmt='o',color=style.BLUE,capsize=2)
        fig.set_size_inches(12, max(6, len(shown)*.6+2))
        fig.subplots_adjust(left=.46)
        labels=[f"{r.batch_id} | "+'\n'.join(textwrap.wrap(NAMES[r.strategy],40))+f"\nControls: {r.controls}; ordinary: {r.ordinary}" for r in shown.itertuples()]
        ax.set_yticks(y,labels,fontsize=max(5,9-len(shown)//12));ax.axvline(0,color=style.GRAY,ls='--');ax.set_xlabel('Shared batch offset, percentage points (95% conditional interval)')
    fig.text(.06,.065,'Empirical Bayes: fitted-parameter uncertainty is not integrated. Sparse-control estimates can be prior-sensitive.',fontsize=8)
    paths+=write_plot(fig,shown,directory,'batch_effect_estimates',context)

    before=table.loc[table.stage.eq('before_batch')&table.evaluation_subset.eq('full_slate')&~table.is_control.astype(bool)].copy()
    slate_path=directory/'slate.csv'
    ranks={}
    if slate_path.exists():
        slate=pd.read_csv(slate_path)
        if 'selection_rank' in slate:ranks=slate.set_index('formulation_id').selection_rank.to_dict()
    # One model and one Group per figure, with a common scale within each Group.
    for batch,cohort in before.groupby('batch_id',sort=True):
        low=float(np.nanmin(cohort[['future_lower','predicted','actual']].to_numpy()))
        high=float(np.nanmax(cohort[['future_upper','predicted','actual']].to_numpy()))
        pad=max(3.,(high-low)*.06)
        for name,g in cohort.groupby('strategy',sort=False):
            g=g.copy()
            g['selection_rank']=g.formulation_id.map(ranks)
            g=g.sort_values(['selection_rank','formulation_id'],na_position='last').reset_index(drop=True)
            g['plot_position']=np.arange(1,len(g)+1)
            style.style()
            fig,ax=plt.subplots(figsize=(11.5,6.5))
            fig.subplots_adjust(left=.10,right=.97,top=.73,bottom=.27)
            fig.suptitle('\n'.join(textwrap.wrap(NAMES[name],88)),x=.10,ha='left',fontsize=13,y=.97)
            fig.text(.10,.855,batch.replace('ROUND_','Group ')+' | before-batch predictions',fontsize=10)
            x=g.plot_position.to_numpy()
            ax.vlines(x,g.future_lower,g.future_upper,color=style.BLUE,alpha=.6,lw=2,label='95% future-observation interval')
            ax.scatter(x,g.predicted,color=style.BLUE,s=28,zorder=3,label='Predicted viability')
            measured=g.actual.notna()
            if measured.any():ax.scatter(x[measured],g.loc[measured,'actual'],color=style.RED,marker='D',s=30,zorder=4,label='Measured viability')
            labels=[(f'Rank {int(r.selection_rank)}\n' if pd.notna(r.selection_rank) else '')+str(r.formulation_id) for r in g.itertuples()]
            ax.set_xticks(x,labels,rotation=45,ha='right',fontsize=8)
            ax.set_xlim(.4,len(g)+.6);ax.set_ylim(low-pad,high+pad)
            ax.set_ylabel('Viability (%)')
            fig.legend(*ax.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.095,.775),ncol=3,fontsize=8,frameon=False)
            fig.text(.10,.055,'Same vertical scale across models for this Group. Intervals include assumed batch/measurement variation where modeled.',fontsize=8)
            fig.text(.10,.025,'Measured outcomes unavailable.' if not measured.any() else 'Measurements are unchanged; errors are evaluated against the displayed predictions.',fontsize=8,color=style.GRAY)
            paths+=write_plot(fig,g,directory,f'gp_predictions_{batch.lower()}_{name}',context)
    # Superseded overlay exports are renderer-owned; immutable forecasts are untouched.
    for suffix in ('.png','_source.csv','_metadata.json'):
        old=directory/('gp_prediction_comparison'+suffix)
        if old.exists():old.unlink()
    notes=['# GP strategy decision report','',context,'',
        'No model is automatically promoted. Current GP and Proposed GP labels identify data and modifications.',
        'Error and bias are percentage points; rank correlation measures within-Group ordering. Coverage and width refer to future-observation intervals.',
        'Historical results are retrospective. Pre-outcome challenger forecasts were made after the official slate was selected.',
        'Control calibration with fewer than two historical reference batches is insufficiently supported; do not infer validated correction from it.',
        'Hyperparameters use bounded maximum likelihood. Intervals do not integrate hyperparameter uncertainty.',
        'Reference-conditioned results exclude the conditioning observations and use matched test rows. They do not retrospectively change selection.',
        '', '## Reading future coverage and width', '',
        '**Future coverage** is the percentage of later measured formulation–batch outcomes inside the nominal 95% future-observation intervals. About 95% is the intended long-run target, not a guarantee for one small Group. Summary values average the per-Group coverage.',
        '**Future width** is the average upper minus lower interval bound, in viability percentage points. An interval from 40% to 80% has width 40 points. It describes uncertainty, not prediction error. Summary values average the per-Group widths.',
        'These intervals include residual noise and, for batch-effect models, new-batch variation. They concern the modeled formulation–batch mean, not every technical replicate. Wider intervals can improve coverage without improving the central prediction; assess coverage and width together.',
        '', '## Summary', '']
    if not s.empty:
        notes+=['| Strategy | Groups | Observations | MAE | Bias | Mean within-Group rank | Future coverage | Future width |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,g in s.groupby('strategy',sort=False):notes.append(f'| {NAMES[name]} | {g.batch_id.nunique()} | {int(g.n.sum())} | {g.mae.mean():.2f} | {g.bias.mean():.2f} | {g.rank_correlation.mean():.2f} | {g.future_coverage.mean():.1%} | {g.future_width.mean():.2f} |')
    else:notes.append('Outcomes are not available; no accuracy claim is possible.')
    notes+=['','## Figures','']+[f'- [{p.stem}]({p.name})' for p in paths if p.suffix=='.png']
    (directory/'README.md').write_text('\n'.join(notes)+'\n')
    return paths
