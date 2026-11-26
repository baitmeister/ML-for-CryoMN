"""Quantitative plates for the frozen forecasts, observed outcomes and policy replay."""

from textwrap import fill

from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np

from .evidence import F, ROUNDS, V
from .layout import BLUE, GREY, GRID, INK, MINT, ORANGE, PALE, PAPER, PEACH, PURPLE, SKY, TEAL, Plate, ax, key, round_ticks


def cohort(fig,d):
    a = ax(fig,(.095,.30,.77,.405),None)
    palette={'control':'#ced3da','screen only':SKY,'paired':TEAL,'gate failed':PEACH}
    symbols={'control':'C','screen only':'·','paired':'+','gate failed':'×'}
    for p in d['samples'].itertuples():
        y=p.round-13
        a.add_patch(patches.Rectangle((p.rank-.41,y-.39),.82,.78,fc=palette[p.status],ec=INK if p.primary else 'none',lw=1.5))
        a.text(p.rank,y,symbols[p.status],ha='center',va='center',fontsize=14,weight='bold',color='white' if p.status=='paired' else INK)
    a.set(xlim=(.5,12.55),ylim=(6.55,-.55),xticks=range(1,13),yticks=range(7),yticklabels=[f'R{n}' for n in ROUNDS],xlabel='Screening rank on the frozen historical slate')
    for sp in a.spines.values():
        sp.set_visible(False)
    handles=[patches.Patch(fc=c,label=label) for c,label in [(palette['control'],'C  Fixed reference'),(SKY,'·  Intact, viability only'),(TEAL,'+  Both endpoints'),(PEACH,'×  Formation failed')]]
    handles.append(patches.Patch(fc='none',ec=INK,label='Outlined  Original primary'))
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.50,.202),ncol=5,frameon=False,fontsize=10.3)
    fig.text(.884,.51,'12 screens\nper round',fontsize=12,weight='bold',va='center',linespacing=1.35)


def lottery(fig,d):
    a=ax(fig,(.083,.29,.82,.42),'x')
    a.spines['left'].set_visible(False)
    for y,n in enumerate(ROUNDS):
        values=d['lottery'][n]
        s=d['runs'][n]['shortlist_score']
        parts=a.violinplot([values],positions=[y],orientation='horizontal',widths=.71,showmeans=False,showmedians=False,showextrema=False)
        for b in parts['bodies']:
            b.set_facecolor('#bdcdcf')
            b.set_edgecolor('#8da7ad')
            b.set_alpha(.88)
            b.set_linewidth(.7)
        a.scatter(s['random_mean'],y,marker='s',s=42,facecolor=PAPER,edgecolor=INK,linewidth=1.25,zorder=3)
        a.scatter(s['selected_mean'],y,marker='D',s=65,color=ORANGE,edgecolor=PAPER,zorder=4)
        a.text(s['selected_mean']+1.6,y,f"{s['selected_mean']:.1f}",fontsize=9.2,color=ORANGE,va='center',weight='bold')
    a.set(ylim=(6.6,-.6),yticks=range(7),yticklabels=[f'R{n}' for n in ROUNDS],xlim=(0,100),xlabel='Mean measured post-thaw viability of three formulations (%)')
    fig.legend(handles=[patches.Patch(fc='#bdcdcf',label='All 165 triplet means per round'),key(INK,'Expected random mean','s',face=PAPER),key(ORANGE,'Frozen model shortlist','D')],
               loc='center',bbox_to_anchor=(.5,.195),ncol=3,frameon=False)


def rank_recovery(fig,d):
    a=ax(fig,(.14,.258,.46,.435),None)
    ranks=d['ranks']
    for x in range(3):
        for y,r in enumerate(ranks[:,x]):
            ok=r<=3
            a.add_patch(patches.Rectangle((x-.47,y-.43),.94,.86,fc=MINT if ok else PEACH,ec='none'))
            a.text(x,y,str(r),ha='center',va='center',fontsize=15,weight='bold',color=TEAL if ok else ORANGE)
    a.set(xlim=(-.5,2.5),ylim=(6.55,-.55),xticks=range(3),xticklabels=['Forecast #1','Forecast #2','Forecast #3'],yticks=range(7),yticklabels=[f'R{n}' for n in ROUNDS])
    a.tick_params(axis='x',labeltop=True,labelbottom=False)
    for sp in a.spines.values():
        sp.set_visible(False)
    fig.text(.65,.586,f"{int((ranks<=3).sum())} / {ranks.size}",fontsize=29,color=TEAL,weight='bold')
    fig.text(.652,.537,'forecast picks finished in the measured top three.',fontsize=11.7)
    fig.text(.65,.43,f"{int((ranks[:,0]==1).sum())} / {len(ranks)}",fontsize=25,color=BLUE,weight='bold')
    fig.text(.652,.382,'first choices finished first on the tested slate.',fontsize=11.7)
    fig.text(.652,.282,'Each cell is the observed rank among 11 candidates.\nOrange cells finished outside the measured top three.',fontsize=10.6,color=GREY,linespacing=1.5)


def parity(fig,d):
    for j,(data,maximum,title,unit,mae) in enumerate([(d['v'],100,'MSC viability','%',d['replay']['totals']['viability']['mae']),
                                                      (d['f'],1.5,'Terminal force','N/loaded needle',d['replay']['totals']['force']['mae'])]):
        a=ax(fig,(.104+j*.475,.266,.325,.43),'both')
        for old,col,mark in [(True,'#6d8490','o'),(False,ORANGE,'D')]:
            g=data[(data.round_number<19)==old]
            a.scatter(g.prediction_mean,g.observed_mean,s=33 if old else 41,c=col,marker=mark,alpha=.82,edgecolor='white',lw=.6,zorder=3)
        a.plot([0,maximum],[0,maximum],lw=1.05,color=GREY,ls='--',zorder=1)
        a.set(xlim=(0,maximum),ylim=(0,maximum),xlabel=f'Frozen forecast ({unit})',ylabel=f'Measured ({unit})',aspect='equal')
        if j:
            a.set(xticks=[0,.5,1,1.5],yticks=[0,.5,1,1.5])
        a.set_title(f'{title}  ·  n = {len(data)}',fontsize=13,loc='left',weight='bold',pad=9)
        a.text(.035,.97,f"MAE  {mae:.{2 if j==0 else 3}f} {'pp' if j==0 else 'N/needle'}",transform=a.transAxes,ha='left',va='top',fontsize=10.2,bbox=dict(fc=PAPER,ec='none',alpha=.9,pad=2))
    fig.legend(handles=[key('#6d8490','R13–R18'),key(ORANGE,'R19','D'),Line2D([],[],c=GREY,ls='--',label='Perfect agreement')],loc='center',bbox_to_anchor=(.5,.195),ncol=3,frameon=False)


def rank_correlation(fig,d):
    a=ax(fig,(.085,.30,.55,.39))
    round_ticks(a)
    for field,col,marker in [('viability',TEAL,'o'),('force',BLUE,'s')]:
        vals=[d['runs'][n][field]['rho'] for n in ROUNDS]
        a.plot(ROUNDS,vals,c=col,marker=marker,lw=2.1,ms=7,ls='-' if field=='viability' else '--')
    a.axhline(0,c=GREY,lw=.7)
    a.set(ylim=(-.06,1.08),yticks=[0,.25,.5,.75,1],ylabel='Within-round Spearman ρ')
    a.set_xticks(ROUNDS,[f'R{n}\n({d["runs"][n]["force"]["n"]})' for n in ROUNDS])
    a.set_xlabel('Validation round (force pairs scored)')
    fig.text(.70,.674,'What pooling changes',fontsize=14,weight='bold')
    for i,(field,label,col) in enumerate([('viability','Viability',TEAL),('force','Force',BLUE)]):
        y=.59-i*.144
        p=d['replay']['totals'][field]
        avg=np.mean([d['runs'][n][field]['rho'] for n in ROUNDS])
        fig.text(.70,y,f"{label}  ·  n = {p['n']}",fontsize=12,weight='bold',color=col)
        fig.text(.70,y-.052,f"Pooled {p['rho']:.3f}   /   mean within round {avg:.3f}",fontsize=10.7)
    fig.legend(handles=[key(TEAL,'Viability · 11 candidates each round','o',line='-'),key(BLUE,'Force · 4–5 measured candidates','s',line='--')],loc='center',bbox_to_anchor=(.49,.192),ncol=2,frameon=False)


def tolerance(fig,d):
    v=d['v']
    lines=[('prediction_mean','Frozen GP',TEAL,'o','-'),('similar_recipes','Similar recipes',BLUE,'s','--'),('earlier_mean','Earlier-round mean',ORANGE,'^',':')]
    for j,tol in enumerate([10,20]):
        a=ax(fig,(.085+j*.463,.305,.386,.38))
        round_ticks(a)
        for col,name,color,marker,ls in lines:
            hits=(v[col]-v.observed_mean).abs().le(tol).groupby(v.round_number).agg(['sum','count']).cumsum()
            a.plot(hits.index,100*hits['sum']/hits['count'],marker=marker,ms=5.8,color=color,lw=1.9,ls=ls)
        a.set(ylim=(0,100),yticks=[0,25,50,75,100],xlabel='Latest included round')
        a.set_title(f'Within ±{tol} percentage points',fontsize=13,weight='bold',loc='left',pad=11)
        if j==0:
            a.set_ylabel('Cumulative forecasts within tolerance (%)')
        counts=[f'{int((v[col]-v.observed_mean).abs().le(tol).sum())}/77' for col,*_ in lines]
        a.text(.01,-.235,'R19 totals    '+'     ·     '.join(counts),transform=a.transAxes,fontsize=10.2,color=GREY)
    fig.legend(handles=[key(c,n,m,line=ls) for _,n,c,m,ls in lines],loc='center',bbox_to_anchor=(.50,.18),ncol=3,frameon=False)


def bands(d,field,typ):
    rows=d['intervals']
    return rows[rows.endpoint.eq(field)&rows.interval_type.eq(typ)].sort_values('n_round')


def interval_coverage(fig,d):
    proposal=bands(d,V,'proposal')
    future=bands(d,V,'future_batch')
    for j,(column,title,ylab) in enumerate([('coverage','How often do the bands contain the outcome?','Covered (%)'),('mean_width','How wide are the bands?','Mean width (percentage points)')]):
        a=ax(fig,(.083+j*.48,.286,.38,.396))
        round_ticks(a)
        for b,c,m,ls in [(proposal,TEAL,'o','-'),(future,BLUE,'s','--')]:
            a.plot(ROUNDS,b[column]*(100 if j==0 else 1),marker=m,ms=6.5,lw=2,ls=ls,c=c)
        if j==0:
            a.axhline(95,color=ORANGE,ls=':',lw=1.4)
            a.set(ylim=(0,106),yticks=[0,25,50,75,100])
        else:
            a.set(ylim=(0,100),yticks=[0,25,50,75,100])
        a.set(ylabel=ylab,xlabel='Validation round')
        a.set_title(title,fontsize=12.5,loc='left',weight='bold',pad=11)
    fig.legend(handles=[key(TEAL,'Proposal interval','o',line='-'),key(BLUE,'Future-batch interval','s',line='--'),Line2D([],[],c=ORANGE,ls=':',label='Nominal 95% coverage')],loc='center',bbox_to_anchor=(.5,.195),ncol=3,frameon=False)


def interval_score(fig,d):
    for row,field in enumerate([V,F]):
        maximum = d['intervals'].loc[d['intervals'].endpoint.eq(field), 'mean_interval_score'].max()
        for col in (0,1):
            a=ax(fig,(.096+col*.465,.50-row*.259,.378,.164))
            round_ticks(a)
            series=[('proposal',TEAL,'o','-'),('future_batch',BLUE,'s','--')] if row==0 else [('proposal',PURPLE,'D','-')]
            for typ,color,m,ls in series:
                b=bands(d,field,typ)
                vals=b.mean_interval_score
                if col:
                    vals=(vals*b.n_scored).cumsum()/b.n_scored.cumsum()
                a.plot(ROUNDS,vals,c=color,marker=m,ms=4.5,lw=1.6,ls=ls)
            if row==0:
                a.set_title('Individual rounds' if not col else 'Cumulative from R13',fontsize=12.6,loc='left',weight='bold',pad=8)
                a.tick_params(labelbottom=False)
            else:
                a.tick_params(axis='x',labelsize=8.5)
                ns=b.n_scored.cumsum() if col else b.n_scored
                a.set_xticks(ROUNDS,[f'R{n}\n{int(k)}' for n,k in zip(ROUNDS,ns)])
            a.set_ylim(0, maximum*1.10)
            if col==0:
                a.set_ylabel('Viability (pp)' if row==0 else 'Force (N/needle)',fontsize=10)
    fig.legend(handles=[key(TEAL,'Viability proposal','o',line='-'),key(BLUE,'Viability future-batch','s',line='--'),key(PURPLE,'Force proposal','D',line='-')],loc='center',bbox_to_anchor=(.5,.183),ncol=3,frameon=False,fontsize=9.5)


def reference_chart(fig,d):
    a=ax(fig,(.089,.278,.61,.41))
    round_ticks(a)
    ref=d['reference']
    raw=d['raw']
    rid=ref.formulation_id.iloc[0]
    for n in ROUNDS:
        v=raw.loc[raw.batch_id.eq(f'ROUND_{n:03}')&raw.formulation_id.eq(rid)&raw.endpoint.eq(V)].sort_values('replicate_id').value
        a.scatter(n+np.linspace(-.12,.12,len(v)),v,s=22,facecolor=PAPER,edgecolor=GREY,linewidth=.95,zorder=3)
    a.plot(ROUNDS,ref['mean'],color=ORANGE,marker='D',ms=6.4,lw=1.8,label='Fixed reference mean')
    means=d['v'].groupby('round_number').observed_mean.mean()
    a.plot(means.index,means,color=TEAL,marker='o',ms=6.4,lw=1.8,ls='--',label='Nonreference slate mean')
    a.set(ylim=(0,72),ylabel='Measured post-thaw viability (%)',xlabel='Validation round')
    fig.text(.737,.589,'R17',fontsize=19,weight='bold',color=ORANGE)
    fig.text(.737,.508,'16.71% reference mean',fontsize=12.6,weight='bold')
    fig.text(.737,.385,'The 11 other candidates\naveraged 40.82%. The cause\nof the reference shift is open.',fontsize=11.6,linespacing=1.38)
    fig.legend(handles=[key(GREY,'Individual reference readout','o',face=PAPER),key(ORANGE,'Fixed reference mean','D',line='-'), key(TEAL,'Nonreference slate mean','o',line='--')],loc='center',bbox_to_anchor=(.5,.192),ncol=3,frameon=False)


def frontier(fig,d):
    a=ax(fig,(.081,.266,.46,.43),'both')
    p=d['points']
    front=d['frontier'].sort_values(V,ascending=False)
    for early,c,m in [(True,'#84959c','o'),(False,ORANGE,'D')]:
        b=p[(p.n_round.lt(19))==early]
        a.scatter(b[V],b[F],color=c,marker=m,s=32,edgecolors='white',lw=.6,zorder=3)
    ordered=front.sort_values(V)
    x=ordered[V].to_list()
    y=ordered[F].to_list()
    a.step([0,*x,x[-1]],[y[0],*y,0],where='pre',lw=1.15,color=TEAL,zorder=2)
    for i,(_,r) in enumerate(front.iterrows()):
        label=chr(65+i)
        a.scatter(r[V],r[F],s=80,facecolor='none',edgecolor=TEAL,lw=1.1,zorder=4)
        a.annotate(label,(r[V],r[F]),xytext=(5,6),textcoords='offset points',fontsize=10.4,weight='bold',color=TEAL)
        top=.686-i*.113
        fig.text(.58,top,f"{label}   {r[V]:.2f}% viability    /    {r[F]:.3f} N/needle",fontsize=12.1,color=TEAL,weight='bold')
        fig.text(.58,top-.021,fill(r['recipe'],58),fontsize=10.1,va='top',linespacing=1.28)
    a.set(xlim=(0,100),ylim=(0,1.51),xlabel='Measured MSC viability (%)',ylabel='Terminal force (N/loaded needle)')
    fig.legend(handles=[key('#84959c','Earlier paired measurements'),key(ORANGE,'R19 paired measurements','D'),Line2D([],[],c=TEAL,label='Boundary of measured trade-offs')],loc='center',bbox_to_anchor=(.5,.188),ncol=3,frameon=False,fontsize=9.7)


def progress(fig,d):
    r=d['runs']
    hv=np.array([r[n]['hypervolume'] for n in ROUNDS])
    pairs=[r[n]['cumulative_pairs'] for n in ROUNDS]
    a=ax(fig,(.084,.284,.43,.405))
    round_ticks(a)
    a.plot(ROUNDS,hv,marker='o',ms=7,c=TEAL,lw=2.2)
    a.set(ylim=(75.6,87.1),ylabel='Observed frontier area (pp × N/needle)',xlabel='Round (cumulative measured pairs)')
    a.set_xticks(ROUNDS,[f'R{n}\n({p})' for n,p in zip(ROUNDS,pairs)])
    for i in [0,1,2,3,6]:
        a.annotate(f'{hv[i]:.2f}',(ROUNDS[i],hv[i]),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9,color=TEAL)
    b=ax(fig,(.585,.284,.345,.405))
    inc=np.diff(hv)
    b.bar(ROUNDS[1:],inc,color=[TEAL if v>0 else GREY for v in inc],width=.58)
    b.set(xticks=ROUNDS[1:],xticklabels=[f'R{n}' for n in ROUNDS[1:]],ylim=(0,max(inc)*1.21),ylabel='Area added since previous round',xlabel='Validation round')
    for n,y in zip(ROUNDS[1:],inc):
        b.text(n,y+.10,f'+{y:.2f}' if y>.004 else '0',ha='center',fontsize=10.2,weight='bold',color=TEAL if y else GREY)
    a.set_title('Accumulated measured frontier',loc='left',fontsize=13,weight='bold',pad=10)
    b.set_title('Incremental gain',loc='left',fontsize=13,weight='bold',pad=10)


def nominations(fig,d):
    a=ax(fig,(.10,.27,.54,.43),None)
    cols={'paired':MINT,'failed_gate':PEACH,'unresolved':PALE}
    ink={'paired':TEAL,'failed_gate':ORANGE,'unresolved':GREY}
    for row in d['nominations'].itertuples():
        y=row.round-13
        x=row.slot-1
        s=row.status
        a.add_patch(patches.FancyBboxPatch((x-.46,y-.41),.92,.82,boxstyle='round,pad=0,rounding_size=.04',fc=cols[s],ec=GREY if not row.kept else 'none',lw=1.15,ls='--'))
        a.text(x,y,{'paired':'Paired','failed_gate':'Gate failed','unresolved':'Unknown'}[s],ha='center',va='center',fontsize=10.8,weight='bold',color=ink[s])
    a.set(xlim=(-.52,3.52),ylim=(6.55,-.55),xticks=range(4),xticklabels=['Slot 1','Slot 2','Slot 3','Slot 4'],yticks=range(7),yticklabels=[f'R{n}' for n in ROUNDS])
    a.tick_params(axis='x',labeltop=True,labelbottom=False)
    for sp in a.spines.values():
        sp.set_visible(False)
    fig.text(.69,.63,'22 shared nominations',fontsize=14,weight='bold',color=TEAL)
    fig.text(.69,.575,'18 paired outcomes\n4 formation failures',fontsize=12,va='top',linespacing=1.4)
    fig.text(.69,.432,'6 substitutions',fontsize=14,weight='bold',color=GREY)
    fig.text(.69,.384,'All six lack a measured joint outcome.\nDashed cells identify these choices.',fontsize=11.2,va='top',linespacing=1.4)
    fig.text(.10,.197,'Counts refer to the initial four nominations, before backup promotion.',fontsize=10.5,color=GREY)


def questions(fig,d):
    ledger=d['replay']['question_ledger']
    a=ax(fig,(.235,.285,.55,.425),None)
    kinds={'pair_screened_hit':'Pair screened hit','repeat_frontier':'Repeat frontier point'}
    labels=[]
    for i,q in enumerate(ledger):
        opened=int(q['opened_after'][-3:])
        close=int(q['answered_in'][-3:]) if q['answered_in'] else None
        labels.append(f"R{opened}   {kinds[q['kind']]}")
        a.plot([opened,close or 19],[i,i],c=ORANGE if close else '#8caaa9',lw=1.7,zorder=1)
        for n in range(opened, (close or 19)+1):
            if n==close:
                a.scatter(n,i,c=ORANGE,marker='D',s=60,zorder=3)
            elif n==opened:
                a.scatter(n,i,c=TEAL,marker='o',s=38,zorder=3)
            else:
                a.scatter(n,i,facecolors=PAPER,edgecolors=TEAL,s=25,zorder=2)
    a.set(ylim=(len(ledger)-.4,-.6),yticks=range(len(ledger)),yticklabels=labels,xlabel='Evidence available at each round close')
    round_ticks(a)
    a.grid(axis='x',color=GRID,lw=.7)
    a.spines['left'].set_visible(False)
    a.tick_params(axis='y',labelsize=9.5)
    fig.text(.81,.58,str(len(ledger)),fontsize=27,weight='bold',color=TEAL)
    fig.text(.81,.526,'distinct questions',fontsize=11.8)
    fig.text(.81,.412,str(sum(q['answered_in'] is not None for q in ledger)),fontsize=27,weight='bold',color=ORANGE)
    fig.text(.81,.358,'resolved by R19',fontsize=11.8)
    fig.legend(handles=[key(TEAL,'New question'),key(TEAL,'Still open',face=PAPER),key(ORANGE,'Answered','D')],loc='center',bbox_to_anchor=(.5,.185),ncol=3,frameon=False)


def case(fig,d):
    raw=d['raw']
    fid=d['leader']
    slates=d['slates']
    for j,(endpoint,title,unit,forecast,limits) in enumerate([(V,'Viability','%', 'viability_forecast',(0,100)),(F,'Compression','N/loaded needle','force_forecast',(0,.88))]):
        a=ax(fig,(.095+j*.465,.28,.365,.395))
        a.set_xlim(-.45,1.45)
        for i,n in enumerate([18,19]):
            vals=raw.loc[raw.batch_id.eq(f'ROUND_{n:03}') & raw.formulation_id.eq(fid) & raw.endpoint.eq(endpoint)].sort_values('replicate_id').value
            pred=slates[n].set_index('formulation_id').at[fid,forecast]
            a.scatter(i-.18,pred,marker='D',s=48,fc=PAPER,ec=BLUE,lw=1.3,zorder=4)
            if n==19:
                a.annotate(f"{pred:.{2 if j==0 else 3}f}",(i-.18,pred),xytext=(-3,-13),textcoords='offset points',ha='right',color=BLUE,fontsize=9.6)
            if len(vals):
                a.scatter(i+np.linspace(-.06,.09,len(vals)),vals,s=34,color=GREY,alpha=.86,edgecolor=PAPER,lw=.5,zorder=3)
                a.plot([i-.105,i+.15],[vals.mean()]*2,c=TEAL,lw=2.8,zorder=4)
                a.annotate(f"{vals.mean():.{2 if j==0 else 3}f}",(i+.16,vals.mean()),xytext=(3,0),textcoords='offset points',va='center',color=TEAL,weight='bold',fontsize=10)
            else:
                a.text(i,.14,'Force unmeasured',ha='center',color=GREY,fontsize=9.7)
        a.set(ylim=limits,xticks=[0,1],xticklabels=['R18','R19'],ylabel=f'{title} ({unit})')
        a.set_title(title,fontsize=13,weight='bold',loc='left',pad=10)
    fig.legend(handles=[key(GREY,'Individual laboratory readout'),Line2D([],[],color=TEAL,lw=2.8,label='Measured mean'),key(BLUE,'Frozen forecast','D',face=PAPER)],loc='center',bbox_to_anchor=(.5,.195),ncol=3,frameon=False)


def performance_plates():
    forecasts='Source • formal frozen forecast evaluation, R13–R19 · reference controls excluded'
    return [
        Plate('assay-coverage',1,'Which candidates reached the compression assay?','Every cell is one historical formulation–round. Mechanical priorities were frozen before formation was known.',
              'Across 84 screens, 7 were references. Of 77 other candidates, 30 provided paired outcomes, 31 passed without force and 16 failed formation.', 'Source • frozen mechanical assignments and observed viability, formation and terminal force',cohort),
        Plate('three-pick-benchmark',1,'The shortlist exceeded the random mean in all seven rounds','Each shaded distribution contains all ways to choose three of the 11 nonreference candidates tested in that round.',
              'The seven shortlist means averaged 63.72% viability. The same-slate random expectation averaged 42.63%, a gain of 21.08 points.',forecasts+' · exact enumeration of all three-pick combinations',lottery),
        Plate('rank-recovery',1,'Where did the three leading forecasts actually finish?','Read across a row. A measured rank of 1 identifies the highest observed viability in that round’s tested slate.',
              'The first pick finished first in every round. Lower shortlist positions were less reliable, including measured ranks 10 and 11.',forecasts+' · ties broken by frozen screening rank',rank_recovery),
        Plate('endpoint-parity',1,'How close were the frozen numbers to the measurements?','Points above the dashed line exceeded their forecasts. Each point is a nonreference formulation–round mean.',
              'Mean absolute error was 13.10 percentage points for viability and 0.196 N/needle for force on the selectively measured subset.',forecasts+' · terminal_force_08mm_after_1N_v1',parity),
        Plate('spearman-ranking',1,'How reliably did the forecasts order each round?','Spearman correlation compares forecast rank with measured rank. A value of 1 means all ranks agree.',
              'R17 force ranks aligned perfectly across four pairs. Pooling all 30 scored force pairs gives a correlation of 0.569.',forecasts+' · pooled estimates concatenate cases, within-round averages weight the seven rounds equally',rank_correlation),
        Plate('cumulative-tolerance',1,'How often were forecasts close enough in absolute terms?','Each checkpoint divides cumulative hits by cumulative cases from R13. The denominator rises from 11 to 77.',
              'At ±10 points, the frozen GP hit 40/77, similar recipes hit 36/77 and the earlier-round mean hit 26/77. All three use the same scored cases.', 'Source • frozen forecasts and earlier nonreference campaign observations · range-scaled recipe comparison using the registered ingredient space',tolerance),
        Plate('interval-width-coverage',1,'Wider future-batch bands covered more viability outcomes','Eleven frozen nonreference forecasts contribute to each round. Both types of interval have nominal coverage of 95%.',
              'Proposal bands covered 65/77 with mean width 52.51 points. Future-batch bands covered 70/77 with mean width 73.15 points.', 'Source • archived proposal bands and matching pre-outcome production-strategy future-batch bands · interval_scores.csv',interval_coverage),
        Plate('interval-score',1,'Was the extra interval width worth its cost?','The 95% interval score adds width to 40 times the distance outside the band. Lower scores reward sharp, well-calibrated bands.',
              'Cumulative viability scores were 132.61 points for proposal bands and 103.41 for future-batch bands. Counts under the lower plots are force cases.', 'Source • interval_scores.csv, R13–R19 · 11 viability cases per round · cumulative scores weight each scored case equally',interval_score),
        Plate('reference-drift',1,'The same reference showed substantial variation across rounds','Open circles are four readouts per round for the fixed campaign reference. The dashed line follows the other 11 formulations.',
              'Reference viability fell to 16.71% in R17 and reached 42.31% in R19. The reference provides context when interpreting a promising batch.', 'Source • reference_monitoring.csv, raw campaign readouts and formal nonreference viability measurements',reference_chart)]


def decision_plates():
    return [
        Plate('measured-frontier',2,'Four formulations define the measured trade-off','Moving up and right improves both endpoints. Every plotted point has two measurements and an intact pass in the same round.',
              'The highest-viability point reached 82.66% at 0.500 N/needle. The highest-force point reached 1.341 N/needle at 46.45% viability.', 'Source • 30 selectively measured nonreference formulation–round means, R13–R19 · FBS = fetal bovine serum',frontier),
        Plate('frontier-progress',2,'R19 expanded the frontier after two flat rounds','Area is measured above a fixed origin of 0% viability and 0 N/needle. The first observed area is the R13 baseline.',
              'The area held at 82.75 through R16–R18 as paired measurements rose from 17 to 25. Five R19 pairs raised the area to 85.43.', 'Source • cumulative same-batch, intact nonreference pairs in the frozen replay · pp = percentage points',progress),
        Plate('nomination-evidence',2,'Which supervisor choices can the archive actually test?','Each cell follows one of the four mechanical nominations in the replayed new-project draft into that round’s historical results.',
              'Every measured pair among the 28 nominations was also an original lab primary. The archive leaves the benefit of the six substitutions unresolved.', 'Source • seven replayed allocations matched to same-formulation, same-round formation and terminal-force results',nominations),
        Plate('question-ledger',2,'Most questions remained open when their targets were absent','A question opens after validation. The ledger checks the exact formulation again when a later round provides usable evidence.',
              'Only the screened hit nominated after R18 received a next-round paired result. The two newest questions enter the R20 planning brief.', 'Source • as-of-R19 question ledger · a repeated mention of the same target and source round is counted once',questions),
        Plate('observed-followup',2,'One follow-up supplies joint evidence and a reason to repeat','Both rounds used 0.0912 M ectoin, 0.6374 M ethylene glycol and 8.04% fetal bovine serum (rounded).',
              'R19 force readings were 0.316 and 0.684 N/needle. Preparation and cell-batch identifiers were missing, leaving independent repeatability open.', 'Source • frozen forecasts and raw R18–R19 readouts · four viability readouts per round, two compression measurements in R19',case)]
