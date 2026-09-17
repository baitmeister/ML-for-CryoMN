"""Selected scientific renderers. All exports use transparent 300 dpi PNGs."""
from __future__ import annotations
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from .plot_theme import apply_plot_theme
from .evaluation_metrics import compute_observed_pareto_front

BLUE, RED, GREEN, GRAY = '#0072B2', '#D55E00', '#009E73', '#666666'
PINK, PINK_EDGE = '#E8BDD1', '#A64D79'
TITLES = {'B': 'Campaign timeline | error and coverage', 'C': 'Surrogate trust | frozen predictions'}

def style():
    apply_plot_theme('legacy_inspired_publication')
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'axes.grid': False, 'figure.facecolor': 'none', 'axes.facecolor': 'none', 'savefig.transparent': True})

def header(ax, title, note='', handles=None):
    ax.set_axis_off()
    ax.text(0, .92, title, transform=ax.transAxes, va='top', fontsize=11, weight='bold')
    if handles:
        ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, .05),
                  borderaxespad=0, ncol=len(handles), fontsize=9, frameon=False)
    elif note:
        ax.text(0, .20, note, transform=ax.transAxes, va='bottom', fontsize=9, color=GRAY)
    ax.plot([0, 1], [0, 0], color='#cccccc', lw=.6, transform=ax.transAxes, clip_on=False)

def finish_axis(ax, grid='y'):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis=grid, color='#dddddd', linewidth=.55)
    ax.margins(y=.12)

def line_handle(label, color, marker, ls='-'):
    return Line2D([], [], label=label, color=color, marker=marker, linestyle=ls, lw=1.5)

def round_series(table, series, rounds):
    data = table.loc[table.series.eq(series)].set_index('round_number')
    return data.reindex(rounds)

def timeline_figure(table):
    style()
    n = 3
    fig = plt.figure(figsize=(11.5, 11))
    gs = fig.add_gridspec(2*n, 1, height_ratios=[.55, 2]*n,
                          left=.10, right=.89, bottom=.13, top=.89, hspace=.20)
    heads = [fig.add_subplot(gs[2*i]) for i in range(n)]
    axes = [fig.add_subplot(gs[2*i+1]) for i in range(n)]
    numbers = pd.to_numeric(table.round_number, errors='coerce').dropna()
    rounds = list(range(int(numbers.min()), int(numbers.max())+1)) if len(numbers) else [1]
    individual = table.loc[table.series.eq('individual_formulation')]
    jitter = ((individual.groupby('round_number').cumcount() % 7)-3)*.025
    axes[0].scatter(individual.round_number+jitter, individual.value, color=GRAY,
                    alpha=.45, s=17, edgecolors='none')
    med = round_series(table, 'median_iqr', rounds)
    axes[0].plot(rounds, med.value, color=BLUE, marker='o', lw=1.5)
    axes[0].vlines(rounds, med.lower, med.upper, color=BLUE, lw=2)
    if 'strategy_decision_id' in table:
        changes=table.loc[table.strategy_decision_id.ne('historical_current')].sort_values('round_number').drop_duplicates('strategy_decision_id')
        for row in changes.itertuples():
            for ax in axes:ax.axvline(row.round_number,color=GRAY,ls=':',lw=.8)
            axes[0].text(row.round_number,102,str(row.strategy_decision_id),rotation=90,va='top',fontsize=7)
    axes[0].set_ylabel('Observed viability (%)')
    axes[0].set_ylim(-3,105)
    header(heads[0], 'Experimental viability', handles=[
        line_handle('Formulation mean', GRAY,'o','None'),line_handle('Median; bars = IQR',BLUE,'o')])
    error_ax = axes[1]
    for key,color,marker,ls in [('mae',BLUE,'o','-'),('bias',RED,'s','--')]:
        data=round_series(table,key,rounds)
        error_ax.plot(rounds,data.value,color=color,marker=marker,ls=ls,lw=1.6)
    error_ax.axhline(0,color=GRAY,lw=.7)
    error_ax.set_ylabel('Error (percentage points)')
    err_handles=[line_handle('MAE',BLUE,'o'),line_handle('Bias',RED,'s','--')]
    cover_ax=error_ax.twinx()
    coverage=round_series(table,'interval_95_coverage',rounds)
    cover_ax.plot(rounds,coverage.value*100,color=GREEN,marker='^',ls='-.',lw=1.6)
    cover_ax.axhline(95,color=GRAY,ls=':',lw=1)
    cover_ax.set_ylim(0,105)
    cover_ax.set_ylabel('Interval coverage (%)')
    cov_handles=[line_handle('Observed coverage',GREEN,'^','-.'),line_handle('95% target',GRAY,'',':')]
    header(heads[1], 'Prediction error and interval reliability',
           handles=err_handles+cov_handles)
    cover_ax.spines['top'].set_visible(False)
    cover_ax.grid(False)
    intact=round_series(table,'intact_pass_proportion',rounds)
    axes[-1].plot(rounds,intact.value,color=GREEN,marker='o',lw=1.6)
    axes[-1].vlines(rounds,intact.lower,intact.upper,color=GREEN,lw=1.6)
    axes[-1].set_ylim(-.04,1.15)
    axes[-1].set_ylabel('Intact pass proportion')
    for r,row in intact.iterrows():
        if pd.notna(row.value):
            axes[-1].text(r,min(row.upper+.05,1.08),f'{int(row.count_pass)}/{int(row.count_pass+row.count_fail)}',
                          ha='center',fontsize=8,color=GRAY)
    header(heads[-1],'Intact-patch formation',note='Green: pass proportion | bars: Wilson 95% CI | labels: passed / tested')
    groups=[]
    for provenance,g in table.groupby('provenance',sort=False):
        rr=sorted(g.round_number.dropna().astype(int).unique())
        if rr:
            label={'formal_frozen':'Formal frozen','reconstructed':'Reconstructed',
                   'migration_frozen_supplementary':'Supplementary frozen','current_proposal':'Proposal'}.get(provenance,provenance)
            groups.append(f'{label}: '+', '.join('R'+str(r) for r in rr))
    for ax in axes:
        ax.set_xticks(rounds, ['R'+str(r) for r in rounds])
        ax.set_xlim(min(rounds)-.5,max(rounds)+.5)
        ax.set_xlabel('Campaign round' if ax is axes[-1] else '')
        finish_axis(ax)
    fig.suptitle('B  '+TITLES['B'],x=.10,ha='left',y=.98,fontsize=15)
    fig.text(.10,.944,' | '.join(groups),fontsize=8.5,color=GRAY)
    count=int(table.loc[table.series.eq('mechanical_measurement_count'),'value'].sum())
    fig.text(.10,.072,'MAE: mean absolute error. Bias: prediction minus observation. Coverage: fraction inside the stated interval.',fontsize=9)
    fig.text(.10,.043,f'Completed critical-load measurements: {count}. '+('Hypervolume is not estimable without paired mechanics evidence.' if count==0 else 'Mechanics count is not a surrogate-performance metric.'),fontsize=9,color=GRAY)
    return fig

def round_distribution(ax, frame, column, kind='summary', color=BLUE, fill=None):
    rounds=sorted(frame.round_id.dropna().unique())
    for i,r in enumerate(rounds,1):
        vals=pd.to_numeric(frame.loc[frame.round_id.eq(r),column],errors='coerce')
        vals=vals[np.isfinite(vals)].to_numpy()
        if not len(vals):
            continue
        if kind=='violin' and len(vals)>=5 and len(np.unique(vals))>=3:
            violin=ax.violinplot([vals],positions=[i],widths=.72,showextrema=False,bw_method='scott')
            for body in violin['bodies']:
                body.set_facecolor(fill or color)
                body.set_edgecolor(color)
                body.set_alpha(.85 if fill else .20)
        if kind in ('summary', 'violin'):
            lo,med,hi=np.quantile(vals,[.25,.5,.75])
            ax.vlines(i,lo,hi,color=color,lw=4)
            ax.scatter([i],[med],color='white',edgecolors=color,s=24,zorder=4)
        offsets=((np.arange(len(vals))%9)-4)*.033
        ax.scatter(i+offsets,vals,s=19,color=color,alpha=.75 if fill else .60,zorder=3)
    ax.set_xticks(range(1,len(rounds)+1),['R'+str(int(r.removeprefix('ROUND_'))) for r in rounds])
    ax.set_xlabel('Formal campaign round')
    if rounds: ax.set_xlim(.5,len(rounds)+.5)
    if not rounds:
        ax.text(.5,.5,'No eligible observations',transform=ax.transAxes,ha='center')

def trust_figure(table, metrics, mechanics_count=0):
    style()
    fig=plt.figure(figsize=(12,9.5))
    gs=fig.add_gridspec(5,2,height_ratios=[.6,3,.45,.6,3],left=.09,right=.97,top=.89,bottom=.14,hspace=.23,wspace=.30)
    heads=[fig.add_subplot(gs[r,c]) for r,c in [(0,0),(0,1),(3,0),(3,1)]]
    axes=[fig.add_subplot(gs[r,c]) for r,c in [(1,0),(1,1),(4,0),(4,1)]]
    v=table.loc[table.panel.eq('viability_prediction')]
    coverage=table.loc[table.panel.eq('coverage')]
    intact=table.loc[table.panel.eq('intact_probability')]
    width=pd.to_numeric(v.interval_width,errors='coerce')
    valid_width=width[np.isfinite(width)]
    round_distribution(axes[0],v,'signed_error','violin',color=PINK_EDGE,fill=PINK)
    axes[0].axhline(0,color=GRAY,ls='--',lw=1)
    axes[0].set_ylabel('Prediction - observation (pp)')
    header(heads[0],'Signed error by round',note='All errors shown | dots: evidence; shapes: smoothed density')
    round_distribution(axes[1],v,'interval_width')
    axes[1].set_ylabel('95% interval width (pp)\nWider = less precise prediction')
    header(heads[1],'How uncertain was the model?',
           note=f'n={len(valid_width)} | dots: widths; bars: IQR; white: median')
    ax=axes[2]
    ax.plot([0,1],[0,1],color=GRAY,ls='--',lw=1)
    ax.plot(coverage.nominal_coverage,coverage.empirical_coverage,color=GREEN,marker='^',lw=1.7)
    ax.fill_between(coverage.nominal_coverage.to_numpy(float),coverage.coverage_lower.to_numpy(float),
                    coverage.coverage_upper.to_numpy(float),color=GREEN,alpha=.12)
    ax.set(xlim=(.45,1),ylim=(0,1.04),xlabel='Nominal interval coverage',ylabel='Observed interval coverage')
    header(heads[2],'Do intervals cover later measurements?',note='Dashed: ideal | shading: descriptive Wilson 95% intervals')
    ax=axes[3]
    for outcome,marker,color,label in [(0,'x',RED,'Fail'),(1,'o',GREEN,'Pass')]:
        group=intact.loc[intact.binary_outcome.eq(outcome)]
        jitter=((np.arange(len(group))%7)-3)*.025
        ax.scatter(group.prediction,outcome+jitter,marker=marker,
                   color=RED,s=26,alpha=.70)
    ax.set(xlim=(-.03,1.03),ylim=(-.3,1.3),xlabel='Frozen intact-pass probability')
    ax.set_yticks([0,1],['Fail','Pass'])
    header(heads[3],'Intact outcomes',note=f'{int(intact.binary_outcome.eq(1).sum())} pass / {int(intact.binary_outcome.eq(0).sum())} fail | circles: pass; crosses: fail')
    for ax in axes: finish_axis(ax)
    for ax,available in zip(axes,[len(v)>0,len(valid_width)>0,coverage.empirical_coverage.notna().any(),len(intact)>0]):
        if not available:
            ax.clear();ax.set_axis_off()
            ax.text(.5,.5,'No eligible formal evidence',ha='center',transform=ax.transAxes,color=GRAY)
    formal=metrics.loc[metrics.scope.eq('pooled_formal') & metrics.endpoint.eq('viability_percent')]
    note='No formal viability metrics available.'
    if len(formal) and float(formal.iloc[0].n_evaluated)>0:
        row=formal.iloc[0]
        note=f"Formal frozen cohort | n={int(row.n_evaluated)} | MAE {row.mae:.1f} pp | bias {row.bias:+.1f} pp | R² {row.r2:.2f} | 95% coverage {100*row.interval_95_coverage:.1f}%"
    fig.suptitle('C  '+TITLES['C'],x=.09,ha='left',y=.98,fontsize=15)
    fig.text(.09,.93,note,fontsize=10)
    fig.text(.09,.065,'pp = percentage points. Interval widths are model-reported uncertainty, not observed errors.',fontsize=9,color=GRAY)
    fig.text(.09,.038,f'Critical-load observations: {mechanics_count}. '+('Mechanical surrogate cannot yet be evaluated.' if mechanics_count==0 else 'Mechanical diagnostics are outside these viability views.'),fontsize=9,color=GRAY)
    return fig

def decision_figure(table):
    style()
    fig=plt.figure(figsize=(17,max(7, 2.8 + .60*len(table))))
    gs=fig.add_gridspec(2,5,height_ratios=[.55,8],width_ratios=[2.5,2,1.65,1.25,2.2],
                        left=.03,right=.985,bottom=.14,top=.88,hspace=.03,wspace=.10)
    names=['Candidate and ingredients','Viability prediction','Intact probability','Mechanical test','Selection reason']
    notes=['Full IDs and concentrations in CSV','Mean and nominal 95% interval','Active empirical combination','Stored rank; no reordering','Plain-language policy attributes']
    axes=[]
    for i in range(5):
        header(fig.add_subplot(gs[0,i]),names[i],notes[i])
        ax=fig.add_subplot(gs[1,i],sharey=axes[0] if axes else None)
        axes.append(ax)
        ax.set_ylim(len(table)-.5,-.5)
        ax.set_yticks([])
        ax.spines[['left','top','right','bottom']].set_visible(False)
        for j in range(len(table)):
            ax.axhspan(j-.5,j+.5,color=GRAY,alpha=.065 if j%2==0 else 0,zorder=0)
        ax.set_xlim(0,1)
        ax.set_xticks([])
    known=table.public_viability_mean.notna()
    lo=table.public_viability_mean-1.96*table.public_viability_std
    hi=table.public_viability_mean+1.96*table.public_viability_std
    finite=pd.concat([lo,hi,table.public_viability_mean]).dropna()
    axes[1].set_xlim(min(0,float(finite.min())-3) if len(finite) else 0,max(100,float(finite.max())+3) if len(finite) else 100)
    axes[1].set_xticks([0,25,50,75,100]);axes[1].set_xlabel('Viability (%)')
    axes[2].set_xlim(-.03,1.03);axes[2].set_xticks([0,.5,1]);axes[2].set_xlabel('Probability')
    for j,row in table.iterrows():
        desc='\n'.join(textwrap.wrap(str(row.formulation_description),width=35))
        axes[0].text(.025,j,(f'#{int(row.selection_rank):02d}  ' if pd.notna(row.selection_rank) else str(row.candidate_id)[:12]+'  ')+desc,va='center',fontsize=9)
        if known.iloc[j]:
            axes[1].errorbar(row.public_viability_mean,j,xerr=1.96*row.public_viability_std if pd.notna(row.public_viability_std) else None,
                            fmt='o',color=BLUE,ecolor=BLUE,capsize=2,elinewidth=1,markersize=5)
        else:
            axes[1].text(.05,j,'Unknown',transform=axes[1].get_yaxis_transform(),va='center',fontsize=9,color=GRAY)
        prob=row.empirical_combination_pass_probability
        if pd.notna(prob):
            axes[2].scatter(prob,j,marker='D' if row.prior_only else 'o',facecolors='white' if row.prior_only else GREEN,edgecolors=GREEN,s=32)
        if row.prior_only:
            axes[2].text(.55,j,'Prior only',va='center',fontsize=8,color=GRAY)
        assignment=row.mechanical_assignment
        color=GREEN if assignment.startswith('Primary') else BLUE if assignment.startswith('Backup') else GRAY
        axes[3].text(.05,j,assignment,va='center',fontsize=10,color=color,weight='bold' if assignment.startswith('Primary') else 'normal')
        axes[4].text(.02,j,'\n'.join(textwrap.wrap(row.selection_reason,32)),va='center',fontsize=9)
    fig.suptitle('D  Next-round decisions | primary tests and ranked backups',x=.03,ha='left',y=.975,fontsize=17)
    fig.text(.03,.925,'All candidates stay in stored selection order. Unknown viability has no plotted numerical position.',fontsize=11,color=GRAY)
    fig.text(.03,.068,'Mechanical execution: intact pass required; use stored rank order within round capacity. Backups replace unavailable primaries.',fontsize=10)
    fig.text(.03,.037,'Hollow diamond = prior-only intact probability (zero effective evidence). Filled circle = empirical evidence available.',fontsize=10,color=GRAY)
    return fig

def publication_figure(timeline, trust, metrics, feasible):
    """Restyle the triptych using actual paired evidence, never the A demo."""
    style()
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(5, 2, height_ratios=[.6, 3, .45, .6, 3],
                          width_ratios=[1.15, 1], left=.08, right=.90,
                          bottom=.14, top=.88, hspace=.23, wspace=.35)
    left_head = fig.add_subplot(gs[0, 0])
    left = fig.add_subplot(gs[1:, 0])
    parity_head = fig.add_subplot(gs[0, 1])
    parity = fig.add_subplot(gs[1, 1])
    progress_head = fig.add_subplot(gs[3, 1])
    progress = fig.add_subplot(gs[4, 1])
    header(left_head, 'Observed feasible trade-off', note='Requires paired viability, load, and an intact pass')
    if feasible.empty:
        left.set_axis_off()
        left.text(.06, .70, 'Mechanical evidence pending', transform=left.transAxes,
                  fontsize=16, weight='bold', color=GRAY, va='top')
        left.text(.06, .54, '0 feasible paired viability + load measurements\n\nObserved Pareto front: not yet estimable\nCampaign hypervolume: not yet estimable',
                  transform=left.transAxes, fontsize=11, color=GRAY, linespacing=1.5, va='top')
        left.text(.06, .30, 'No surrogate predictions or synthetic demo points\nare substituted for experimental evidence.',
                  transform=left.transAxes, fontsize=10, color=GRAY, linespacing=1.5, va='top')
    else:
        labelled = compute_observed_pareto_front(feasible)
        for flag, color, marker, label in [(False, GRAY, '.', 'Dominated'),
                                            (True, GREEN, 'o', 'Nondominated')]:
            group = labelled.loc[labelled.is_pareto.eq(flag)]
            left.scatter(group.viability_percent, group.critical_axial_load_N_per_needle,
                         color=color, marker=marker, label=label)
        left.set(xlabel='Observed viability (%)', ylabel='Observed critical load (N/needle)')
        left_head.legend(*left.get_legend_handles_labels(), frameon=False, loc='lower left', bbox_to_anchor=(0, -.2), ncol=2, fontsize=8)
        finish_axis(left)
    header(parity_head, 'Formal prospective validation', note='Dots: frozen predictions | dashed: perfect prediction')
    viability = trust.loc[trust.panel.eq('viability_prediction')]
    parity.scatter(viability.observed, viability.prediction, color=BLUE, alpha=.65, s=24)
    values = pd.concat([viability.observed, viability.prediction]).dropna()
    limits = [min(-5, values.min()-5), max(105, values.max()+5)] if len(values) else [-5, 105]
    parity.plot(limits, limits, ls='--', lw=1, color=GRAY)
    parity.set(xlim=limits, ylim=limits, xlabel='Observed viability (%)', ylabel='Frozen prediction (%)')
    header(progress_head, 'Experimental batch outcomes', handles=[
        line_handle('Median viability', BLUE, 'o'), line_handle('Intact pass rate', GREEN, 's', '--')])
    numbers = pd.to_numeric(timeline.round_number, errors='coerce').dropna()
    rounds = list(range(int(numbers.min()), int(numbers.max())+1)) if len(numbers) else [1]
    viability_round = round_series(timeline, 'median_iqr', rounds)
    intact_round = round_series(timeline, 'intact_pass_proportion', rounds)
    progress.plot(rounds, viability_round.value, color=BLUE, marker='o', lw=1.5)
    progress.set(xlabel='Campaign round', ylabel='Median viability (%)', ylim=(0, 100))
    progress.set_xticks(rounds, ['R'+str(r) for r in rounds])
    rate_axis = progress.twinx()
    rate_axis.plot(rounds, intact_round.value, color=GREEN, marker='s', ls='--', lw=1.5)
    rate_axis.set(ylabel='Intact pass proportion', ylim=(0, 1.05))
    rate_axis.spines['top'].set_visible(False)
    rate_axis.grid(False)
    finish_axis(parity)
    finish_axis(progress)
    fig.suptitle('E  Publication triptych | current evidence', x=.08, ha='left', y=.98, fontsize=15)
    formal = metrics.loc[metrics.scope.eq('pooled_formal') & metrics.endpoint.eq('viability_percent')]
    if len(formal):
        row = formal.iloc[0]
        fig.text(.08, .93, f'Formal frozen viability | n={int(row.n_evaluated)} | MAE {row.mae:.1f} pp | bias {row.bias:+.1f} pp | 95% coverage {100*row.interval_95_coverage:.1f}%', fontsize=10)
    fig.text(.08, .066, 'pp = percentage points. Batch medians summarize tested formulations, not standalone optimizer improvement.', fontsize=9, color=GRAY)
    fig.text(.08, .037, 'Outcome panel uses two labelled scales. Curve crossings do not have a scientific interpretation.', fontsize=9, color=GRAY)
    return fig


def save_png(fig, path):
    """Export only PNG; transparent canvas and axes retain intentional data fills."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, transparent=True, facecolor='none', edgecolor='none')
    plt.close(fig)
    return path


def mechanical_trace_figure(table, analysis, formulation_number, replicate_id):
    """Render one terminal-method trace with its trigger, endpoint and secant."""
    style()
    fig = plt.figure(figsize=(11.5, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[.9, 5], left=.10,
                          right=.96, top=.88, bottom=.14, hspace=.12)
    head, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax.plot(table.displacement_mm, table.force_N, color=GRAY, lw=.8,
            alpha=.55, label='Recorded trajectory')
    window = table.loc[table.analysis_window]
    ax.plot(window.displacement_mm, window.force_N, color=BLUE, lw=1.8,
            label='Trigger-to-terminal window')
    x0 = analysis['contact_displacement_mm']
    y0 = analysis['trigger_force_N']
    x1 = analysis['terminal_displacement_mm']
    y1 = analysis['endpoint_N_total']
    ax.scatter([x0], [y0], color=GREEN, marker='o', s=48, zorder=4)
    ax.scatter([x1], [y1], color=RED, marker='s', s=48, zorder=4)
    ax.plot([x0, x1], [y0, y1], color=RED, ls='--', lw=1.3,
            label='Apparent secant')
    ax.axvline(x0, color=GREEN, lw=.8, ls=':')
    ax.axvline(x1, color=RED, lw=.8, ls=':')
    ax.set(xlabel='Recorded displacement (mm)', ylabel='Recorded force (N)')
    finish_axis(ax)
    handles = [
        line_handle('Recorded trajectory', GRAY, '', '-'),
        line_handle('Trigger-to-terminal window', BLUE, '', '-'),
        line_handle('Sustained 1 N origin', GREEN, 'o', 'None'),
        line_handle('Force at +0.8 mm', RED, 's', 'None'),
        line_handle('Apparent secant', RED, '', '--'),
    ]
    header(head, f'Formulation {formulation_number}, {replicate_id}', handles=handles)
    fig.suptitle('Mechanical trace | terminal compression method', x=.10,
                 ha='left', y=.98, fontsize=15)
    fig.text(.10, .065,
             f"Terminal force {y1:.3f} N total | apparent secant stiffness "
             f"{analysis['apparent_secant_stiffness_N_per_mm_total']:.3f} N/mm total",
             fontsize=9)
    fig.text(.10, .037,
             'Secant = (terminal force − trigger-origin force) / 0.8 mm. '
             'Markers show interpolated protocol positions.', fontsize=9, color=GRAY)
    return fig


def mechanical_summary_figure(table, context='Completed round'):
    """Show individual replicate evidence and formulation summaries."""
    style()
    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[.8, 5], left=.09,
                          right=.97, top=.88, bottom=.15, hspace=.12, wspace=.28)
    heads = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    panels = [
        ('terminal_force_N_per_needle', 'Terminal force', 'N per loaded needle', BLUE),
        ('apparent_secant_stiffness_N_per_mm_per_needle', 'Apparent secant stiffness',
         'N/mm per loaded needle', RED),
    ]
    positions = {value: index for index, value in enumerate(table.formulation_number.unique(), 1)}
    for head, ax, (column, title, ylabel, color) in zip(heads, axes, panels):
        for formulation, group in table.groupby('formulation_number', sort=False):
            x = positions[formulation]
            values = pd.to_numeric(group[column], errors='coerce').dropna().to_numpy()
            if not len(values):
                ax.annotate('per-needle unavailable', (x, .5),
                            xycoords=('data', 'axes fraction'), ha='center',
                            fontsize=8, color=GRAY, rotation=90)
                continue
            offsets = np.linspace(-.07, .07, len(values)) if len(values) > 1 else np.array([0.])
            ax.scatter(x + offsets, values, color=color, s=42, alpha=.75, zorder=3)
            mean = float(np.mean(values))
            if len(values) >= 2:
                ax.scatter([x], [mean], facecolors='white', edgecolors=color,
                           marker='D', s=58, linewidth=1.4, zorder=4)
                ax.errorbar([x], [mean], yerr=[float(np.std(values, ddof=1))],
                            fmt='none', ecolor=color, capsize=4, lw=1.3, zorder=2)
            ax.annotate(f'n={len(values)}', (x, max(values)), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=8, color=GRAY)
            if bool(group.replicate_lost.any()):
                ax.annotate('replicate lost', (x, min(values)), xytext=(0, -17),
                            textcoords='offset points', ha='center', fontsize=7.5, color=GRAY)
        ax.set_xticks(list(positions.values()), [str(value) for value in positions])
        ax.set(xlabel='Formulation number', ylabel=ylabel)
        finish_axis(ax)
        header(head, title, handles=[
            line_handle('Replicate', color, 'o', 'None'),
            line_handle('Mean; bars = sample SD when n≥2', color, 'D', 'None'),
        ])
    fig.suptitle('Mechanical results | terminal compression method', x=.09,
                 ha='left', y=.98, fontsize=15)
    fig.text(.09, .07, context, fontsize=9)
    fig.text(.09, .04,
             'No error bar is drawn for n=1. Missing specimens are documented and are not imputed.',
             fontsize=9, color=GRAY)
    return fig


def pareto_figure(feasible, excluded, candidates):
    """Real feasible evidence defines the frontier; predictions never do."""
    style()
    fig = plt.figure(figsize=(11.5, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 6], left=.10,
                          right=.97, top=.88, bottom=.15, hspace=.13)
    head, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    x, y = 'viability_percent', 'critical_axial_load_N_per_needle'
    labelled = compute_observed_pareto_front(feasible)
    for flag, color, marker, label in [(False, GRAY, 'o', 'Dominated observation'),
                                      (True, GREEN, 'o', 'Nondominated observation')]:
        group = labelled.loc[labelled.is_pareto.eq(flag)]
        if len(group):
            ax.scatter(group[x], group[y], color=color, marker=marker, s=42, label=label)
    if not labelled.empty:
        frontier = labelled.loc[labelled.is_pareto].sort_values(x)
        ax.plot(frontier[x], frontier[y], color=GREEN, lw=1.3)
    failed = excluded.loc[excluded.get('exclusion_reason', pd.Series(dtype=str)).eq('failed_intact_gate')]
    if len(failed): ax.scatter(failed[x], failed[y], color=RED, marker='x', s=55, label='Failed intact gate')
    unknown = excluded.loc[excluded.get('exclusion_reason', pd.Series(dtype=str)).eq('unknown_intact_status')]
    if len(unknown): ax.scatter(unknown[x], unknown[y], facecolors='none', edgecolors=GRAY, marker='s', s=40, label='Intact status unknown')
    # Only public means are allowed, and no missing objective receives a position.
    cx, cy = 'predicted_viability_percent', 'predicted_critical_axial_load_N_per_needle'
    measured_load = pd.concat([feasible.get(y,pd.Series(dtype=float)),excluded.get(y,pd.Series(dtype=float))])
    if cx in candidates and cy in candidates and np.isfinite(pd.to_numeric(measured_load,errors='coerce')).any():
        public = candidates.copy()
        valid = np.isfinite(pd.to_numeric(public[cx],errors='coerce')) & np.isfinite(pd.to_numeric(public[cy],errors='coerce'))
        for status in ['viability_prediction_status','critical_axial_load_prediction_status']:
            if status in public: valid &= ~public[status].fillna('unknown_unrecorded').astype(str).str.startswith('unknown')
        public = public.loc[valid]
        if len(public):
            ax.scatter(public[cx],public[cy],marker='D',facecolors='none',edgecolors=BLUE,s=50,label='Reportable proposal prediction')
            for _, row in public.iterrows():
                sx = pd.to_numeric(row.get('viability_std'),errors='coerce')
                sy = pd.to_numeric(row.get('critical_axial_load_std'),errors='coerce')
                ax.errorbar(row[cx],row[cy],xerr=1.96*sx if np.isfinite(sx) and sx>=0 else None,
                            yerr=1.96*sy if np.isfinite(sy) and sy>=0 else None,
                            fmt='none',ecolor=BLUE,elinewidth=.8,capsize=2)
    if not ax.collections:
        ax.set_axis_off()
        ax.text(.04,.65,'Mechanical evidence pending',transform=ax.transAxes,fontsize=17,weight='bold',color=GRAY)
        ax.text(.04,.45,'No feasible paired viability + load measurements.\n\nObserved Pareto front: not yet estimable.',transform=ax.transAxes,fontsize=12,color=GRAY)
    else:
        ax.set(xlabel='Viability (%)',ylabel='Critical axial load (N/needle)')
        finish_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    header(head,'Observed trade-off and reportable proposals',note='' if handles else 'Only measured feasible pairs can define an observed frontier')
    if handles: head.legend(handles,labels,loc='lower left',bbox_to_anchor=(0,0),ncol=2,frameon=False,fontsize=9)
    fig.suptitle('A  Feasible Pareto evidence map',x=.10,ha='left',y=.98,fontsize=15)
    fig.text(.10,.93,f'Feasible paired observations: {len(feasible)} | Excluded or incomplete rows: {len(excluded)}'+(' | Frontier not estimable' if feasible.empty else ''),fontsize=10,color=GRAY)
    fig.text(.10,.075,'Connecting lines guide the eye; they do not establish tested intermediate formulations.',fontsize=9,color=GRAY)
    fig.text(.10,.045,'Prediction bars: marginal nominal 95% intervals, not a joint region. Missing evidence is not substituted.',fontsize=9,color=GRAY)
    return fig


def _diagnostic_axes(title, subtitle):
    style()
    fig=plt.figure(figsize=(13,10))
    gs=fig.add_gridspec(5,2,height_ratios=[.7,3,.45,.7,3],left=.09,right=.96,top=.88,bottom=.12,hspace=.25,wspace=.30)
    heads=[fig.add_subplot(gs[r,c]) for r,c in [(0,0),(0,1),(3,0),(3,1)]]
    axes=[fig.add_subplot(gs[r,c]) for r,c in [(1,0),(1,1),(4,0),(4,1)]]
    fig.suptitle(title,x=.09,ha='left',y=.98,fontsize=15)
    fig.text(.09,.93,subtitle,fontsize=10,color=GRAY)
    return fig,heads,axes


def _parity(ax,head,frame,title,unit,color):
    from sklearn.metrics import r2_score, mean_absolute_error
    frame=frame.reindex(columns=['actual','predicted']).apply(pd.to_numeric,errors='coerce')
    frame=frame.loc[np.isfinite(frame).all(axis=1)]
    note='n=0 | No recoverable evaluated observations'
    if len(frame):
        actual,predicted=frame.actual,frame.predicted
        r2=r2_score(actual,predicted) if len(frame)>1 and np.std(actual)>0 else np.nan
        note=f'n={len(frame)} | MAE={mean_absolute_error(actual,predicted):.2f} | R²={r2:.2f}' if np.isfinite(r2) else f'n={len(frame)} | MAE={mean_absolute_error(actual,predicted):.2f} | R²=not estimable'
        low,high=float(frame.min().min()),float(frame.max().max())
        pad=max((high-low)*.08,.01)
        ax.scatter(actual,predicted,color=color,s=27,alpha=.7)
        ax.plot([low-pad,high+pad],[low-pad,high+pad],color=GRAY,ls='--',lw=1)
        ax.set(xlim=(low-pad,high+pad),ylim=(low-pad,high+pad))
    else:
        ax.set_axis_off()
        ax.text(.5,.5,'Evidence unavailable',ha='center',transform=ax.transAxes,color=GRAY)
    ax.set(xlabel='Observed '+unit,ylabel='Cross-validated prediction '+unit)
    header(head,title,note)
    finish_axis(ax)


def _round_trend(ax,frame,series,ylabel):
    from .plot_data import _round_number
    if frame.empty:
        ax.set_axis_off()
        ax.text(.5,.5,'No recoverable paired-round evidence',ha='center',transform=ax.transAxes,color=GRAY)
    else:
        data=frame.copy()
        data['_round']=data.batch_id.map(_round_number)
        data=data.dropna(subset=['_round']).set_index('_round')
        if len(data):
            rounds=range(int(data.index.min()),int(data.index.max())+1)
            data=data.reindex(rounds)
            for column,color,marker,ls in series:
                ax.plot(list(rounds),pd.to_numeric(data[column],errors='coerce'),color=color,marker=marker,ls=ls,lw=1.5)
            ax.set_xticks(list(rounds),['R'+str(r) for r in rounds])
    ax.set(xlabel='Campaign round',ylabel=ylabel)
    finish_axis(ax)


def diagnostic_figures(frames, paired_frames, metrics, r2_frame, context='Current evidence'):
    from sklearn.metrics import accuracy_score, brier_score_loss
    v='viability_percent';m='critical_axial_load_N_per_needle';g='intact_patch_formation_pass'
    empty=pd.DataFrame()
    fig,h,a=_diagnostic_axes('Diagnostics 1  Model validation',context+' | Grouped cross-validation; distinct from formal prospective validation')
    _parity(a[0],h[0],frames.get(v,empty),'All-data viability CV','viability (%)',BLUE)
    _parity(a[1],h[1],frames.get(m,empty),'All-data mechanics CV','load (N/needle)',RED)
    gate=frames.get(g,empty).reindex(columns=['actual','predicted']).apply(pd.to_numeric,errors='coerce').dropna()
    note='n=0 | No recoverable intact classifier evaluations'
    if len(gate):
        for value,marker in [(0,'x'),(1,'o')]:
            group=gate.loc[gate.actual.eq(value)]
            jitter=((np.arange(len(group))%7)-3)*.025
            a[2].scatter(group.predicted,value+jitter,marker=marker,color=GREEN,s=26,alpha=.7)
        note=f'n={len(gate)} | accuracy={accuracy_score(gate.actual,gate.predicted.ge(.5)):.2f} | Brier={brier_score_loss(gate.actual,gate.predicted):.3f}'
    else:a[2].text(.5,.5,'Evidence unavailable',ha='center',transform=a[2].transAxes,color=GRAY)
    header(h[2],'Diagnostic intact classifier',note)
    a[2].set(xlim=(-.03,1.03),ylim=(-.3,1.3),xlabel='Cross-validated pass probability')
    a[2].set_yticks([0,1],['Fail ×','Pass ○']);finish_axis(a[2])
    header(h[3],'Cumulative paired-cohort R²',handles=[line_handle('Viability',BLUE,'o'),line_handle('Mechanics',RED,'s','--')])
    _round_trend(a[3],r2_frame,[('viability_r2',BLUE,'o','-'),('load_r2',RED,'s','--')],'R² (negative values retained)')
    fig.text(.09,.055,'Top/classifier panels: endpoint-specific all-data cohorts. R² trend: paired viability/load cohort only.',fontsize=9,color=GRAY)
    fig2,h,a=_diagnostic_axes('Diagnostics 2  Paired-objective evaluation',context+' | Paired measurements; retrospective comparison reference')
    _parity(a[0],h[0],paired_frames.get(v,empty),'Paired-cohort viability CV','viability (%)',BLUE)
    _parity(a[1],h[1],paired_frames.get(m,empty),'Paired-cohort mechanics CV','load (N/needle)',RED)
    header(h[2],'Normalized hypervolume',note='Higher is better | normalized to final observed frontier')
    _round_trend(a[2],metrics,[('normalized_hypervolume',GREEN,'o','-')],'Normalized hypervolume')
    header(h[3],'Inverted generational distance (IGD)',note='Lower is better | distance to final observed frontier')
    _round_trend(a[3],metrics,[('igd',PINK_EDGE,'s','--')],'IGD in normalized objective space')
    fig2.text(.09,.055,'Reference uses the final observed data in this report, not a known optimum or a proposal-time benchmark.',fontsize=9,color=GRAY)
    return fig,fig2
