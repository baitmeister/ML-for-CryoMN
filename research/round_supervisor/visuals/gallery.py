"""Write the portable figure navigator and the round-by-round evidence explorer."""

from html import escape
import json

from .evidence import F, ROUNDS, V
from .layout import SECTIONS


LABELS = {
    'round-loop': 'Round loop', 'evidence-cutoff': 'Evidence at decision time',
    'slot-policy': 'Rules for mechanical slots', 'scientist-critic': 'Scientist and critic',
    'assay-coverage': 'Which assays were measured?', 'three-pick-benchmark': 'Three picks versus random',
    'rank-recovery': 'Top-three recovery', 'endpoint-parity': 'Forecast errors',
    'spearman-ranking': 'Ranking within and across rounds', 'cumulative-tolerance': 'Accuracy within tolerance',
    'interval-width-coverage': 'Interval width and coverage', 'interval-score': 'Interval scores',
    'reference-drift': 'Variation in the reference', 'measured-frontier': 'The measured trade-off',
    'frontier-progress': 'Progress by round', 'nomination-evidence': 'Which nominations have outcomes?',
    'question-ledger': 'Questions carried across rounds', 'observed-followup': 'A real follow-up, R18 to R19',
    'round20-budget': 'Four tests in Round 20', 'round20-branches': 'How Round 20 could change direction',
    'prospective-test': 'Testing the value of steering', 'paired-assay': 'How a measured pair is formed',
    'terminal-traces': 'Two actual compression traces',
}


def manifest(plates):
    return [dict(number=i, slug=p.slug, chapter=p.chapter, section=SECTIONS[p.chapter], label=LABELS[p.slug],
                 title=p.title, subtitle=p.subtitle, takeaway=p.takeaway, source=p.source, kind=p.kind,
                 image=f'figures/{i:02d}-{p.slug}.png') for i,p in enumerate(plates,1)]


def explorer_rows(d):
    rows = {}
    for n in ROUNDS:
        r = d['runs'][n]
        next_plan = d['runs'][n+1]['plan'] if n < ROUNDS[-1] else d['replay']['pending_plan']
        slots = next_plan['new_project_draft']['compression_priorities']
        requests = []
        for kind, target in r['next_agenda']['requests'].items():
            viability, force = target[V], target[F]
            observed = f'{viability:.2f}% viability'
            observed += f' and {force:.3f} N/loaded needle' if force is not None else ', force unmeasured'
            role = 'First force measurement' if kind=='pair_screened_hit' else 'Independent repeat'
            slot = next((i for i,s in enumerate(slots, 1) if s['formulation_id']==target['formulation_id']), None)
            requests.append(dict(role=role, recipe=target['recipe'], observed=observed,
                                 slot=f'First-choice slot {slot}' if slot else 'Carried as an open question'))
        rows[str(n)] = dict(n=n, count=r['cumulative_pairs'], area=f"{r['hypervolume']:.2f}",
                            gain=f"{r['shortlist_score']['selected_mean']-r['shortlist_score']['random_mean']:+.2f}",
                            focus=r['next_agenda']['focus'].capitalize()+'.', requests=requests)
    return rows


def write_html(path, items, data, pdf_name):
    nav, panels, options = [], [], []
    for chapter, section in enumerate(SECTIONS):
        nav.append(f'<h2>{escape(section)}</h2><ol>')
        options.append(f'<optgroup label="{escape(section)}">')
        for item in (i for i in items if i['chapter']==chapter):
            n, slug = item['number'], item['slug']
            short, title, takeaway = (escape(item[k]) for k in ['label','title','takeaway'])
            nav.append(f'<li><a href="#{slug}" data-target="{slug}"><small>{n:02d}</small>{short}</a></li>')
            options.append(f'<option value="{slug}">{n:02d} · {short}</option>')
            panels.append(f'''<article class="panel" id="{slug}" aria-label="{title}">
              <h2 class="sr">{title}</h2>
              <figure><a href="{item['image']}" title="Open full-resolution figure"><img src="{item['image']}" loading="{'eager' if n==1 else 'lazy'}" width="1848" height="1050" alt="{title}. {takeaway}"></a><figcaption class="mobile-summary">{takeaway}</figcaption></figure>
              <div class="foot"><details><summary>Figure notes and source</summary><p>{escape(item['subtitle'])} {takeaway}</p><p>{escape(item['source'])}</p></details><a href="{item['image']}">Full-size PNG ↗</a></div>
            </article>''')
        nav.append('</ol>')
        options.append('</optgroup>')
    rows = explorer_rows(data)
    rounds = []
    for n, r in rows.items():
        reqs=''.join(f'''<section class="request"><h4>{escape(q['role'])}</h4><p>{escape(q['recipe'])}</p>
            <p class="secondary">{escape(q['observed'])}</p><p><strong>{escape(q['slot'])}</strong></p></section>''' for q in r['requests'])
        rounds.append(f'''<section class="record" id="after-{n}"><h3>After Round {n}</h3>
          <div class="stats"><div><b>{r['count']}</b><span>Cumulative paired cases</span></div>
          <div><b>{r['area']}</b><span>Observed frontier area¹</span></div>
          <div><b>{r['gain']} pp</b><span>Frozen shortlist gain this round²</span></div></div>
          <h4>Focus for Round {int(n)+1}</h4><p>{escape(r['focus'])}</p><div class="requests">{reqs}</div></section>''')
    round_options=''.join(f'<option value="{n}" {"selected" if n==19 else ""}>After Round {n}</option>' for n in ROUNDS)
    explorer=f'''<article id="round-explorer" class="panel explorer"><h2>What did the supervisor know after each round?</h2>
       <p>Choose a validation cutoff to see the evidence available then and how the next new-project draft allocated its first choices.</p>
       <label for="round">Evidence available</label><select id="round">{round_options}</select>{''.join(rounds)}
       <p class="explainer">¹ From an origin of zero for both endpoints, in percentage points × N/loaded needle. ² Mean viability of the three frozen model picks minus the expected mean of three random picks from the same tested slate. R20 outcomes are pending.</p>
       </article>'''
    source_notes='''<details class="methods"><summary>Data, methods and source files</summary>
      <p>The evaluation covers R13–R19. The historical slates contain 12 formulations per round, including a fixed reference. Every nonreference formulation has a measured viability. The historical lab selected a subset for compression. A paired case requires an intact patch and both endpoint means in one formulation and round.</p>
      <p>The frozen model is a Gaussian process (GP). The three-pick comparison enumerates all 165 triplets on each tested 11-candidate slate. The similar-recipe comparator uses the most recent earlier exact recipe when available. Otherwise it averages the historical means of the three closest distinct recipes in registry-scaled ingredient coordinates. The earlier-round comparator averages earlier nonreference formulation–round means. Both comparators may use observations preceding R13.</p>
      <p>The mechanical endpoint is whole-patch force at 0.8 mm after a sustained 1 N trigger, divided by the number of loaded needles. A point on the measured Pareto frontier has no other observed point at least as high on both endpoints and higher on one. Force evidence is restricted to the tested subset.</p>
      <p>The round supervisor was designed retrospectively. The replay reconstructs decisions in time order and reveals archived outcomes only after each draft. A missed or absent measurement remains unknown. R20 figures show a pending new-project draft and conditional branches. Independent preparations and cell-batch IDs are needed to evaluate repeatability.</p>
      <p class="sources"><a href="../replay.md">Round decisions and endpoint tables</a><a href="../round20_agent_review.json">Recorded scientist–critic review</a><a href="../../../results/multi_objective_v2/reports/prospective/tables/prospective_evaluation_table.csv">Frozen evaluation table</a><a href="../../../results/multi_objective_v2/reports/prospective/tables/interval_scores.csv">Interval scores</a><a href="../README.md">System design and publications</a></p></details>'''
    narration = ''
    recording = path.parent.parent / 'findings_read_aloud.m4a'
    if recording.is_file() and recording.with_suffix('.md').is_file():
        narration = '''<details class="narration"><summary>Listen to the findings</summary>
          <div class="listen"><audio controls preload="none" aria-label="What did the agents add to the CryoMN project?" src="../findings_read_aloud.m4a"></audio>
          <a href="../findings_read_aloud.md">Transcript and comparison tables</a></div></details>'''
    html = TEMPLATE.replace('@@NARRATION@@',narration).replace('@@NAV@@',''.join(nav)).replace('@@OPTIONS@@',''.join(options)).replace('@@PANELS@@',''.join(panels)+explorer)
    html = html.replace('@@COUNT@@',str(len(items))).replace('@@METHODS@@',source_notes).replace('@@PDF@@',escape(pdf_name)).replace('@@IDS@@',json.dumps([i['slug'] for i in items]+['round-explorer']))
    path.write_text(html, encoding='utf-8')


TEMPLATE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light"><title>CryoMN | Visual guide to the round supervisor</title><style>
:root{--ink:#172f3c;--muted:#526977;--teal:#08766b;--line:#d8e1df;--paper:#f7f7f2;--white:#fdfcf9;font-family:Inter,"Helvetica Neue",Arial,sans-serif;color:var(--ink);background:var(--paper)}
*{box-sizing:border-box}body{margin:0}a{color:#075f59}a:focus-visible,select:focus-visible,summary:focus-visible,button:focus-visible{outline:3px solid #a54b23;outline-offset:3px}
header{padding:1.65rem clamp(1rem,4vw,4rem);border-bottom:1px solid var(--line);background:var(--white);display:flex;justify-content:space-between;align-items:center;gap:1rem}
h1{font-size:clamp(1.35rem,2.75vw,2.45rem);letter-spacing:-.035em;line-height:1.07;margin:0}header p{margin:.5rem 0 0;color:var(--muted);font-size:.95rem}
.header-actions{display:flex;align-items:center;gap:1.05rem;white-space:nowrap}.header-actions>a:first-child{font-size:.9rem}.pdf{font-weight:bold;color:var(--white);background:var(--ink);padding:.65rem 1rem;border-radius:5px;text-decoration:none;white-space:nowrap}.mobile-summary{display:none}
.layout{max-width:1700px;margin:auto;display:grid;grid-template-columns:minmax(205px,252px) minmax(0,1fr);gap:clamp(1rem,2.9vw,3.1rem);padding:1.5rem clamp(1rem,4vw,4rem) 3rem}
nav{align-self:start;position:sticky;top:1rem;max-height:calc(100vh - 2rem);overflow:auto;scrollbar-width:thin;padding-right:.4rem}
nav h2{font-size:.83rem;margin:1.32rem 0 .42rem;color:var(--ink);line-height:1.35}nav h2:first-child{margin-top:.35rem}
nav ol{list-style:none;margin:0;padding:0}nav a{display:flex;align-items:baseline;gap:.6rem;padding:.31rem .46rem;line-height:1.26;color:var(--muted);text-decoration:none;border-left:3px solid transparent;font-size:.84rem}
nav small{font-variant-numeric:tabular-nums;font-size:.72rem;color:#687a82}nav a:hover,nav a[aria-current="page"]{background:#e1eeea;border-color:var(--teal);color:#075b55}nav>a{font-weight:bold;margin-top:1rem}
main{min-width:0}.mobile{display:none}figure{margin:0}figure img{display:block;width:100%;height:auto;border:1px solid var(--line);background:var(--white);border-radius:4px}
.foot,.pager{display:flex;align-items:baseline;justify-content:space-between;gap:.7rem;padding:.78rem .12rem}.foot{color:var(--muted);font-size:.82rem}.foot>a{white-space:nowrap}summary,select,button{cursor:pointer}
.pager{border-top:1px solid var(--line);align-items:center}.pager button{font:inherit;font-size:.89rem;border:1px solid var(--line);background:var(--white);color:var(--ink);border-radius:4px;padding:.4rem .75rem}.pager button:disabled{color:#859096;cursor:default}
.pager output{color:var(--muted);font-size:.85rem}select{max-width:100%;font:inherit;background:var(--white);color:var(--ink);padding:.48rem;border:1px solid #96aaa6;border-radius:3px}
.panel{margin:0 0 1rem}.js .panel[hidden],.js .record[hidden]{display:none}details p,.explorer p{line-height:1.55}.methods{border-top:1px solid var(--line);padding-top:1rem;color:var(--muted);font-size:.9rem;margin-top:.6rem}.methods summary{color:var(--ink);font-weight:bold}
.sources{display:flex;gap:.5rem 1.1rem;flex-wrap:wrap}.js-only{display:none}.js .pager{display:flex}.explorer{padding:clamp(1rem,2.5vw,2.3rem);border:1px solid var(--line);border-radius:4px;background:var(--white)}
.explorer h2{margin:0;font-size:clamp(1.45rem,2.5vw,2.2rem);letter-spacing:-.035em}.explorer h3{font-size:1.35rem}.explorer h4{margin:1.2rem 0 .2rem}.explorer p{margin:.45rem 0 1.2rem}.explorer select{margin-left:.6rem}
.stats,.requests{display:grid;gap:.75rem}.stats{grid-template-columns:repeat(3,1fr)}.stats div{padding:1rem;background:#e6f1ed;border-top:2px solid var(--teal)}.stats b,.stats span{display:block}.stats b{font-variant-numeric:tabular-nums;font-size:clamp(1.2rem,2.3vw,1.8rem);color:var(--teal)}.stats span,.secondary,.explainer{color:var(--muted);font-size:.88rem}
.requests{grid-template-columns:1fr 1fr}.request{border-top:1px solid var(--line)}.explorer .explainer{border-top:1px solid var(--line);padding-top:1rem;margin-bottom:0}
.narration{margin:0 0 .85rem;color:var(--teal);font-size:.92rem}.narration summary{font-weight:bold}.listen{display:flex;align-items:center;gap:.7rem 1rem;flex-wrap:wrap;padding:.65rem 0}.listen audio{width:min(100%,28rem)}
.sr{height:1px;width:1px;position:absolute;overflow:hidden;clip-path:inset(50%)}@media(max-width:850px){.layout{display:block}.js nav{display:none}.js .mobile{display:flex;align-items:center;gap:.6rem;margin-bottom:.9rem}.mobile label{white-space:nowrap}.mobile select{min-width:0}header{padding:1rem}.pdf{font-size:.85rem}.layout{padding:1rem}.stats{grid-template-columns:1fr 1fr 1fr}.stats div{padding:.65rem}.requests{grid-template-columns:1fr}nav{position:static;max-height:none}}
@media(max-width:650px){.mobile-summary{display:block;font-size:.94rem;line-height:1.45;padding:.7rem .15rem 0}.header-actions{flex-direction:column-reverse;gap:.5rem}}
@media(max-width:470px){.stats{grid-template-columns:1fr}.stats div{display:flex;align-items:center;gap:.8rem}.stats b{font-size:1.3rem}.foot{font-size:.76rem}header p{font-size:.8rem}}
</style></head><body><header><div><h1>CryoMN round supervisor</h1><p>@@COUNT@@ figures on the system, seven archived rounds and the pending Round 20 decision</p></div><div class="header-actions"><a href="#round-explorer">Round explorer</a><a class="pdf" href="@@PDF@@">Open PDF</a></div></header>
<div class="layout"><nav aria-label="Figure index">@@NAV@@<a href="#round-explorer" data-target="round-explorer">Explore by round →</a></nav><main>@@NARRATION@@
<div class="mobile"><label for="jump">Go to</label><select id="jump">@@OPTIONS@@<option value="round-explorer">Explore by round</option></select></div>
@@PANELS@@<div class="pager js-only"><button id="prev" type="button">← Previous</button><output id="position" aria-live="polite"></output><button id="next" type="button">Next →</button></div>
@@METHODS@@</main></div><script>
const ids=@@IDS@@;
const jump=document.getElementById('jump'),round=document.getElementById('round'),prev=document.getElementById('prev'),next=document.getElementById('next');
function render(){
 const key=ids.includes(location.hash.slice(1))?location.hash.slice(1):ids[0], i=ids.indexOf(key);
 for(const id of ids){document.getElementById(id).hidden=id!==key}
 document.querySelectorAll('[data-target]').forEach(a=>{if(a.dataset.target===key)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current')});
 jump.value=key;prev.disabled=i===0;next.disabled=i===ids.length-1;
 document.getElementById('position').textContent=i===ids.length-1?'Round explorer':'Figure '+(i+1)+' of '+(ids.length-1);
}
function renderRound(){document.querySelectorAll('.record').forEach(el=>el.hidden=el.id!=='after-'+round.value)}
function change(delta){const i=ids.indexOf(jump.value)+delta;if(i>=0&&i<ids.length)location.hash=ids[i]}
jump.addEventListener('change',()=>{location.hash=jump.value});round.addEventListener('change',renderRound);
prev.addEventListener('click',()=>change(-1));next.addEventListener('click',()=>change(1));window.addEventListener('hashchange',render);
document.addEventListener('keydown',event=>{if(['SELECT','INPUT','TEXTAREA','BUTTON','SUMMARY'].includes(event.target.tagName)||event.metaKey||event.ctrlKey||event.altKey)return;
 if(event.key==='ArrowRight'||event.key==='ArrowLeft'){event.preventDefault();change(event.key==='ArrowRight'?1:-1)}});
document.documentElement.classList.add('js');renderRound();render();
</script></body></html>'''
