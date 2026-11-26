"""Schematics for the workflow, decision policy, and pending experiment."""

from matplotlib import patches

from .layout import BLUE, GREY, INK, MINT, ORANGE, PALE, PEACH, PURPLE, SKY, TEAL, Plate, arrow, canvas, compact_recipe, node


def loop(fig, data):
    a = canvas(fig)
    node(a, 0, 57, 27, 34, "Researcher", "Runs the bench assays.\nReturns measurements and\nsample identities.", color=ORANGE, face=PEACH)
    node(a, 36, 57, 27, 34, "Validate the round", "Match the frozen worksheet,\nformation calls, readouts\nand compression traces.")
    node(a, 72, 57, 27, 34, "Evaluate progress", "Score frozen forecasts.\nUpdate paired outcomes and\nthe measured frontier.")
    node(a, 72, 5, 27, 34, "Scientist and critic", "Propose a question, an\nalternative explanation and\na result that changes direction.", color=BLUE, face=SKY)
    node(a, 36, 5, 27, 34, "Allocate the budget", "Reserve follow-up slots,\ncheck constraints and fill\nother slots with the optimizer.")
    node(a, 0, 5, 27, 34, "Next worksheet", "Freeze candidates, forecasts\nand backup order. Deliver the\nbatch to the researcher.", color=ORANGE, face=PEACH)
    for path in [[(27,74),(36,74)],[(63,74),(72,74)],[(85.5,57),(85.5,39)],[(72,22),(63,22)],[(36,22),(27,22)],[(13.5,39),(13.5,57)]]:
        arrow(a, path)


def cutoff(fig, data):
    a = canvas(fig)
    nodes = [(0, "After R18", "64.63% viability\nIntact patch\nForce unmeasured", MINT, TEAL),
             (25, "Freeze R19 plan", "Nominate the intact\nscreened hit for its\nfirst paired test.", SKY, BLUE),
             (50, "Reveal R19", "82.66% viability\n0.500 N/needle\nTwo force readings", PEACH, ORANGE),
             (75, "Draft R20", "Request a paired test\nfor a second hit and a\nrepeat of the leader.", MINT, TEAL)]
    for x, title, body, face, color in nodes:
        node(a, x, 39, 21.7, 36, title, body, color=color, face=face)
    for x in (21.7, 46.7, 71.7):
        arrow(a, [(x,56),(x+3.3,56)])
    for x, txt in [(0,"Evidence available for R19 planning"),(50,"Evidence available for R20 planning")]:
        a.plot([x,x+46.7],[28,28], c=GREY, lw=1.3)
        a.text(x+23.35, 22, txt, fontsize=12, ha="center", va="top", color=GREY)


def slot_rule(fig, data):
    a = canvas(fig)
    node(a, 0, 37, 23, 33, "First priority", "Pair the latest round’s\nmost viable intact hit\nthat still lacks force.")
    node(a, 31, 37, 26, 33, "Prior primary gates", "Did all predesignated\nmechanical candidates\npass formation?")
    node(a, 67, 65, 32, 30, "All passed", "A frontier point seen in only one\npaired round can receive a repeat slot.", color=BLUE, face=SKY)
    node(a, 67, 35, 32, 23, "A primary failed", "Retain optimizer slots and backups.", color=ORANGE, face=PEACH)
    node(a, 67, 4, 32, 23, "A gate is missing", "Flag the missing formation evidence.", color=GREY, face=PALE)
    arrow(a, [(23,53),(31,53)])
    for y in (80,46.5,15.5):
        arrow(a, [(57,53),(61,53),(61,y),(67,y)])
    a.text(0, 21, "Every draft must fit the screen and compression budgets,", fontsize=11.5, color=GREY)
    a.text(0, 14, "preserve protected roles and pass the composition caps.", fontsize=11.5, color=GREY)


def critic(fig, data):
    a = canvas(fig)
    node(a, 0, 56, 28, 34, "Scientist", "Names a question, a rival\nexplanation and what each\noutcome would change.", color=BLUE, face=SKY)
    node(a, 36, 56, 27, 34, "Independent critic", "Receives the same evidence.\nChecks whether the proposed\nmeasurement can decide.")
    node(a, 72, 56, 27, 34, "Accepted", "Use the reviewed choice\nwithin the eligible slot\nallocations.", color=BLUE, face=SKY)
    node(a, 36, 2, 27, 31, "One revision", "The scientist addresses the\nobjection. The critic reads\nthe revised proposal.", color=GREY, face=PALE)
    node(a, 72, 2, 27, 31, "Default allocation", "A failed review or a further\nrevision request retains the\nversioned default.", color=GREY, face=PALE)
    arrow(a,[(28,73),(36,73)])
    arrow(a,[(63,73),(72,73)])
    arrow(a,[(46,56),(46,33)])
    arrow(a,[(54,33),(54,56)])
    arrow(a,[(63,17),(72,17)],dashed=True)
    a.text(.6, 28, "Round 20", fontsize=14, fontweight="bold", color=BLUE)
    a.text(.6, 20, "The recorded review accepted\nthe screened hit and the repeat.", fontsize=11.5, va="top", linespacing=1.4)


def budget(fig, d):
    a = canvas(fig)
    plan = d['replay']['pending_plan']
    slate = d['slates'][20].set_index('formulation_id')
    original = slate.loc[slate.primary].sort_values('mechanical_selection_rank')
    draft = plan['new_project_draft']['compression_priorities']
    labels = {r.formulation_id: r.recipe for r in original.reset_index().itertuples()}
    labels.update({r['formulation_id']:r['recipe'] for r in draft})
    def row(x,y,w,key,title,color,face):
        a.add_patch(patches.FancyBboxPatch((x,y),w,14.9,boxstyle="round,pad=.1,rounding_size=1",facecolor=face,edgecolor=color,lw=1.05,zorder=2))
        a.text(x+1.1,y+10.8,title,ha='left',va='center',fontsize=11.7,color=color,weight='bold')
        recipe = compact_recipe(labels[key]).replace('ethylene glycol','EG').replace('propylene glycol','PG')
        a.text(x+1.1,y+4.8,recipe,ha='left',va='center',fontsize=9.8,color=INK)
    a.text(0,95,"Frozen lab priorities",weight='bold',fontsize=15,va='top')
    a.text(56,95,"Agent-reviewed new-project draft",weight='bold',fontsize=15,va='top')
    ys=[69,50,31,12]
    oldkeys=original.index.tolist()
    newkeys=[r['formulation_id'] for r in draft]
    for i,k in enumerate(oldkeys):
        row(0,ys[i],44,k,f"Priority {i+1}   ·   screening rank {int(original.at[k,'selection_rank'])}",GREY,PALE)
        if k in newkeys:
            arrow(a,[(44,ys[i]+7.5),(56,ys[newkeys.index(k)]+7.5)],color=TEAL)
    for i,slot in enumerate(draft):
        purpose={'pair_screened_hit':'paired follow-up','repeat_frontier':'independent repeat','optimizer primary':'optimizer choice'}[slot['purpose']]
        is_new=slot['formulation_id'] not in oldkeys
        row(56,ys[i],43.5,slot['formulation_id'],f"Priority {i+1}   ·   {purpose}",ORANGE if is_new else TEAL, PEACH if is_new else MINT)
    a.text(.5,3,"EG = ethylene glycol     PG = propylene glycol     FBS = fetal bovine serum     Concentrations rounded to three significant figures",fontsize=9.4,color=GREY)


def outcomes(fig, data):
    a = canvas(fig)
    node(a, 0, 40, 29, 34, "Compare the two recipes", "Document the cell batch and\npreparations. Evaluate patch\nformation and both endpoints.")
    options=[(77,"Repeat matches or exceeds both endpoints", "Keep it as the next comparison formulation.",TEAL,MINT),
             (54,"Screened hit dominates the repeat*", "Prioritize it for confirmation.",BLUE,SKY),
             (31,"Each recipe leads on one endpoint", "Keep both measured trade-offs.",PURPLE,"#eee8f2"),
             (8,"Formation fails or force is missing", "Record the failed gate or leave the joint comparison open.",GREY,PALE)]
    for y,title,body,c,face in options:
        node(a, 42, y, 57, 19.7, title, body, color=c, face=face, title_size=12.6,body_size=10.8)
        arrow(a,[(29,57),(35,57),(35,y+9.9),(42,y+9.9)])
    a.text(0,25,"*At least as high on both endpoints\n  and strictly higher on one.",fontsize=10.6,color=GREY,va='top',linespacing=1.4)


def prospective(fig, d):
    a = canvas(fig)
    node(a,0,36,24.5,34,"Before validation", "Fix the discretionary slot,\ncontrols, eligibility, budget\nand success definition.")
    node(a,35,62,26.5,30,"Optimizer choice", "The base selector nominates\na candidate for the slot.",color=BLUE,face=SKY)
    node(a,35,15,26.5,30,"Supervisor choice", "The reviewed question guides\nselection for the same slot.",color=ORANGE,face=PEACH)
    node(a,72,36,27,34,"Common evaluation", "Use the same assays and\nindependent confirmation\ncriteria for both strategies.")
    for y in (77,30):
        arrow(a,[(24.5,53),(29.5,53),(29.5,y),(35,y)])
        arrow(a,[(61.5,y),(66.5,y),(66.5,53),(72,53)])
    a.text(48.5,53,"Randomize across comparable batches", ha='center',va='center',fontsize=10.3,color=GREY)
    a.text(50,1,"Log nomination overlap, assay costs, failed gates and missing results for each assigned slot.",ha='center',fontsize=11.2,color=GREY)


def first_plates():
    return [
        Plate('round-loop',0,'One validation round feeds the next research question','The goal is high post-thaw cell viability and high force per loaded needle. The researcher supplies each round’s bench results.',
              'The prototype runs validation replay, scoring and planning. Intake events and worksheet delivery are deployment integrations.', 'Source • CryoMN supervisor design and round-close protocol',loop,'Schematic'),
        Plate('evidence-cutoff',0,'What could the supervisor know at each decision?','The same formulation crosses three rounds. Its first force measurement arrives only after the R19 plan is frozen.',
              'The policy was designed retrospectively. Each replayed decision uses the evidence available by its stated cutoff.', 'Source • frozen R18–R20 proposals and R18–R19 result receipts',cutoff,'Schematic'),
        Plate('slot-policy',0,'A formation result can change the next four slots','The default planning rule chooses which uncertainty to investigate before filling the remaining places.',
              'R18 had one failed mechanical primary. R19 had four passes, which made a confirmation slot eligible in the R20 draft.', 'Source • supervisor next_agenda and draft_allocation, R18–R19 gate results',slot_rule,'Schematic'),
        Plate('scientist-critic',0,'The scientist proposes a test the critic can challenge','A dated evidence packet fixes the observations, eligible decisions and experimental budget for both agents.',
              'The optional agent step can keep the default or release an eligible confirmation slot to the optimizer.', 'Source • deliberation interface and recorded R20 scientist–critic exchange',critic,'Schematic')]


def last_plates():
    return [
        Plate('round20-budget',3,'Where does the R20 repeat slot come from?','Both designs have 12 screens and four first-choice compression slots. Arrows connect formulations retained in both.',
              'The original fourth primary moves to backup. Its frozen forecasts were 64.20% viability and 0.502 N per loaded needle.', 'Source • frozen R20 proposal and accepted as-of-R19 planning packet · R20 measurements pending',budget,'Schematic'),
        Plate('round20-branches',3,'How each possible R20 result changes the decision','The reviewed comparison is between the ectoin / ethylene glycol / FBS repeat and the propylene glycol / raffinose hit.',
              'A measured ordering would apply to the tested batch. The project has yet to define an application or reproducibility margin.', 'Source • recorded R20 scientist–critic exchange · conditional decisions only',outcomes,'Schematic'),
        Plate('prospective-test',3,'How to measure the benefit of agent steering','A future comparison can isolate the value of assigning a discretionary test to a supervised scientific question.',
              'A candidate primary measure is independently confirmed joint improvements per mechanical test. Define the improvement before assignment.', 'Design proposal • budget-matched prospective comparison · outcomes have yet to be collected',prospective,'Schematic')]
