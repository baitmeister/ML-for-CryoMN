"""Read-only distinction between configured code and a frozen live proposal."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helper.group10_config import load_group10_config_for_round, reference_readiness
from helper.terminal_force import DEFINITION


def inspect_readiness(campaign_root, code_root=None):
    code_root = Path(code_root or Path(__file__).resolve().parents[3])
    campaign_root = Path(campaign_root).resolve()
    config = load_group10_config_for_round(10, code_root/'config_v2/group10.yaml')
    observations_path = campaign_root/'data/processed_v2/observations.csv'
    observations = pd.read_csv(observations_path)
    rounds = pd.to_numeric(observations.batch_id.astype(str).str.extract(r'^ROUND_(\d+)$')[0],errors='coerce')
    last = int(rounds.max()) if rounds.notna().any() else 0
    results = campaign_root/'results/multi_objective_v2'
    worksheet = results/'next_round/next_round_candidates.csv'
    sheet = pd.read_csv(worksheet) if worksheet.exists() else pd.DataFrame()
    batches = sheet.get('batch_id',pd.Series(dtype=str)).dropna().unique().tolist()
    live_config_exists = (campaign_root/'config_v2/group10.yaml').exists()
    live_config = load_group10_config_for_round(10, campaign_root/'config_v2/group10.yaml') if live_config_exists else {}
    relevant_code = ['helper/terminal_force.py','helper/feedback.py','helper/group10_config.py',
                     'helper/candidate_workflow.py','helper/group10_selection.py','helper/selection_reporting.py']
    code_matches = all((campaign_root/'src/08_multi_objective'/name).exists() and
                       hashlib.sha256((campaign_root/'src/08_multi_objective'/name).read_bytes()).digest() ==
                       hashlib.sha256((code_root/'src/08_multi_objective'/name).read_bytes()).digest()
                       for name in relevant_code)
    proposal = results/'rounds/ROUND_010/proposal/selection_metadata.json'
    frozen = json.loads(proposal.read_text()) if proposal.exists() else {}
    frozen_endpoint = frozen.get('group10',{}).get('effective_config',{}).get('mechanical_endpoint',{})
    frozen_matches = frozen.get('group10',{}).get('effective_config') == config
    blockers = []
    if last < 9:
        blockers.append('Group 9 results are not yet ingested; no final Group 10 slate can be approved from this evidence snapshot')
    if last >= 10:
        blockers.append('Group 10 or later observations already exist; this is not an unstarted Group 10 transition')
    if not frozen_matches:
        blockers.append('No Group 10 proposal frozen with the current terminal endpoint in this campaign')
    if batches != ['ROUND_010']:
        blockers.append('The active worksheet is not Group 10')
    if not live_config_exists:
        blockers.append('This campaign checkout does not contain the Group 10 implementation; run the reviewed branch against the intended campaign only after transition')
    elif live_config != config or not code_matches:
        blockers.append('Campaign code/settings differ from the reviewed implementation')
    branch = subprocess.run(['git','-C',str(code_root),'branch','--show-current'],capture_output=True,text=True)
    return dict(code_root=str(code_root), code_branch=branch.stdout.strip() or 'isolated_validation_copy',
                campaign_root=str(campaign_root), policy_version=config['policy_version'],
                earliest_activation_group=config['activation_round'], configured_endpoint=DEFINITION,
                endpoint_code_configured=True, reference=reference_readiness(config),
                gp_methodology_revision=config['production_model_revision'],
                noise_revision=config['production_noise_revision'],
                latest_observed_group=last, active_worksheet_groups=batches,
                campaign_has_group10_code=live_config_exists,
                campaign_code_matches_reviewed=code_matches,
                group10_frozen_with_current_endpoint=frozen_endpoint==config['mechanical_endpoint'],
                ready_for_group10_wetlab=not blockers, blockers=blockers,
                settings_from_round_csv=['replicate_id / repeated result rows','instron_file','needles_compressed'],
                no_files_modified=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--require-ready',action='store_true')
    args=parser.parse_args()
    state=inspect_readiness(args.campaign_root)
    print(json.dumps(state,indent=2))
    if args.require_ready and not state['ready_for_group10_wetlab']:
        sys.exit(2)
