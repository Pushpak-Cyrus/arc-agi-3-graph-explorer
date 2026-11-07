import csv, json
from pathlib import Path
from arc_agi.scorecard import EnvironmentScoreCalculator
root=Path.cwd(); run= root/'research/system2/reachability_system1_2026-10-30/20261030T092714Z'
p=run/'runs.csv'; rows=list(csv.DictReader(p.open(encoding='utf-8-sig')))
for row in rows:
    mem=root/row['memory_dir']; exp=mem/'experiments.jsonl'
    if not exp.exists(): continue
    events=[]
    for line in exp.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        x=json.loads(line)
        if x.get('levels_after',0)>x.get('levels_before',0):
            for level in range(int(x['levels_before'])+1,int(x['levels_after'])+1): events.append((level,int(x['step'])+1))
    baselines=json.loads(row['environment_baseline_actions']); completed=int(row['levels_completed']); totals=dict(events); prior=0
    calc=EnvironmentScoreCalculator(id=row['game_id'])
    for i,b in enumerate(baselines,1):
        if i<=completed:
            cum=totals.get(i,prior); acts=max(0,cum-prior); done=True; prior=cum
        elif i==completed+1:
            acts=max(0,int(row['actions_including_resets'])-prior); done=False
        else: acts=0; done=False
        calc.add_level(level_index=i,completed=done,actions_taken=acts,baseline_actions=int(b),game_id=row['game_id'])
    row['scorecard_score']=str(calc.to_score().score)
with p.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# rebuild summary rows
(run/'runs.jsonl').write_text(''.join(json.dumps({**r,'levels_completed':int(r['levels_completed']),'budget':int(r['budget']),'scorecard_score':float(r['scorecard_score'])},sort_keys=True)+'\n' for r in rows),encoding='utf-8')
for r in rows:
    if int(r['levels_completed']): print(r['game_id'],r['budget'],r['levels_completed'],r['first_level_action'],r['scorecard_score'])
