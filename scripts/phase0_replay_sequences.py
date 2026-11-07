from __future__ import annotations
import copy, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'Cyrus')]
import arc_agi
from arc_agi import OperationMode
from tools.oracle_dev_search import last_frame, apply_signature
OUT=ROOT/'research/system2/repair_phase0_2026-10-31/winning_trace_replay'
SEQS={
'cd82':[(1,None,None),(6,31,63),(3,None,None),(3,None,None),(2,None,None),(2,None,None),(4,None,None),(5,None,None)],
'sp80':[(4,None,None),(4,None,None),(5,None,None),(1,None,None),(1,None,None),(4,None,None),(4,None,None),(2,None,None),(2,None,None),(3,None,None),(2,None,None),(5,None,None)],
'tu93':[(a,None,None) for a in [4,2,2,4,1,4,2,2,3,3,3,1,1,2,1,3,1,3,3,2,4,4,4,1,2,2,3,4,1,4,3,1,3,4,2]]}
# The 34-step L2 path is read from the recorded trace; L1 replay sequence is only validated below.

def describe(a,b):
 if a.shape!=b.shape:return {'changed_pixels':None,'bbox':None,'colors_before_after':[]}
 m=a!=b; ys,xs=np.where(m); colors=sorted(set(map(int,a[m]))|set(map(int,b[m]))) if m.any() else []
 return {'changed_pixels':int(m.sum()),'bbox':[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if m.any() else None,'colors_before_after':colors}
arc=arc_agi.Arcade(operation_mode=OperationMode.OFFLINE); results=[]
for gid,seq in SEQS.items():
 env=arc.make(gid,seed=0,scorecard_id=f'phase0-replay-{gid}'); raw=env.reset(); game=copy.deepcopy(env._game); rows=[]
 for i,sig in enumerate(seq,1):
  before=last_frame(raw); afterraw=apply_signature(game,sig); after=last_frame(afterraw); diff=describe(before,after)
  h=hashlib.sha256(after.tobytes()).hexdigest()
  rows.append({'step':i,'action_signature':sig,'state_hash':h,'no_op':diff['changed_pixels']==0,'frame_diff':diff,'levels_completed':int(afterraw.levels_completed),'state':afterraw.state.name})
  raw=afterraw
  if int(raw.levels_completed)>0:break
 (OUT/f'{gid}_replay.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
 results.append({'game':gid,'steps_executed':len(rows),'levels_completed':int(raw.levels_completed),'final_state':raw.state.name,'trace':f'{gid}_replay.jsonl'})
(OUT/'summary.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'seed':0,'known_sequences':{g:[list(s) for s in q] for g,q in SEQS.items()},'results':results},indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))

