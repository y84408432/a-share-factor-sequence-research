#!/usr/bin/env python3
"""Make a PRIVATE local checkpoint+metadata backup, verifying checkpoint hashes."""
import argparse,hashlib,json,sys,platform,zipfile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as fd:
  for buf in iter(lambda:fd.read(1024*1024),''): h.update(buf)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--output',required=True,help='Target ZIP for private local archive')
 args=ap.parse_args();out=Path(args.output).expanduser().resolve()
 reg=json.loads((ROOT/'archive_metadata/model_registry_sha256.json').read_text())
 checkpoints=[]
 for model,items in reg.items():
  for item in items:
   p=ROOT/item['path']
   if not p.is_file():raise SystemExit(f'Missing required checkpoint for {model}: {p}')
   if sha(p)!=item['sha256']:raise SystemExit(f'Checkpoint hash mismatch: {p}')
   checkpoints.append(p)
 desired=[]
 desired.extend(p for p in ROOT.glob('*.py'))
 desired.extend(p for p in ROOT.glob('docs/**/*.md') if p.is_file())
 desired.extend(p for p in ROOT.glob('docs/**/*.pdf') if p.is_file())
 desired.extend(p for p in ROOT.glob('docs/assets/*') if p.is_file())
 desired.extend(p for p in ROOT.glob('scripts/*.py'))
 desired.extend(p for p in ROOT.glob('lstm_research/*.py'))
 desired.extend(p for p in ROOT.glob('tests/*.py'))
 desired.extend(p for p in ROOT.glob('archive_metadata/*.json'))
 desired.extend(p for p in ROOT.glob('results/verified_summaries/*') if p.is_file())
 desired.extend(p for p in ROOT.glob('output/v*/**/*') if p.is_file() and (p.name.endswith(('.json','.csv','.md','.log')) or p.name=='v12_review_bundle.zip'))
 desired.extend(checkpoints)
 for extra in [ROOT/'README.md',ROOT/'requirements.txt',ROOT/'output/v12_final/v12_preflight.json',ROOT/'output/v12_final/v12_final_manifest.json']:
  if extra.is_file():desired.append(extra)
 desired=sorted(set(desired))
 if out in desired:raise SystemExit('Output path collides with input source')
 out.parent.mkdir(parents=True,exist_ok=True)
 env={'python':sys.version,'platform':platform.platform(),'pip_freeze':None}
 try:
  p=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,timeout=30)
  if p.returncode==0:env['pip_freeze']=p.stdout
 except Exception:pass
 hashes={p.relative_to(ROOT).as_posix():sha(p) for p in desired}
 with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
  for p in desired:z.write(p,arcname=p.relative_to(ROOT).as_posix())
  z.writestr('archive_metadata/LOCAL_EXPORT_MANIFEST.json',json.dumps({'hashes':hashes,'environment':env,'data_excluded':'Qlib historical bin source and large NumPy/pickle cache'},ensure_ascii=False,indent=2))
 print(f'Private local evidence saved: {out} ({len(desired)} files, {len(checkpoints)} checkpoints).')
 print('Not a public-release ZIP; source Qlib market snapshot is NOT included.')
if __name__=='__main__':main()
