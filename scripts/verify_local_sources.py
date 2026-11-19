#!/usr/bin/env python3
"""Read-only verification of frozen V12 checkpoint paths and their SHA256."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/'archive_metadata/model_registry_sha256.json'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),''):h.update(b)
 return h.hexdigest()

def main():
 reg=json.loads(REG.read_text())
 bad=[];checked=0
 for model,items in reg.items():
  for item in items:
   p=ROOT/item['path']; expected=item['sha256']
   if not p.exists():
    print(f'[MISSING] {model} {item["path"]}');bad.append(str(p));continue
   actual=sha(p)
   if actual!=expected:
    print(f'[HASH MISMATCH] {model} {p} expected={expected} actual={actual}')
    bad.append(str(p))
   else:print(f'[OK] {model} {item["path"]}');checked+=1
 print(f'Validated={checked}, missing_or_mismatch={len(bad)}')
 if bad:
  print('Normal for code-only ZIP; run this in original Mac project with model outputs.')
  return 1
 return 0
if __name__=='__main__':sys.exit(main())
