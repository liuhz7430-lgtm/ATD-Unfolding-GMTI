"""Atomic results and self-contained checkpoints with explicit source provenance."""
from pathlib import Path
import hashlib
import json
import math
import os
import platform
import subprocess
import numpy as np
import torch
import yaml
from .config import PROJECT


def safe_value(value):
    if isinstance(value,dict):return {str(k):safe_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe_value(v) for v in value]
    if isinstance(value,np.ndarray):return safe_value(value.tolist())
    if isinstance(value,np.generic):return safe_value(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(safe_value(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def manifest():
    files=[*sorted((PROJECT/'gmti').glob('*.py')),PROJECT/'train.py',PROJECT/'test.py']
    return {str(p.relative_to(PROJECT)):file_hash(p) for p in files}


def signature(entries):return hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()


def save_checkpoint(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary);temporary.replace(path)


def load_checkpoint(path,resume=False):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'Checkpoint not found: {path}. Finish a training epoch before testing.')
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('format_version')!=2:raise ValueError('This entrypoint requires a consolidated version-2 checkpoint')
    saved=state['source_manifest']
    if signature(saved)!=state['source_signature']:raise ValueError('Checkpoint provenance is inconsistent')
    critical={'gmti/model.py','gmti/frontend.py','gmti/physics.py','gmti/operators.py','gmti/temporal.py'}
    current=manifest()
    changed=[name for name,h in saved.items() if (resume or name in critical) and current.get(name)!=h]
    if changed:raise ValueError(f'Checkpoint source differs: {changed}')
    return state,state['config']


def environment(device,world_size):
    try:gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):gpu=None
    return dict(python=platform.python_version(),torch=torch.__version__,cuda_runtime=torch.version.cuda,
        device=str(device),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),world_size=world_size,
        gpus=gpu,source_signature=signature(manifest()))


def save_configuration(folder,cfg):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    (folder/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    write_json(folder/'source_manifest.json',manifest())
