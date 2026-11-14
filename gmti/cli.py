"""Small independent train/test entrypoints; launcher runs this same version."""
import argparse
from datetime import datetime,timedelta,timezone
import os
from pathlib import Path
import subprocess
import sys

PROJECT=Path(__file__).resolve().parent.parent


def main(stage):
    parser=argparse.ArgumentParser(description=f'Current 640 MHz / 60x60m GMTI {stage}; ideal channels only')
    parser.add_argument('--gpus',help='GPU IDs, e.g. 0,1,2; omit for one device')
    if stage=='train':
        parser.add_argument('--config',type=Path,default=PROJECT/'configs/current.yaml')
        parser.add_argument('--run-dir',type=Path)
        parser.add_argument('--resume',action='store_true')
        parser.add_argument('--model',choices=['temporal','single'],default='temporal')
        for key in ('epochs','train-samples','validation-samples'):parser.add_argument('--'+key,type=int)
    else:
        parser.add_argument('--checkpoint',type=Path,default=PROJECT/'checkpoints/best.pt')
        parser.add_argument('--dataset',choices=['simulated','semireal','both'],default='simulated')
        parser.add_argument('--output',type=Path)
        parser.add_argument('--scene-config',type=Path,default=PROJECT/'configs/current.yaml',help='Source image/crop configuration for semireal tests')
        parser.add_argument('--scene-crop',type=int,nargs=4,metavar=('R0','R1','A0','A1'),help='Override semireal source crop in native pixels')
        parser.add_argument('--trajectories',type=int,default=4)
        parser.add_argument('--scr-db',type=float,nargs='+',default=[-5.,0.,10.])
        parser.add_argument('--snr-db',type=float,default=15.)
        parser.add_argument('--seed',type=int,default=20261112)
        parser.add_argument('--traditional-iterations',type=int,default=20)
        parser.add_argument('--focus-frames',type=int,default=3)
        parser.add_argument('--focus-examples',type=int,default=None,help='Maximum trajectories to refocus; default: all trajectories')
    args=parser.parse_args()
    counts=('epochs','train_samples','validation_samples') if stage=='train' else ('trajectories','traditional_iterations','focus_frames','focus_examples')
    if any(getattr(args,key) is not None and getattr(args,key)<1 for key in counts):parser.error('Counts must be positive')
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    argv=list(sys.argv[1:])
    if stage=='train':
        if args.resume and args.run_dir is None:parser.error('--resume requires --run-dir')
        if args.run_dir is None:
            args.run_dir=PROJECT/'runs'/('train_'+stamp)
            argv.extend(['--run-dir',str(args.run_dir)])
        if args.resume:
            if not (args.run_dir/'checkpoints/last.pt').is_file():parser.error('Missing last.pt for resume')
        elif args.run_dir.exists() and any(args.run_dir.iterdir()):parser.error('Training directory is not empty; use --resume or a new --run-dir')
    else:
        if not args.checkpoint.is_file():parser.error(f'Checkpoint not found: {args.checkpoint}. Finish training before testing.')
        if args.dataset in ('semireal','both') and not args.scene_config.is_file():parser.error(f'Scene configuration not found: {args.scene_config}')
        if args.output is None:
            args.output=PROJECT/'runs'/('test_'+stamp)
            argv.extend(['--output',str(args.output)])
        if args.output.exists() and any(args.output.iterdir()):parser.error('Use a new --output directory')
    os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:128')
    os.environ.setdefault('PYTHONFAULTHANDLER','1');os.environ.setdefault('TORCH_SHOW_CPP_STACKTRACES','1')
    if args.gpus:
        if 'LOCAL_RANK' in os.environ:parser.error('Do not combine --gpus with torchrun')
        ids=args.gpus.split(',')
        if not all(i.isdigit() for i in ids) or len(ids)!=len(set(ids)):parser.error('GPU IDs must be distinct nonnegative integers')
        for i,arg in enumerate(argv):
            if arg=='--gpus':del argv[i:i+2];break
            if arg.startswith('--gpus='):del argv[i];break
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=args.gpus
        code=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(ids)}',str(PROJECT/f'{stage}.py'),*argv],env=env).returncode
        raise SystemExit(code)
    import math
    import torch
    import torch.distributed as dist
    from .engine import train,evaluate
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    device=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(1)
    if device.type=='cuda':torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if stage=='test' and (not all(math.isfinite(v) for v in args.scr_db) or len(set(args.scr_db))!=len(args.scr_db) or not math.isfinite(args.snr_db)):
        parser.error('Require finite distinct SCR values and finite SNR')
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',**({'device_id':device} if device.type=='cuda' else {}))
    try:
        if world>1:
            target=[str(args.run_dir if stage=='train' else args.output) if rank==0 else None]
            dist.broadcast_object_list(target,src=0)
            if stage=='train':args.run_dir=Path(target[0])
            else:args.output=Path(target[0])
        if stage=='train':train(args,device,rank,world)
        else:
            for name in (('simulated','semireal') if args.dataset=='both' else (args.dataset,)):
                evaluate(args,device,rank,world,name)
    finally:
        if world>1:dist.destroy_process_group()
