"""Direct training/evaluation orchestration for the current implementation."""
import copy
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .artifacts import environment,file_hash,load_checkpoint,manifest,save_checkpoint,save_configuration,signature,write_json
from .config import PROJECT,load,diagnostics
from .data import EchoDataset,SimulatedTestDataset,to_device
from .focus import example_ids,save_focus
from .model import JointUnfoldNet
from .objective import loss_function
from .physics import RadarPhysics
from .report import make_report
from .semireal import SemiRealDataset,scatter_field,project_background,configure_scene
from .traditional import ALGORITHM_VERSION,TraditionalEstimator


def reduce_gradients(model,count):
    params=list(model.parameters())
    packed=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).double() for p in params])
    dist.all_reduce(packed);packed/=count
    start=0
    for p in params:
        p.grad=packed[start:start+p.numel()].reshape_as(p).to(p.dtype).clone();start+=p.numel()


def train(args,device,rank,world):
    folder=args.run_dir.resolve();state=None
    if args.resume:
        state,cfg=load_checkpoint(folder/'checkpoints/last.pt',resume=True)
        cfg=copy.deepcopy(cfg)
        if state['world_size']!=world:raise ValueError('Resume using the original GPU count/global batch')
        if args.epochs is not None:
            if args.epochs<=state['epoch']+1:raise ValueError('--epochs must exceed the completed epoch count')
            cfg['training']['epochs']=args.epochs
    else:
        cfg=load(args.config);cfg['temporal']['enabled']=args.model=='temporal'
        for key in ('epochs','train_samples','validation_samples'):
            if getattr(args,key) is not None:cfg['training'][key]=getattr(args,key)
    if rank==0:
        if not args.resume and folder.exists() and any(folder.iterdir()):raise ValueError('Training directory is not empty')
        save_configuration(folder,cfg)
        write_json(folder/'sampling.json',diagnostics(cfg))
    if world>1:dist.barrier()
    sources=manifest();started_training=time.time()
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    model=JointUnfoldNet(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
    begin=0;best=float('inf')
    if state is not None:
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        begin=state['epoch']+1;best=state['best_validation_loss']
    data=EchoDataset(cfg,'train');validation=EchoDataset(cfg,'validation')
    batch_size=cfg['training']['batch_size'];global_batch=batch_size*world
    if rank==0:
        write_json(folder/'environment.json',environment(device,world))
        print(f'Training {len(data)} ideal trajectories; per-device batch={batch_size}; global batch={global_batch}; sampling={cfg["radar"]["frequency_samples"]}x{cfg["radar"]["pulses"]}; output={folder}',flush=True)
    for epoch in range(begin,cfg['training']['epochs']):
        started=time.time();model.train();total=0.
        order=torch.randperm(len(data),generator=torch.Generator().manual_seed(cfg['seed']+epoch)).tolist()
        for step,start in enumerate(range(0,len(order),global_batch)):
            selected=order[start:start+global_batch];local=selected[rank::world]
            optimizer.zero_grad(set_to_none=True)
            if local:
                batch=to_device(torch.utils.data.default_collate([data[i] for i in local]),device)
                pred=model(batch['echo']);loss,_=loss_function(pred,batch['truth'],batch['echo'],cfg)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                (loss*len(local) if world>1 else loss).backward();total+=float(loss.detach())*len(local)
                del batch,pred,loss
            if world>1:reduce_gradients(model,len(selected))
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();model.frontend.clear()
            if rank==0:print(f'epoch {epoch+1}, update {step+1}/{(len(data)+global_batch-1)//global_batch}',flush=True)
        model.eval();value=0.;count=0
        with torch.no_grad():
            for i in range(rank,len(validation),world):
                batch=to_device(torch.utils.data.default_collate([validation[i]]),device)
                pred=model(batch['echo']);loss,_=loss_function(pred,batch['truth'],batch['echo'],cfg)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite validation loss')
                value+=float(loss);count+=1;del batch,pred,loss
        totals=torch.tensor([total,value,count],device=device,dtype=torch.float64)
        if world>1:dist.all_reduce(totals)
        val=float(totals[1]/totals[2]);better=val<best;best=min(best,val)
        if rank==0:
            record=dict(epoch=epoch+1,train_loss=float(totals[0])/len(data),validation_loss=val,seconds=time.time()-started)
            with (folder/'history.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            saved=dict(format_version=2,model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best_validation_loss=best,
                config=cfg,source_manifest=sources,source_signature=signature(sources),world_size=world)
            if state is not None and 'lineage' in state:saved['lineage']=state['lineage']
            save_checkpoint(folder/'checkpoints/last.pt',saved)
            if better:save_checkpoint(folder/'checkpoints/best.pt',saved)
            print(json.dumps(record),flush=True)
    memory=dict(rank=rank,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None)
    workers=[memory]
    if world>1:
        workers=[None]*world;dist.all_gather_object(workers,memory)
    if rank==0:write_json(folder/'runtime.json',dict(elapsed_s=time.time()-started_training,world_size=world,workers=workers))
    del model,optimizer;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()


@torch.no_grad()
def predict(model,dataset,device,rank,world,focus_output=None,focus_frames=3,focus_examples=None):
    model.eval();rows=[]
    selected=example_ids(dataset.entries,focus_examples) if focus_output is not None else []
    for i in range(rank,len(dataset),world):
        sample=dataset[i];pred=model(sample['echo'][None].to(device))
        row=dict(id=i,info=sample['info'])
        for key in ('position','velocity','radial','single_position','single_velocity'):
            if key in pred:row[key]=pred[key][0].cpu().numpy().tolist()
        if 'objective_history' in pred:
            row['traditional_diagnostics']={key:pred[key][0].cpu().numpy().tolist()
                for key in ('ati_radial','objective_history','rejected_updates')}
        for key in ('position','velocity','radial'):row['true_'+key]=sample['truth'][key].numpy().tolist()
        rows.append(row)
        if i in selected:save_focus(model,sample,pred,i,focus_output,focus_frames)
        del sample,pred
        print(f'rank {rank}: {type(model).__name__} sample {i+1}/{len(dataset)}',flush=True)
    if world>1:
        parts=[None]*world;dist.all_gather_object(parts,rows)
        rows=sorted([row for part in parts for row in part],key=lambda r:r['id'])
    return rows


def evaluate(args,device,rank,world,dataset_name):
    state,cfg=load_checkpoint(args.checkpoint)
    checkpoint_crop=list(cfg['wide']['crop'])
    if dataset_name=='semireal':cfg=configure_scene(cfg,args.scene_config,args.scene_crop)
    folder=args.output/dataset_name
    if rank==0:folder.mkdir(parents=True,exist_ok=False)
    if world>1:dist.barrier()
    started=time.time();radar=RadarPhysics(cfg).to(device)
    settings=dict(dataset=dataset_name,channel_model='ideal',trajectories=args.trajectories,scr_db=args.scr_db,snr_db=args.snr_db,
        seed=args.seed,checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=file_hash(args.checkpoint),sampling=diagnostics(cfg))
    settings['traditional']=dict(algorithm_version=ALGORITHM_VERSION,iterations=args.traditional_iterations,
        velocity_initialization='Local quadratic derivative (up to 5 frames), projected onto observed ATI radial velocity',
        objective='Variable-projection least squares on orthonormal channel contrasts',
        safeguard='Direction-preserving step bounds and per-frame residual backtracking')
    if dataset_name=='semireal':
        settings['scene_selection']=dict(config=str(args.scene_config.resolve()),config_sha256=file_hash(args.scene_config),
            checkpoint_crop=checkpoint_crop,selected_crop=cfg['wide']['crop'],selection='current scene configuration with optional --scene-crop override')
        paths=list((PROJECT/'real_scene').rglob(cfg['wide']['scene_file']))
        if len(paths)!=1:raise ValueError('Expected one source GFF in real_scene/')
        scene=paths[0]
        points,coeff,field,_=scatter_field(scene,cfg['wide']['crop'],radar,spacing=cfg['wide']['scatter_spacing_uq_m'],half=cfg['wide']['background_half_uq_m'])
        background=project_background(radar,points,coeff,device)
        data=SemiRealDataset(cfg,background,args.trajectories,args.scr_db,args.snr_db,args.seed,half_uq=field['half_uq_m'])
        settings.update(scene=str(scene.resolve()),scene_sha256=file_hash(scene),field=field)
        del points,coeff
    else:data=SimulatedTestDataset(cfg,args.trajectories,args.scr_db,args.snr_db,args.seed)
    settings['samples']=len(data)
    settings['visualization']=dict(state_sample_ids=example_ids(data.entries),focus_sample_ids=example_ids(data.entries,args.focus_examples),
        focus_frames=args.focus_frames,selection='One SCR closest to 0 dB for each trajectory, without error ranking')
    del radar;gc.collect();methods={}
    for name in ('trained','fixed','traditional'):
        torch.manual_seed(cfg['seed'])
        if name=='traditional':
            model=TraditionalEstimator(cfg,device,args.traditional_iterations)
        else:
            model=JointUnfoldNet(cfg).to(device)
            if name=='trained':model.load_state_dict(state['model'])
        methods[name]=predict(model,data,device,rank,world,folder if name=='trained' else None,args.focus_frames,args.focus_examples)
        if name=='trained' and cfg['temporal']['enabled']:
            before=copy.deepcopy(methods[name]);platform=model.radar.platform.cpu().numpy()
            for row in before:
                row['position']=row['single_position'];row['velocity']=row['single_velocity']
                delta=np.asarray(row['position'])-platform[:,:2]
                line=delta/np.sqrt((delta*delta).sum(-1)+model.radar.height**2)[:,None]
                row['radial']=(line*np.asarray(row['velocity'])).sum(-1).tolist()
            methods['before']=before
        del model;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    if rank==0:
        save_configuration(folder,cfg);write_json(folder/'settings.json',settings)
        make_report(folder,methods,settings,cfg)
        write_json(folder/'runtime.json',dict(elapsed_s=time.time()-started,world_size=world,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None))
        print(f'Results: {folder}',flush=True)
    if world>1:dist.barrier()
    del data,methods;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
