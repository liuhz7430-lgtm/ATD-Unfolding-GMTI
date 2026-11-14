"""Ideal-channel mean errors, source annotation, and state/refocus examples."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .artifacts import write_json
from .gff import read_gff,display_db
from .semireal import source_pixels
from .focus import example_ids
import json

LABELS={'traditional':'DPCA/ATI + GN','fixed':'Fixed unfolding','before':'Joint model before refinement',
        'single':'Independently trained single-frame','trained':'Trained temporal'}


def annotation(output,scene,field,rows,cfg):
    """Full source -> selected native crop -> same crop with all target paths."""
    from matplotlib.patches import Rectangle,ConnectionPatch
    image,_=read_gff(scene);r0,r1,a0,a1=field['crop'];patch=image[r0:r1,a0:a1]
    fig,axes=plt.subplots(1,3,figsize=(19,6))
    axes[0].imshow(display_db(image),cmap='gray',vmin=-40,vmax=0,interpolation='none')
    axes[0].add_patch(Rectangle((a0,r0),a1-a0,r1-r0,fill=False,edgecolor='tab:orange',linewidth=2))
    axes[0].set_title('Full measured scene and selected region')
    hu,hq=field['half_uq_m']
    low,high=source_pixels(np.array([[-hu,-hq],[hu,hq]]),field)
    axes[0].add_patch(Rectangle(low,*(high-low),fill=False,edgecolor='cyan',linestyle='--',linewidth=1.4))
    for ax in axes[1:]:
        ax.imshow(display_db(patch),cmap='gray',vmin=-40,vmax=0,interpolation='none',
                  extent=(a0-.5,a1-.5,r1-.5,r0-.5))
        ax.set_xlim(a0-.5,a1-.5);ax.set_ylim(r1-.5,r0-.5)
        ax.add_patch(Rectangle(low,*(high-low),fill=False,edgecolor='cyan',linestyle='--',linewidth=1.4,label=f'{2*hu:g} x {2*hq:g} m simulated ROI'))
    axes[1].set_title('Selected native region; cyan = simulated ROI')
    axes[1].legend(fontsize=8,loc='upper right')
    axes[2].set_title('Injected target trajectories (separate samples)')
    theta=np.deg2rad(cfg['radar']['initial_angle_deg'])
    basis=np.array([[-np.sin(theta),np.cos(theta)],[np.cos(theta),np.sin(theta)]])
    seen=set()
    for row in rows:
        trajectory=row['info']['trajectory']
        if trajectory in seen:continue
        seen.add(trajectory)
        pixels=source_pixels(np.asarray(row['true_position'])@basis,field)
        color=f'C{trajectory%10}';ax=axes[2]
        ax.plot(pixels[:,0],pixels[:,1],'.-',color=color,linewidth=1.5,markersize=2.5,label=f'Trajectory {trajectory}')
        ax.scatter(*pixels[0],s=45,marker='o',facecolors='none',edgecolors=color)
        ax.scatter(*pixels[-1],s=65,marker='*',color=color,edgecolors='black',linewidths=.4)
    axes[2].legend(fontsize=8)
    for ax in axes:ax.set(xlabel='Original azimuth pixel',ylabel='Original range pixel')
    fig.add_artist(ConnectionPatch(xyA=(a1,r0),coordsA=axes[0].transData,
        xyB=(a0,r0),coordsB=axes[1].transData,color='tab:orange',alpha=.65))
    fig.tight_layout();fig.savefig(output/'figures/scene_annotation.png',dpi=220);plt.close(fig)


def mean_report(output,methods,settings,cfg):
    folder=output/'figures';folder.mkdir(parents=True,exist_ok=True)
    reference=methods['trained'];ids=[r['id'] for r in reference]
    expected=set(range(settings['samples']))
    if len(ids)!=len(set(ids)) or set(ids)!=expected:raise ValueError('Incomplete evaluation sample coverage')
    for name,rows in methods.items():
        if [r['id'] for r in rows]!=ids:raise ValueError('Unpaired evaluation sample IDs')
        for row,other in zip(rows,reference):
            if row['info']!=other['info']:raise ValueError('Methods evaluated different echoes')
            for key in ('true_position','true_velocity'):
                np.testing.assert_array_equal(row[key],other[key])
    table=[]
    modes=sorted({r['info']['mode'] for r in reference})
    scr_values=sorted({r['info']['scr_db'] for r in reference})
    for name,rows in methods.items():
        for mode in modes:
            for scr in scr_values:
                group=[r for r in rows if (r['info']['mode'],r['info']['scr_db'])==(mode,scr)]
                for key in ('position','velocity'):
                    error=np.asarray([r[key] for r in group],dtype=float)-np.asarray([r['true_'+key] for r in group],dtype=float)
                    norm=np.linalg.norm(error,axis=-1);invalid=int((~np.isfinite(norm)).sum())
                    # A failed frame makes this group's mean unavailable; no failed-frame filtering.
                    table.append(dict(method=name,mode=mode,echo_scr_db=scr,quantity=key,
                        mean_error=float(norm.mean()) if not invalid else float('nan'),
                        trajectories=len(group),frames=int(norm.size),invalid_frames=invalid))
    write_json(output/'mean_errors.json',table)
    with (output/'mean_errors.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    for key,unit in (('position','m'),('velocity','m/s')):
        fig,axes=plt.subplots(1,len(modes),figsize=(6*len(modes),4.8),squeeze=False,sharey=True)
        for ax,mode in zip(axes[0],modes):
            for name in methods:
                selected=[next(r for r in table if (r['method'],r['mode'],r['echo_scr_db'],r['quantity'])==(name,mode,scr,key)) for scr in scr_values]
                ax.plot(scr_values,[r['mean_error'] for r in selected],'o-',label=LABELS[name])
                failures=sum(r['invalid_frames'] for r in selected)
                if failures:ax.text(.02,.98-.06*list(methods).index(name),f'{name}: {failures} failed frames',transform=ax.transAxes,va='top',fontsize=8)
            ax.set(title=mode,xlabel='Global echo SCR (dB)',ylabel=f'Mean 2D {key} error ({unit})')
            ax.set_xticks(scr_values);ax.grid(alpha=.3);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(folder/f'mean_{key}_error.png',dpi=200);plt.close(fig)
    if settings['dataset']=='semireal':annotation(output,settings['scene'],settings['field'],reference,cfg)
    write_json(output/'predictions.json',methods)
    lines=['# 大区域测试：平均误差','',f"采样 {cfg['radar']['frequency_samples']}×{cfg['radar']['pulses']}，带宽 {cfg['radar']['bandwidth_hz']/1e6:g} MHz；配置以 config.yaml 为准。",
        '每组平均误差 = 对所有轨迹、所有帧的二维误差模长求算术平均；不是 RMSE，也不是速度模长之差。',
        '本版本仅测试ideal通道，各档SCR共用横轴。组内有失败帧时均值记为不可用，不删除失败样本。',
        'before 来自同一个联合模型的多帧修正前输出；single（若提供）来自独立训练的单帧检查点。',
        'SCR 为全回波目标/背景平均功率比；半实测背景是实测复图像驱动的等效散射场，不是原始多通道实测回波。','',
        '| 方法 | 通道 | 回波 SCR (dB) | 指标 | 平均误差 | 失败帧 |','|---|---|---:|---|---:|---:|']
    for r in table:lines.append(f"| {r['method']} | {r['mode']} | {r['echo_scr_db']:g} | {r['quantity']} | {r['mean_error']:.6g} | {r['invalid_frames']} |")
    (output/'TEST_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def _draw_focus(output,records,filename):
    frames=sorted({k for r in records for k in r['frames']})
    fig,axes=plt.subplots(len(frames),2*len(records),figsize=(8*len(records),3.3*len(frames)),squeeze=False,layout='constrained')
    for column,record in enumerate(records):
        for row,k in enumerate(frames):
            path=output/'data/focus'/f'sample_{record["sample_id"]:04d}'/f'frame_{k:03d}.npz'
            if not path.exists():
                for j in range(2):axes[row,2*column+j].text(.5,.5,'Failed estimate',ha='center')
                continue
            with np.load(path,allow_pickle=False) as d:
                ref=max(np.abs(d['before']).max()**2,np.abs(d['after']).max()**2,1e-20)
                for j,key in enumerate(('before','after')):
                    ax=axes[row,2*column+j];u=d['u_'+key];q=d['q_'+key]
                    db=10*np.log10(np.maximum(np.abs(d[key])**2/ref,1e-12))
                    ax.pcolormesh(u,q,db,shading='auto',cmap='viridis',vmin=-35,vmax=0,rasterized=True)
                    ax.plot(*d['centers_uq'][j],'w+',label='Estimate')
                    truth=d['truth_uq']
                    if u.min()<=truth[0]<=u.max() and q.min()<=truth[1]<=q.max():ax.plot(*truth,'rx',label='Truth')
                    else:ax.text(.02,.95,'Truth outside window',transform=ax.transAxes,va='top',fontsize=7,color='white')
                    label='Stationary imaging' if j==0 else 'Estimated CV refocus'
                    info=record['info']
                    ax.set(title=f'Trajectory {info["trajectory"]} | {info["mode"]}, SCR {info["scr_db"]:g} dB | Frame {k+1}\n{label}',xlabel='Local u (m)',ylabel='Local q (m)')
                    ax.legend(fontsize=7,loc='lower right')
    fig.suptitle('Same residual echo; equal-size estimate-centered windows; shared power scale within each frame pair')
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-35,0),cmap='viridis'),ax=axes.ravel().tolist(),label='Relative power (dB)',shrink=.65)
    filename.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(filename,dpi=180);plt.close(fig)


def render_focus(output):
    records=[json.loads(path.read_text()) for path in sorted((output/'data/focus').glob('sample_*/selection.json'))]
    if not records:return
    records.sort(key=lambda r:(r['info']['mode'],r['info']['trajectory']))
    # Two trajectories per overview page keep labels and image details readable.
    for start in range(0,len(records),2):
        name='refocus_selected_frames.png' if start==0 else f'refocus_selected_frames_page_{start//2+1:02d}.png'
        _draw_focus(output,records[start:start+2],output/'figures'/name)
    for record in records:
        trajectory=record['info']['trajectory']
        _draw_focus(output,[record],output/'figures'/f'trajectory_{trajectory:03d}'/'refocus_selected_frames.png')
    write_json(output/'focus_selection.json',records)


def _draw_states(output,methods,cfg,selected,position_file,velocity_file):
    reference=methods['trained']
    theta=np.deg2rad(cfg['radar']['initial_angle_deg'])
    basis=np.array([[-np.sin(theta),np.cos(theta)],[np.cos(theta),np.sin(theta)]])
    frames=len(reference[0]['position'])
    times=(np.arange(frames)-(frames-1)/2)*cfg['radar']['frame_interval_s']
    columns=min(2,len(selected));rows=(len(selected)+columns-1)//columns
    fig,axes=plt.subplots(rows,columns,figsize=(6*columns,5*rows),squeeze=False,layout='constrained')
    for ax,i in zip(axes.flat,selected):
        truth=np.asarray(reference[i]['true_position'])@basis
        ax.plot(*truth.T,'k.-',linewidth=2,label='Truth')
        for name,values in methods.items():
            estimate=np.asarray(values[i]['position'])@basis
            ax.plot(*estimate.T,'.-',markersize=3,alpha=.8,label=LABELS[name])
        info=reference[i]['info']
        ax.set(title=f'Trajectory {info["trajectory"]} | {info["mode"]} | SCR {info["scr_db"]:g} dB',xlabel='Local u (m)',ylabel='Local q (m)')
        ax.grid(alpha=.3);ax.legend(fontsize=7);ax.set_aspect('equal',adjustable='datalim')
    for ax in list(axes.flat)[len(selected):]:ax.set_visible(False)
    position_file.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(position_file,dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2*rows,columns,figsize=(6*columns,6.5*rows),squeeze=False,layout='constrained')
    for offset,i in enumerate(selected):
        row,column=divmod(offset,columns)
        truth=np.asarray(reference[i]['true_velocity'])@basis
        for component in range(2):
            ax=axes[2*row+component,column]
            ax.plot(times,truth[:,component],'k.-',linewidth=2,label='Truth')
            for name,values in methods.items():
                estimate=np.asarray(values[i]['velocity'])@basis
                ax.plot(times,estimate[:,component],'.-',markersize=3,alpha=.8,label=LABELS[name])
            info=reference[i]['info']
            ax.set(title=f'Trajectory {info["trajectory"]} | {info["mode"]} | SCR {info["scr_db"]:g} dB',xlabel='Frame-center time (s)',ylabel=f'Local {"u" if component==0 else "q"} velocity (m/s)')
            ax.grid(alpha=.3);ax.legend(fontsize=7)
    for offset in range(len(selected),rows*columns):
        row,column=divmod(offset,columns)
        for component in range(2):axes[2*row+component,column].set_visible(False)
    fig.savefig(velocity_file,dpi=180);plt.close(fig)


def render_states(output,methods,cfg):
    selected=example_ids([r['info'] for r in methods['trained']])
    if not selected:return
    for start in range(0,len(selected),4):
        suffix='' if start==0 else f'_page_{start//4+1:02d}'
        _draw_states(output,methods,cfg,selected[start:start+4],output/'figures'/f'position_estimates{suffix}.png',output/'figures'/f'velocity_estimates{suffix}.png')
    for i in selected:
        trajectory=methods['trained'][i]['info']['trajectory']
        folder=output/'figures'/f'trajectory_{trajectory:03d}'
        _draw_states(output,methods,cfg,[i],folder/'position_estimates.png',folder/'velocity_estimates.png')
    write_json(output/'state_plot_selection.json',[dict(methods['trained'][i]['info'],sample_id=methods['trained'][i]['id']) for i in selected])

def make_report(output,methods,settings,cfg):
    mean_report(output,methods,settings,cfg)
    render_states(output,methods,cfg)
    render_focus(output)
    with (output/'TEST_REPORT.md').open('a',encoding='utf-8') as stream:
        stream.write('\n位置/速度图覆盖全部轨迹，每条轨迹选择最接近0 dB的SCR，仅ideal通道；不按误差挑选。每条轨迹的独立图片位于figures/trajectory_NNN/，总览图按页保存。\n')
        stream.write('重聚焦使用同一前端目标残差，补偿速度由模型估计；真值只作标记。前后局部窗各以对应估计为中心，窗尺寸一致，每帧共用功率标尺。所有样本仍计入平均误差。\n')
