#!/usr/bin/env python3
"""Run the two-drone collective-blanket benchmark.

Author: Luca M. Possati
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import platform
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from active_drone.collective import (COLLECTIVE_CONDITIONS, COLLECTIVE_TASKS,
                                     make_collective_system, run_collective_mission)
from active_drone.graphics import BLUE, CYAN, GREEN, GREY, NAVY, ORANGE, PURPLE, RED, style

LABELS={"collective_constitutive":"Collective constitutive","collective_uniform_posterior":"Uniform posterior",
        "collective_diagnostic":"Collective diagnostic","individual_constitutive":"Local / independent",
        "local_shared_policy":"Local / shared policy","pooled_independent_policy":"Pooled / independent",
        "fixed_team":"Fixed team","communication_only":"Communication only",
        "random_joint":"Random joint","oracle_collective":"Oracle"}
COLORS={"collective_constitutive":BLUE,"collective_uniform_posterior":ORANGE,"collective_diagnostic":RED,
        "individual_constitutive":PURPLE,"local_shared_policy":CYAN,"pooled_independent_policy":"#5B8FF9",
        "fixed_team":"#94A3B8","communication_only":GREY,"random_joint":"#13B5A4","oracle_collective":GREEN}


def run_one(spec):
    condition,task,seed=spec; world,team=make_collective_system(seed,condition,task)
    return run_collective_mission(world,team)


def mean_se(values):
    v=np.asarray(values,float); v=v[np.isfinite(v)]
    if len(v)==0:return float("nan"),float("nan")
    return float(v.mean()),float(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.


def write_csv(path,rows):
    with Path(path).open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


def architecture_figure(path:Path):
    style(); fig,ax=plt.subplots(figsize=(15.5,7.5)); ax.set(xlim=(0,15.5),ylim=(0,7.5)); ax.axis("off")
    boxes=[(.5,4.6,2.5,"Drone A\npartial paths",PURPLE),(3.45,4.6,2.5,"Local posterior\n$q^A(M)$",BLUE),
           (9.55,4.6,2.5,"Local posterior\n$q^B(M)$",BLUE),(12.5,4.6,2.5,"Drone B\npartial paths",PURPLE),
           (5.9,2.55,3.7,"Evidence pooling\n$q^{AB}(M)$",CYAN),(5.9,.55,3.7,"Joint EFE + handshake\n$(u_A,u_B,d_A,d_B)$",GREEN)]
    for x,y,w,label,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,1.25,boxstyle="round,pad=.12,rounding_size=.12",fc=color,ec="none")); ax.text(x+w/2,y+.625,label,ha="center",va="center",color="white",fontweight="bold",fontsize=11)
    arrows=[((3.0,5.22),(3.4,5.22)),((12.5,5.22),(12.1,5.22)),((4.7,4.55),(6.55,3.83)),((10.8,4.55),(8.95,3.83)),((7.75,2.48),(7.75,1.86))]
    for a,b in arrows: ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=14,lw=2,color=NAVY))
    ax.add_patch(FancyArrowPatch((5.82,1.17),(2.15,3.95),arrowstyle="-|>",mutation_scale=14,lw=1.8,color=GREEN,connectionstyle="arc3,rad=-.18"))
    ax.add_patch(FancyArrowPatch((9.68,1.17),(13.35,3.95),arrowstyle="-|>",mutation_scale=14,lw=1.8,color=GREEN,connectionstyle="arc3,rad=.18"))
    ax.text(7.75,7.02,"A collective Markov blanket is inferred and enacted by both drones",ha="center",fontsize=19,fontweight="bold")
    ax.text(7.75,6.57,"No unilateral proposal changes the group organization",ha="center",fontsize=12,color=GREY)
    fig.savefig(path,dpi=220,bbox_inches="tight"); plt.close(fig)


def benchmark_figure(summary,path:Path):
    style(); shown=("collective_constitutive","collective_diagnostic","individual_constitutive","communication_only","random_joint","oracle_collective")
    fig,axes=plt.subplots(2,2,figsize=(15.5,10),constrained_layout=True)
    metrics=(("success_rate","Collective task success",100,"%"),("viability","Group viability",1,"score"),
             ("formation_rmse","Formation error",1,"RMSE"),("architecture_score","Group architecture score",1,"score"))
    x=np.arange(3); width=.13
    for ax,(metric,title,scale,ylabel) in zip(axes.flat,metrics):
        for k,c in enumerate(shown):
            vals=[next(r[metric] for r in summary if r["condition"]==c and r["task"]==t)*scale for t in COLLECTIVE_TASKS]
            errs=[next(r[metric+"_se"] for r in summary if r["condition"]==c and r["task"]==t)*scale for t in COLLECTIVE_TASKS]
            ax.bar(x+(k-2.5)*width,vals,width,yerr=errs,capsize=2,color=COLORS[c],alpha=.9,label=LABELS[c])
        ax.set_xticks(x,[t.capitalize() for t in COLLECTIVE_TASKS]); ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y"); ax.spines[["top","right"]].set_visible(False)
        if scale==100:ax.set_ylim(0,105)
    axes[0,0].legend(ncol=2,fontsize=8)
    fig.suptitle("Strategic value of a jointly inferred and constituted group boundary",fontsize=17,fontweight="bold")
    fig.savefig(path,dpi=220,bbox_inches="tight"); plt.close(fig)


def matrix_figure(summary,path:Path):
    style(); fig,axes=plt.subplots(1,3,figsize=(16,6.5),constrained_layout=True)
    for ax,(metric,title,scale) in zip(axes,(('success_rate','Success',100),('architecture_score','Architecture',1),('viability','Viability',1))):
        m=np.array([[next(r[metric] for r in summary if r['condition']==c and r['task']==t) for t in COLLECTIVE_TASKS] for c in COLLECTIVE_CONDITIONS])*scale
        im=ax.imshow(m,cmap='YlGnBu',vmin=0,vmax=100 if scale==100 else 1,aspect='auto'); ax.set_title(title); ax.set_xticks(range(3),[t.capitalize() for t in COLLECTIVE_TASKS]); ax.set_yticks(range(len(COLLECTIVE_CONDITIONS)),[LABELS[c] for c in COLLECTIVE_CONDITIONS])
        for i in range(len(COLLECTIVE_CONDITIONS)):
            for j in range(3):
                v=m[i,j]; ax.text(j,i,f'{v:.0f}%' if scale==100 else f'{v:.3f}',ha='center',va='center',fontsize=8,fontweight='bold',color='white' if v>(58 if scale==100 else .58) else NAVY)
        fig.colorbar(im,ax=ax,shrink=.78,pad=.02)
    fig.suptitle('Collective-boundary ablation matrix · paired seeds',fontsize=17,fontweight='bold'); fig.savefig(path,dpi=220,bbox_inches='tight'); plt.close(fig)


def example_figure(examples,path:Path):
    style(); fig,axes=plt.subplots(3,2,figsize=(15.5,11),constrained_layout=True)
    for row,task in enumerate(COLLECTIVE_TASKS):
        r=examples[task].records; t=np.arange(len(r)); gap=np.array([x['factorization_gap'] for x in r]); payload=np.array([x['payload'] for x in r]); form=np.array([x['formation_error'] for x in r]); changed=np.array([x['design_changed_causal_matrix'] for x in r],bool); entropy_v=np.array([x['pooled_entropy'] for x in r])
        ax=axes[row,0]; ax.plot(t,gap,color=RED,lw=2,label='Group path gap'); ax.plot(t,entropy_v,color=PURPLE,lw=1.3,label='Posterior entropy'); ax.scatter(t[changed],gap[changed],marker='D',s=38,color=GREEN,label='Bilateral redesign'); ax.set(title=f'{task.capitalize()}: collective boundary',ylabel='Gap / entropy'); ax.grid(True)
        ax=axes[row,1]; ax.plot(t,payload,color=BLUE,lw=2,label='Shared payload'); ax.plot(t,form,color=ORANGE,lw=1.5,label='Formation error'); ax.axhline(1.55,color=RED,ls='--',lw=1); ax.axhline(-1.55,color=RED,ls='--',lw=1); ax.set(title=f'{task.capitalize()}: strategic performance',ylabel='Displacement'); ax.grid(True)
        if row==0: axes[row,0].legend(ncol=3,fontsize=8); axes[row,1].legend(ncol=2,fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Time step')
    fig.suptitle('The group boundary is constructed, repaired, and reconfigured by agreement',fontsize=17,fontweight='bold'); fig.savefig(path,dpi=220,bbox_inches='tight'); plt.close(fig)


def paired_advantage(rows,path:Path):
    style(); fig,axes=plt.subplots(1,3,figsize=(16,5.3),constrained_layout=True)
    for ax,task in zip(axes,COLLECTIVE_TASKS):
        seeds=sorted({r['seed'] for r in rows if r['task']==task})
        for seed in seeds:
            base=next(r for r in rows if r['condition']=='communication_only' and r['task']==task and r['seed']==seed); full=next(r for r in rows if r['condition']=='collective_constitutive' and r['task']==task and r['seed']==seed)
            ax.plot([0,1],[base['viability'],full['viability']],color='#CBD5E1',lw=1); ax.scatter(0,base['viability'],color=GREY,s=22); ax.scatter(1,full['viability'],color=BLUE,s=25)
        ax.set_xticks([0,1],['Communication\nonly','Collective MB']); ax.set_ylim(0,1.02); ax.set_title(task.capitalize()); ax.set_ylabel('Group viability'); ax.grid(axis='y'); ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Paired strategic advantage over equal-bandwidth communication',fontsize=17,fontweight='bold'); fig.savefig(path,dpi=220,bbox_inches='tight'); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--seeds',type=int,default=20); p.add_argument('--seed-offset',type=int,default=2000); p.add_argument('--workers',type=int,default=1); p.add_argument('--quick',action='store_true'); p.add_argument('--output',type=Path,default=Path('collective_results')); args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if args.quick:args.seeds=min(args.seeds,4);args.seed_offset=0
    specs=[(c,t,args.seed_offset+k) for c in COLLECTIVE_CONDITIONS for t in COLLECTIVE_TASKS for k in range(args.seeds)]
    if args.workers>1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool: results=list(pool.map(run_one,specs,chunksize=1))
    else:results=[run_one(s) for s in specs]
    rows=[];examples={}
    for (c,t,seed),r in zip(specs,results):
        rows.append({'condition':c,'task':t,'seed':seed,'success':r.success,'collision':r.collision,'viability':r.viability,'payload_rmse':r.payload_rmse,'formation_rmse':r.formation_rmse,'energy':r.energy,'final_gap':r.final_gap,'architecture_score':r.architecture_score,'role_accuracy':r.role_accuracy,'blanket_accuracy':r.blanket_accuracy,'exact_partition':r.exact_partition,'agreements':r.agreements,'causal_changes':r.causal_changes,'synergy':r.synergy})
        if c=='collective_constitutive' and t not in examples and r.success:examples[t]=r
    summary=[]
    metrics=('success','collision','viability','payload_rmse','formation_rmse','energy','final_gap','architecture_score','role_accuracy','blanket_accuracy','exact_partition','agreements','causal_changes','synergy')
    for c in COLLECTIVE_CONDITIONS:
        for t in COLLECTIVE_TASKS:
            subset=[r for r in rows if r['condition']==c and r['task']==t]; out={'condition':c,'task':t,'missions':len(subset)}
            for m in metrics:
                mean,se=mean_se([r[m] for r in subset]); key={'success':'success_rate','collision':'collision_rate'}.get(m,m);out[key]=mean;out[key+'_se']=se
            summary.append(out)
    paired=[]
    for task in COLLECTIVE_TASKS:
        for comparator in ('collective_diagnostic','individual_constitutive','communication_only','random_joint'):
            for metric in ('success','viability','architecture_score','formation_rmse','energy'):
                d=[]
                for k in range(args.seeds):
                    seed=args.seed_offset+k; f=next(r[metric] for r in rows if r['condition']=='collective_constitutive' and r['task']==task and r['seed']==seed); b=next(r[metric] for r in rows if r['condition']==comparator and r['task']==task and r['seed']==seed);d.append(f-b)
                mean,se=mean_se(d);paired.append({'task':task,'comparison':f'collective_constitutive - {comparator}','metric':metric,'mean_difference':mean,'standard_error':se})
    write_csv(args.output/'mission_level.csv',rows);write_csv(args.output/'summary.csv',summary);write_csv(args.output/'paired_effects.csv',paired)
    meta={'version':'0.5.0-collective','author':'Luca M. Possati','seeds':args.seeds,'seed_offset':args.seed_offset,'missions':len(rows),'conditions':COLLECTIVE_CONDITIONS,'tasks':COLLECTIVE_TASKS,'paired':True,'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'quick':args.quick};(args.output/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    architecture_figure(args.output/'collective_architecture.png');benchmark_figure(summary,args.output/'collective_benchmark.png');matrix_figure(summary,args.output/'collective_ablation_matrix.png');paired_advantage(rows,args.output/'collective_strategic_advantage.png')
    if len(examples)==3:example_figure(examples,args.output/'collective_examples.png')
    print(json.dumps({'summary':summary,'paired_effects':paired},indent=2))


if __name__=='__main__':main()
