#!/usr/bin/env python3
"""Render the saved experiment without loading or modifying model weights."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
read=lambda name:json.loads((ROOT/name).read_text())
scan=read('layer-scan.json')
lens=read('logit-lens.json')['observed']
validations=read('validation.json')
neurons=read('neuron-validation.json')
fig,axes=plt.subplots(3,1,figsize=(12,11),layout='constrained')
colors={'attention':'#2978a0','mlp':'#d77332'}
for kind,offset in [('attention',-0.19),('mlp',0.19)]:
 rows=[x for x in scan if x['kind']==kind]
 axes[0].bar(np.arange(24)+offset,[100*(r['kapustin_probability_ratio']-1) for r in rows],width=.38,label=kind.upper(),color=colors[kind])
axes[0].axhline(0,color='black',lw=.6)
axes[0].set_xticks(range(24))
axes[0].set(xlabel='Layer (zero-based)',ylabel='Target prefix probability change (%)',title='Half-scaling each component: observed passage only (not name-specificity)')
axes[0].legend()
for i,label in enumerate(['Ка: first fragment (shared)','пусти | Ка: spelling continuation','н | Капусти: ending']):
 axes[1].plot(range(24),[math.exp(row['scores']['Капустин']['token_logp'][i]) for row in lens],marker='.',label=label)
axes[1].set_yscale('log')
axes[1].set(xlabel='Layer (zero-based)',ylabel='Conditional token probability',title='Logit-lens readout: diagnostic projection, not a causal attribution')
axes[1].legend(loc='lower right')
keys=['observed','observed_without_gasparov','short_clause','phone','door','report','genitive','dative','copy_positive_control']
labels=['Observed','Name swapped','Short clause','Phone','Door','Report','Genitive','Dative','Repeated name']
rows=[]; names=[]
for layer in [9,16]:
 x=next(x for x in validations if x['component']['kind']=='mlp' and x['component']['layer']==layer)
 rows.append([x['contexts'][k]['kapustin_probability_ratio'] for k in keys]); names.append(f'MLP {layer}')
x=next(x for x in validations if x['component']['kind']=='head' and x['component']['layer']==15 and x['component']['index']==1)
rows.append([x['contexts'][k]['kapustin_probability_ratio'] for k in keys]); names.append('Attention head 15.1')
for x in neurons:
 rows.append([x['contexts'][k]['kapustin_probability_ratio'] for k in keys]); names.append(f"MLP {x['candidate']['layer']} neuron {x['candidate']['neuron']}")
ratios=np.array(rows)
im=axes[2].imshow(np.log2(ratios),aspect='auto',cmap='RdBu_r',vmin=-.7,vmax=.7)
axes[2].set_xticks(range(len(keys)),labels,rotation=25,ha='right')
axes[2].set_yticks(range(len(names)),names)
for i in range(len(names)):
 for j in range(len(keys)):
  axes[2].text(j,i,f'{ratios[i,j]:.2f}×',ha='center',va='center',fontsize=9)
axes[2].set_title('Validation across contexts: ratio to unchanged model (1.00× = no effect)')
fig.colorbar(im,ax=axes[2],label='log₂ probability ratio',shrink=.8)
fig.suptitle('Where does Капустин come from? Untouched Porfirievich, temporary activation interventions',fontsize=14)
fig.savefig(ROOT/'trace.png',dpi=180)
fig.savefig(ROOT/'trace.svg')
