#!/usr/bin/env python3
"""Separate name initiation from spelling completion and refine one MLP group."""
import json
import math
from pathlib import Path
import sys

import torch
from transformers import GPT2LMHeadModel

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from run_cpu import DEFAULT_MODEL, OriginalYTEncoder, resolve_device
from trace import TARGETS, CONTROLS, attenuate, file_hash, parameter_hash, save


def main():
    design=json.loads((ROOT/'design.json').read_text())
    assert file_hash(DEFAULT_MODEL/'pytorch_model.bin') == design['checkpoint_sha256']
    device=resolve_device(design['device'])
    torch.set_num_threads(6)
    model=GPT2LMHeadModel.from_pretrained(DEFAULT_MODEL,local_files_only=True,use_safetensors=False).to(device).eval()
    before=parameter_hash(model)
    tokenizer=OriginalYTEncoder(DEFAULT_MODEL/'encoder.model')
    contexts=design['contexts']

    def score(context,target):
        prefix=tokenizer.encode(context)
        full=tokenizer.encode(context+' '+target)
        assert full[:len(prefix)]==prefix
        ids=torch.tensor([full[:-1]],device=device)
        hidden=model.transformer(ids,use_cache=False).last_hidden_state[:,len(prefix)-1:]
        logits=model.lm_head(hidden)
        targets=torch.tensor(full[len(prefix):],device=device)
        logp=logits.log_softmax(-1)[0].gather(-1,targets[:,None]).squeeze(-1).cpu().tolist()
        return {'token_logp':logp,'logp':sum(logp)}

    with torch.inference_mode():
        stages=[]
        for layer in (9,16):
            for key in ('observed','phone','genitive','dative'):
                for target in TARGETS[:3]:
                    baseline=score(contexts[key],target)
                    with attenuate(model,'mlp',layer):
                        edited=score(contexts[key],target)
                    stages.append({'layer':layer,'context':key,'target':target,
                                   'baseline':baseline,'attenuated':edited,
                                   'token_probability_ratios':[math.exp(b-a) for a,b in zip(baseline['token_logp'],edited['token_logp'])]})
        save('token-stages.json',stages)
        # Prespecified refinement rule: among already selected groups, inspect
        # the one with the greatest absolute target suppression at discovery.
        groups=[x for x in json.loads((ROOT/'selected-components.json').read_text()) if x['kind']=='mlp_group']
        group=min(groups,key=lambda x:x['delta_logp']['Капустин'])
        layer=group['layer']; first=group['index']*64
        baseline=score(contexts['observed'],'Капустин')
        scan=[]
        for neuron in range(first,first+64):
            with attenuate(model,'neuron',layer,neuron):
                result=score(contexts['observed'],'Капустин')
            delta=result['logp']-baseline['logp']
            scan.append({'layer':layer,'neuron':neuron,'delta_logp':delta,'ratio':math.exp(delta)})
        save('neuron-scan.json',{'selection_rule':'Strongest absolute target suppression among previously selected 64-neuron groups; then strongest two individual suppressors.',
                                'group':group,'neurons':scan})
        selected=sorted(scan,key=lambda x:x['delta_logp'])[:2]
        validation=[]
        for candidate in selected:
            cases={}
            for key,context in contexts.items():
                changes={}
                for target in TARGETS:
                    base=score(context,target)
                    with attenuate(model,'neuron',candidate['layer'],candidate['neuron']):
                        modified=score(context,target)
                    changes[target]=modified['logp']-base['logp']
                cases[key]={'delta_logp':changes,
                            'kapustin_probability_ratio':math.exp(changes['Капустин']),
                            'specificity_delta':changes['Капустин']-sum(changes[t] for t in CONTROLS)/len(CONTROLS)}
            validation.append({'candidate':candidate,'contexts':cases})
            save('neuron-validation.json',validation)
            print('Validated neuron',candidate,flush=True)
        after=parameter_hash(model)
        assert after==before
        assert file_hash(DEFAULT_MODEL/'pytorch_model.bin')==design['checkpoint_sha256']
        restored=score(contexts['observed'],'Капустин')
        error=abs(restored['logp']-baseline['logp'])
        assert error<1e-5
        save('refinement-integrity.json',{'parameter_sha256_before':before,'parameter_sha256_after':after,
             'checkpoint_sha256':design['checkpoint_sha256'],'restored_logp_error':error,'passed':True})


if __name__=='__main__': main()
