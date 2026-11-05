#!/usr/bin/env python3
"""Compare reversible amplification and prefix-aware surname sampling bias."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sys
import time

import torch
from transformers import GPT2LMHeadModel

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
sys.path.insert(0,str(ROOT.parent/'kapustin-trace'))
from run_cpu import (DEFAULT_MODEL, OriginalYTEncoder, CachedNextLogits, build_filters,
                     filter_logits_batched, resolve_device, seed_runtime, trim_like_site,
                     generate_replies)
from trace import file_hash, parameter_hash


def save(name,value):
    (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


class NameBias:
    """Bias the next token along the longest matching surname prefix.

    This deliberately does not promise exact sequence-level reweighting:
    the empty prefix boosts the shared word-initial Ка token.
    """
    def __init__(self,paths,strength,first_only=False):
        self.strength=strength
        self.prefixes={}
        for path in paths:
            for i,token in enumerate(path):
                if first_only and i: continue
                self.prefixes.setdefault(tuple(path[:i]),set()).add(token)
        self.prefixes=sorted(self.prefixes.items(),key=lambda item:len(item[0]))
        self.max_prefix=max(len(p) for p,_ in self.prefixes)

    def __call__(self,logits,history):
        bonus=torch.zeros_like(logits)
        for prefix,next_tokens in self.prefixes:
            if len(prefix)>history.shape[1]: continue
            current=torch.zeros_like(logits)
            current[:,list(next_tokens)]=self.strength
            if not prefix:
                bonus=current
            else:
                match=(history[:,-len(prefix):]==torch.tensor(prefix,device=history.device)).all(-1)
                bonus=torch.where(match[:,None],current,bonus)
        return logits+bonus


@contextmanager
def amplify(model,variant):
    handle=None
    if variant['kind']=='mlp':
        factor=variant['factor']
        handle=model.transformer.h[variant['layer']].mlp.register_forward_hook(
            lambda module,inputs,output: output*factor)
    elif variant['kind']=='neuron':
        def hook(module,inputs):
            x=inputs[0].clone()
            x[...,variant['neuron']]*=variant['factor']
            return (x,)+inputs[1:]
        handle=model.transformer.h[variant['layer']].mlp.c_proj.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        if handle: handle.remove()


def biased_sequence(model,context,length,one,two,device,bias=None):
    generated=torch.empty((1,len(context)+length),dtype=torch.long,device=device)
    generated[0,:len(context)]=torch.tensor(context,device=device)
    forward=CachedNextLogits(model)
    for pos in range(len(context),len(context)+length):
        logits=forward(generated[:,:pos])
        if bias: logits=bias(logits,generated[:,:pos])
        logits=filter_logits_batched(logits,generated[:,pos-1],one,two,top_p=0.9)
        generated[0,pos]=torch.multinomial(logits[0].softmax(-1),1)[0]
    return generated[0].cpu().tolist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('discovery','test'),default='discovery')
    args=parser.parse_args()
    design=json.loads((ROOT/'design.json').read_text())
    device=resolve_device('auto')
    torch.set_num_threads(6)
    expected=next(l.split()[0] for l in (ROOT.parents[1]/'SHA256SUMS').read_text().splitlines() if 'pytorch_model.bin' in l)
    assert file_hash(DEFAULT_MODEL/'pytorch_model.bin')==expected
    model=GPT2LMHeadModel.from_pretrained(DEFAULT_MODEL,local_files_only=True,use_safetensors=False).to(device).eval()
    before=parameter_hash(model)
    tokenizer=OriginalYTEncoder(DEFAULT_MODEL/'encoder.model')
    one,two=build_filters(tokenizer,'latest-site',False)
    paths=[tokenizer.encode(s) for s in design['forms']]
    pattern=re.compile(r'\b(?:'+ '|'.join(map(re.escape,design['forms']))+r')\b',re.I)
    controls=re.compile(r'\b(?:Гаспаров\w*|Сидоров\w*|Петров\w*|Иванов\w*|Сар[аыуеой]+)\b')
    prefix_words=re.compile(r'\bКа[а-яА-ЯёЁ]*\b')
    baseline={'name':'baseline','kind':'none'}
    if args.phase=='discovery':
        variants=[baseline,
                  {'name':'prefix-bias-1.5','kind':'bias','strength':1.5},
                  {'name':'prefix-bias-3.0','kind':'bias','strength':3.0}]
        variants += [{'name':f'mlp-{layer}-x{factor}','kind':'mlp','layer':layer,'factor':factor}
                     for layer in (9,16) for factor in (1.25,1.5)]
        variants += [{'name':'neuron-9-3384-x2','kind':'neuron','layer':9,'neuron':3384,'factor':2.0}]
    else:
        selected=json.loads((ROOT/'selected.json').read_text())
        variants=[baseline, selected['bias'],selected['internal'],
                  {'name':'first-token-only','kind':'first_only','strength':selected['bias']['strength']}]
    prompts=design[args.phase+'_prompts']; seeds=design[args.phase+'_seeds']

    def distribution(text,bias):
        seq=tokenizer.encode(text)
        ids=torch.tensor([seq[:-1]],device=device)
        logits=model(ids,use_cache=False).logits[0]
        if bias:
            # One history per scored next token; left padding cannot match a
            # real name prefix because token 0 is absent from every prefix.
            width=max(1,bias.max_prefix)
            history=torch.nn.functional.pad(ids[0],(width-1,0)).unfold(0,width,1)
            logits=bias(logits,history)
        targets=torch.tensor(seq[1:],device=device)
        return logits.log_softmax(-1).gather(1,targets[:,None]).squeeze(1),seq

    def prefix_logp(prompt,target,bias):
        prefix=tokenizer.encode(prompt)
        scores,seq=distribution(prompt+' '+target,bias)
        assert seq[:len(prefix)]==prefix
        return scores[len(prefix)-1:].sum().item()

    results=[]
    with torch.inference_mode():
        # Verify the experimental baseline loop against the production runner.
        seed_runtime(42,device)
        prompt=prompts[0]; context=tokenizer.encode(prompt)
        tokens=biased_sequence(model,context,15,one,two,device)
        ours=trim_like_site(tokenizer.decode(tokens)[len(tokenizer.decode(context)):],'latest-site')
        seed_runtime(42,device)
        reference=generate_replies(model,tokenizer,prompt,15,1,'latest-site',one,two,device=device)[0]
        assert ours==reference
        for variant in variants:
            started=time.perf_counter()
            bias=NameBias(paths,variant['strength'],variant['kind']=='first_only') if variant['kind'] in ('bias','first_only') else None
            with amplify(model,variant):
                prose_nll=sum(-distribution(t,bias)[0].mean().item() for t in design['prose'])/len(design['prose'])
                scores=[]
                for prompt in prompts:
                    lp={t:prefix_logp(prompt,t,bias) for t in design['forms']+design['controls']}
                    family=torch.tensor([lp[t] for t in design['forms']],dtype=torch.float64).logsumexp(0).item()
                    scores.append({'prompt':prompt,'logp':lp,'family_logp':family,
                                   'family_vs_controls':family-sum(lp[t] for t in design['controls'][:5])/5})
                generations=[]
                for prompt in prompts:
                    context=tokenizer.encode(prompt)
                    for seed in seeds:
                        seed_runtime(seed,device)
                        ids=biased_sequence(model,context,design['length'],one,two,device,bias)
                        raw=tokenizer.decode(ids)[len(tokenizer.decode(context)):]
                        visible=trim_like_site(raw,'latest-site')
                        name_hits=pattern.findall(visible)
                        other_ca=[w for w in prefix_words.findall(visible) if not pattern.fullmatch(w)]
                        grams=[tuple(ids[i:i+4]) for i in range(len(context),len(ids)-3)]
                        generations.append({'prompt':prompt,'seed':seed,'continuation':visible,'raw_continuation':raw,
                             'name_mentions':len(name_hits),'name_forms':name_hits,'raw_name_mentions':len(pattern.findall(raw)),
                             'other_names':controls.findall(visible),'other_capital_ka_words':other_ca,
                             'repeated_4gram_fraction':1-len(set(grams))/len(grams) if grams else 0})
            n=len(generations)
            result={'variant':variant,'prose_nll':prose_nll,'scores':scores,'generations':generations,
                    'summary':{'continuations':n,'with_name':sum(g['name_mentions']>0 for g in generations),
                    'name_mentions':sum(g['name_mentions'] for g in generations),
                    'raw_with_name':sum(g['raw_name_mentions']>0 for g in generations),
                    'with_other_ka_word':sum(bool(g['other_capital_ka_words']) for g in generations),
                    'with_other_name':sum(bool(g['other_names']) for g in generations),
                    'mean_repeated_4gram_fraction':sum(g['repeated_4gram_fraction'] for g in generations)/n,
                    'mean_family_vs_controls':sum(s['family_vs_controls'] for s in scores)/len(scores)},
                    'seconds':time.perf_counter()-started}
            results.append(result)
            save(args.phase+'.json',{'phase':args.phase,'device':str(device),'torch':torch.__version__,'checkpoint_sha256':expected,'results':results})
            print(variant['name'],result['summary'],'NLL',round(prose_nll,4),flush=True)
        if args.phase=='discovery':
            biases=[r for r in results if r['variant']['kind']=='bias']
            selected_bias=max(biases,key=lambda r:(r['summary']['with_name'],-r['variant']['strength']))
            internals=[r for r in results if r['variant']['kind'] in ('mlp','neuron')]
            eligible=[r for r in internals if r['prose_nll']-results[0]['prose_nll']<=.1]
            selected_internal=max(eligible,key=lambda r:r['summary']['mean_family_vs_controls']) if eligible else min(internals,key=lambda r:r['prose_nll'])
            save('selected.json',{'bias':selected_bias['variant'],'internal':selected_internal['variant'],
                 'internal_passed_nll_gate':bool(eligible),'selection_rule':design['selection']})
        after=parameter_hash(model)
        assert after==before and file_hash(DEFAULT_MODEL/'pytorch_model.bin')==expected
        save(args.phase+'-integrity.json',{'checkpoint_sha256':expected,'parameter_sha256_before':before,
             'parameter_sha256_after':after,'baseline_matches_production_runner':True,'passed':True})


if __name__=='__main__': main()
