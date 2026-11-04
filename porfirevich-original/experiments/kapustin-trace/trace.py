#!/usr/bin/env python3
"""Read-only causal localization of a surname in the untouched checkpoint."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import transformers
from transformers import GPT2LMHeadModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from run_cpu import DEFAULT_MODEL, OriginalYTEncoder, resolve_device

TARGETS = ['Капустин', 'Капустина', 'Капустину', 'Гаспаров', 'Сидоров', 'Петров', 'Иванов', 'Сара', 'Капитан', 'капуста', 'человек']
CONTROLS = ['Гаспаров', 'Сидоров', 'Петров', 'Иванов', 'Сара']


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def parameter_hash(model):
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        h.update(name.encode())
        h.update(param.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@contextmanager
def attenuate(model, kind, layer, index=None):
    """Half-scale temporary activations at ALL positions, never weights."""
    block = model.transformer.h[layer]
    if kind == 'mlp':
        handle = block.mlp.register_forward_hook(lambda module, inputs, output: output * 0.5)
    elif kind == 'attention':
        handle = block.attn.register_forward_hook(lambda module, inputs, output: (output[0] * 0.5,) + output[1:])
    else:
        module = block.attn.c_proj if kind == 'head' else block.mlp.c_proj
        width = model.config.n_embd // model.config.n_head if kind == 'head' else (1 if kind == 'neuron' else 64)
        def hook(module, inputs):
            x = inputs[0].clone()
            x[..., index * width:(index + 1) * width] *= 0.5
            return (x,) + inputs[1:]
        handle = module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='mps', choices=('cpu','mps'))
    args = parser.parse_args()
    device = resolve_device(args.device)
    torch.set_num_threads(6)
    expected = next(l.split()[0] for l in (ROOT.parents[1] / 'SHA256SUMS').read_text().splitlines() if 'pytorch_model.bin' in l)
    before = file_hash(DEFAULT_MODEL / 'pytorch_model.bin')
    assert before == expected, 'Checkpoint differs from recovered original'
    model = GPT2LMHeadModel.from_pretrained(DEFAULT_MODEL, local_files_only=True, use_safetensors=False).to(device).eval()
    before_parameters = parameter_hash(model)
    tokenizer = OriginalYTEncoder(DEFAULT_MODEL / 'encoder.model')
    saved = json.loads((ROOT.parent / 'surreal-sane' / 'discovery.json').read_text())
    baseline = next(r for r in saved['results'] if r['name'] == 'baseline')
    found = next(g['text'] for g in baseline['generations'] if 'Капустин' in g['text'])
    natural = found.split('Капустин')[0].rstrip()
    contexts = {
        'observed': natural,
        'observed_without_gasparov': natural.replace('Гаспаров', 'Сидоров'),
        'short_clause': 'Еще он сказал, что от него хочет сказать',
        'phone': 'Он поднял трубку. На другом конце провода был',
        'door': 'В дверь постучали. На пороге стоял',
        'report': 'Начальник уголовного розыска сообщил, что завтра приедет',
        'genitive': 'В кабинете ждали ответа от',
        'dative': 'Секретарь передал письмо',
        'copy_positive_control': 'Капустин вошёл в комнату. Все знали, что это',
    }
    save('design.json', {'checkpoint_sha256': before, 'device': str(device),
         'runtime': {'python': sys.version, 'torch': torch.__version__, 'transformers': transformers.__version__, 'dtype': str(next(model.parameters()).dtype)},
         'targets': TARGETS,
         'controls': CONTROLS, 'contexts': contexts, 'discovery_context': 'observed',
         'method': 'Half-scale activations at all positions; select on observed context then validate elsewhere. Raw full-vocabulary probabilities, no sampling, temperature, top-p or style patch.',
         'interpretation': 'Scores are token-prefix likelihoods, not complete-word probabilities or entity probabilities. Teacher-forced later fragments can measure spelling completion.',
         'tokenization': {t: tokenizer.encode(t) for t in TARGETS}})

    def inputs(context):
        prefix = tokenizer.encode(context)
        rows, labels = [], []
        for target in TARGETS:
            full = tokenizer.encode(context + ' ' + target)
            assert full[:len(prefix)] == prefix, (context, target, 'token boundary changed')
            labels.append(full[len(prefix):])
            rows.append(full[:-1])
        maxlen = max(map(len, rows)); maxtarget = max(map(len, labels))
        tokens = torch.zeros((len(rows), maxlen), dtype=torch.long, device=device)
        attention = torch.zeros_like(tokens)
        positions = torch.zeros((len(rows), maxtarget), dtype=torch.long, device=device)
        targets = torch.zeros_like(positions)
        valid = torch.zeros_like(positions, dtype=torch.bool)
        for i, (row, label) in enumerate(zip(rows, labels)):
            tokens[i, :len(row)] = torch.tensor(row, device=device)
            attention[i, :len(row)] = 1
            positions[i, :len(label)] = torch.arange(len(prefix)-1, len(prefix)-1+len(label), device=device)
            targets[i, :len(label)] = torch.tensor(label, device=device)
            valid[i, :len(label)] = True
        return tokens, attention, positions, targets, valid

    batches = {key: inputs(value) for key,value in contexts.items()}
    def readout(hidden, positions, targets, valid, normalize=False):
        selected = hidden[torch.arange(len(TARGETS), device=device)[:,None], positions]
        if normalize:
            selected = model.transformer.ln_f(selected)
        logits = model.lm_head(selected)
        logprobs = logits.log_softmax(-1).gather(-1, targets[...,None]).squeeze(-1)
        total = logprobs.masked_fill(~valid, 0).sum(-1).cpu().tolist()
        token_scores = logprobs.cpu().tolist(); counts=valid.sum(-1).cpu().tolist()
        return {name: {'logp': total[i], 'probability': math.exp(total[i]),
                       'token_logp': token_scores[i][:counts[i]]} for i,name in enumerate(TARGETS)}

    def score(key, lens=False):
        tokens, attention, positions, targets, valid = batches[key]
        out = model.transformer(tokens, attention_mask=attention, use_cache=False, output_hidden_states=lens)
        result = readout(out.last_hidden_state, positions, targets, valid)
        if lens:
            traces=[]
            for layer in range(model.config.n_layer):
                traces.append({'layer':layer,'scores':readout(out.hidden_states[layer+1], positions, targets, valid,
                                                             normalize=layer < model.config.n_layer-1)})
            return result,traces
        return result

    def effects(result, base):
        delta = {t: result[t]['logp']-base[t]['logp'] for t in TARGETS}
        specificity = delta['Капустин'] - sum(delta[t] for t in CONTROLS)/len(CONTROLS)
        return {'delta_logp':delta,'specificity_delta':specificity,
                'kapustin_probability_ratio': math.exp(delta['Капустин'])}

    with torch.inference_mode():
        baselines={}; lenses={}
        for key in contexts:
            baselines[key],lenses[key]=score(key,True)
        save('baselines.json',baselines); save('logit-lens.json',lenses)
        scan=[]
        for kind in ('attention','mlp'):
            for layer in range(model.config.n_layer):
                with attenuate(model,kind,layer):
                    result=score('observed')
                scan.append({'kind':kind,'layer':layer,**effects(result,baselines['observed'])})
                save('layer-scan.json',scan)
            print('Finished',kind,'layer scan',flush=True)
        # Select only from the discovery context, before validation results.
        candidates=[]
        for kind in ('attention','mlp'):
            subset=[r for r in scan if r['kind']==kind]
            best=sorted(subset,key=lambda r:r['specificity_delta'])[:2]
            candidates.extend(best)
        save('selected-layers.json',candidates)
        detail=[]
        for candidate in candidates:
            layer=candidate['layer']; kind='head' if candidate['kind']=='attention' else 'mlp_group'
            count=model.config.n_head if kind=='head' else model.config.n_inner or 4*model.config.n_embd
            if kind=='mlp_group': count//=64
            for index in range(count):
                with attenuate(model,kind,layer,index):
                    result=score('observed')
                detail.append({'kind':kind,'layer':layer,'index':index,**effects(result,baselines['observed'])})
            save('component-scan.json',detail)
            print('Finished',kind,'layer',layer,flush=True)
        picked=[]
        for kind in ('head','mlp_group'):
            picked.extend(sorted([r for r in detail if r['kind']==kind],key=lambda r:r['specificity_delta'])[:3])
        save('selected-components.json',picked)
        validations=[]
        for candidate in candidates+picked:
            cases={}
            for key in contexts:
                with attenuate(model,candidate['kind'],candidate['layer'],candidate.get('index')):
                    result=score(key)
                cases[key]=effects(result,baselines[key])
            validations.append({'component':candidate,'contexts':cases})
            save('validation.json',validations)
            print('Validated',candidate['kind'],candidate['layer'],candidate.get('index'),flush=True)
        restored=score('observed')
        error=max(abs(restored[t]['logp']-baselines['observed'][t]['logp']) for t in TARGETS)
        assert error < 1e-5, error
    after_parameters=parameter_hash(model)
    after=file_hash(DEFAULT_MODEL / 'pytorch_model.bin')
    assert after==before and after_parameters==before_parameters
    save('integrity.json', {'checkpoint_sha256_before':before,'checkpoint_sha256_after':after,
         'parameter_sha256_before':before_parameters,'parameter_sha256_after':after_parameters,
         'restored_max_logp_error':error,'style_patch_loaded':False,'passed':True})
    print('DONE: original file and all parameter tensors unchanged',flush=True)


if __name__=='__main__':
    main()
