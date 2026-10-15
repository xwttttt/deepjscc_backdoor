"""Lazy dependency helpers and phase separation for independent c16 surrogates."""
P0_SHA256='24070ef082a37ef5c4a232857daed760442b2fd3850ef617537e39fae5ff8542'
PROXY_SPECS=[{'name':'same_arch_seed11','seed':11,'c':16,'depth':5,'width':32},
             {'name':'deeper_seed29','seed':29,'c':16,'depth':7,'width':40}]


def enforce_phase(stage,target_checkpoint):
    if stage not in ['fit','attack','evaluate']:
        raise ValueError('unknown stage')
    if stage!='evaluate' and target_checkpoint is not None:
        raise ValueError('target checkpoint is forbidden during surrogate fitting/attack generation')
    if stage=='evaluate' and target_checkpoint is None:
        raise ValueError('final evaluation requires the target checkpoint')


def validate_proxy_record(record):
    if record.get('role')!='surrogate' or record.get('initialization')!='random' or record.get('spec',{}).get('c')!=16:
        raise ValueError('requires a random-initialized c16 surrogate')
    if record.get('checkpoint_sha256')==P0_SHA256:
        raise ValueError('P0 cannot be supplied as a surrogate')


def state_fingerprint(model):
    import hashlib
    result=hashlib.sha256()
    for value in model.state_dict().values():
        result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def make_proxy(spec,device):
    import torch
    from rf_backdoor.magmaw_full import ImageCodec
    torch.manual_seed(spec['seed'])
    return ImageCodec(spec['c'],spec['depth'],spec['width']).to(device)


def load_proxies(directory,records,device):
    import torch
    models=[]
    for record in records:
        validate_proxy_record(record)
        model=make_proxy(record['spec'],device)
        payload=torch.load(directory/(record['name']+'.pt'),map_location=device,weights_only=True)
        model.load_state_dict(payload['model'],strict=True)
        models.append(model.eval().requires_grad_(False))
    return models


def received_attack(link,pattern,state,budget):
    import torch
    from rf_backdoor.magmaw_paper import paper_attack
    from rf_backdoor.magmaw_effective import cap_received
    delta,powers=paper_attack(link,pattern,state,budget,max_symbols=1408)
    powers['pre_cap_received_psr_db']=powers['received_psr_db']
    delta=cap_received(delta,state,budget)
    powers['received_psr_db']=10*torch.log10(delta.abs().square().mean(-1)/state['power'][:,0])
    if not bool(torch.isfinite(delta).all()) or bool((powers['received_psr_db']>budget+1e-4).any()):
        raise AssertionError('nonfinite or over-budget interference')
    return delta,powers
