import hashlib, numpy as np

def split_seed(root_seed:int, namespace:str, index:int=0)->int:
    payload=f"chem-prior-0.1.0:{int(root_seed)}:{namespace}:{int(index)}".encode()
    return int.from_bytes(hashlib.blake2b(payload,digest_size=8).digest(),"little") & 0x7FFF_FFFF_FFFF_FFFF

def rng(seed:int): return np.random.default_rng(int(seed) % (2**63-1))
