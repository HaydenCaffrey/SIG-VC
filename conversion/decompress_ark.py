# decomporess ark file to a batch of numpy arrays
# 
# usage: 
# python decomporess_ark.py <../bn_extractor/data/synthesis/bnf> <bnf_train>
import argparse
from kaldiio import ReadHelper
import numpy as np 
from tqdm import tqdm
import os
import os.path as osp
import glob


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('bnf_path')
    parser.add_argument('bnf_dump')
    
    args = parser.parse_args()
    [ark_path] = glob.glob(osp.join(args.bnf_path, 'data', 'raw_bnfeat*.ark'))

    os.makedirs(args.bnf_dump, exist_ok=True)

    with ReadHelper(f'ark:{ark_path}') as reader : 
        for k,v in tqdm(reader) : 
            np.save(osp.join(args.bnf_dump, f'{k}.npy'), v)

if __name__ == '__main__': main()
