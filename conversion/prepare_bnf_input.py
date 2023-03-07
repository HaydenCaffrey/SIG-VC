# automaticly collect filelist for bnf extraction
# 
# usage: 
# python prepare_bnf_input.py <training_data> <../bn_extractor/data/synthesis>
# note: switch above <...> to appropriate paths
import glob
import argparse
import os.path as osp
from typing import Sequence


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir')
    parser.add_argument('target_dir')

    args = parser.parse_args()
    
    wavepaths = glob.glob(osp.join(args.source_dir, '*.wav'))
    print(f"found {len(wavepaths)} wave files")

    wavscp = [(osp.basename(p), osp.abspath(p)) for p in wavepaths]
    utt2spk = [(uid, uid) for uid, _ in wavscp]

    def dump_table(filename: str, xss: Sequence[Sequence[str]]) -> None: 
        with open(filename, 'w') as f: 
            for xs in xss: 
                print('\t'.join(xs), file=f)

    print(f"dumping results to {args.target_dir}")
    dump_table(osp.join(args.target_dir, 'wav.scp'), wavscp)
    dump_table(osp.join(args.target_dir, 'utt2spk'), utt2spk)
    dump_table(osp.join(args.target_dir, 'text'), utt2spk)

if __name__ == '__main__': main()
