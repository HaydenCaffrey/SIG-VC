from multiprocessing.pool import Pool 
from synthesizer import audio
from functools import partial
from itertools import chain
#from encoder import inference as encoder
from pathlib import Path
#from utils import logmmse
from tqdm import tqdm
import numpy as np
import librosa
import os
import argparse
import glob 

def preprocess_aishell(datasets_root, out_dir, n_processes, 
                           skip_existing, hparams, pairs, gen_manifest):
    # Gather the input directories
    
    print("\n  Using data from:  " + datasets_root)
    try:
    # Create the output directories for each output file type
        out_dir.joinpath("mels").mkdir(exist_ok=True)
        out_dir.joinpath("audios").mkdir(exist_ok=True)

        if gen_manifest: 
            # Create a metadata file
            metadata_fpath = out_dir.joinpath("train.txt")
            metadata_file = metadata_fpath.open("a" if skip_existing else "w", encoding="utf-8")

        #print(speaker_dirs)
        func = partial(preprocess_speaker, out_dir=out_dir, skip_existing=skip_existing, 
                       hparams=hparams, datasets_root=datasets_root)
        job = Pool(n_processes).imap(func, pairs)
        for speaker_metadata in tqdm(job, "AI-SHELL", len(pairs), unit="speakers"):
            for metadatum in speaker_metadata:
                if gen_manifest: 
                    metadata_file.write("|".join(str(x) for x in metadatum) + "\n")

        if gen_manifest: metadata_file.close()

        if gen_manifest: 
            # Verify the contents of the metadata file
            with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
                metadata = [line.split("|") for line in metadata_file]
            mel_frames = sum([int(m[4]) for m in metadata])
            timesteps = sum([int(m[3]) for m in metadata])
            sample_rate = hparams.sample_rate
            hours = (timesteps / sample_rate) / 3600
            print("The dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
              (len(metadata), mel_frames, timesteps, hours))
            print("Max input length (text chars): %d" % max(len(m[5]) for m in metadata))
            print("Max mel frames length: %d" % max(int(m[4]) for m in metadata))
            print("Max audio timesteps length: %d" % max(int(m[3]) for m in metadata))
    except Exception as e:
        print('an exception was raised during feature extraction...')
        print(e)
        print('&&&&&&&&&&&&')

def preprocess_speaker(speaker_dir, out_dir: Path, skip_existing: bool, hparams, datasets_root):
    metadata = []
    #print(speaker_dir.glob("*.wav"))
    #wav_paths = list(chain.from_iterable(speaker_dir.glob("*.wav")))
    wav_texts = list((datasets_root +x[0], x[1]) for x in speaker_dir)
    
    wavs = []
    texts = []
    wav_paths = []
    for wav_path, text in wav_texts:
    
        wav = split_on_silences(wav_path, hparams)

        wavs.append(wav)
        texts.append(text)
        wav_paths.append(wav_path)
    assert len(wav_paths) == len(wavs) == len(texts)
    for i, (wav, text) in enumerate(zip(wavs, texts)):
            metadata.append(process_utterance(wav, text, out_dir, wav_paths[i].split('/')[-1], 
                                            skip_existing, hparams))
    return [m for m in metadata if m is not None]
def split_on_silences(wav_fpath, hparams):
    # Load the audio waveform
    wav, _ = librosa.load(wav_fpath, hparams.sample_rate)
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
    
    return wav
def process_utterance(wav, text, out_dir, basename, 
                      skip_existing, hparams):
    ## FOR REFERENCE:
    # For you not to lose your head if you ever wish to change things here or implement your own
    # synthesizer.
    # - Both the audios and the mel spectrograms are saved as numpy arrays
    # - There is no processing done to the audios that will be saved to disk beyond volume  
    #   normalization (in split_on_silences)
    # - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
    #   is why we re-apply it on the audio on the side of the vocoder.
    # - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
    #   without extra padding. This means that you won't have an exact relation between the length
    #   of the wav and of the mel spectrogram. See the vocoder data loader.
    
    
    # Skip existing utterances if needed
    mel_fpath = out_dir.joinpath("mels", "mel-%s.npy" % basename.replace('.wav',''))
    wav_fpath = out_dir.joinpath("audios", "audio-%s.npy" % basename.replace('.wav',''))
    if skip_existing and mel_fpath.exists() and wav_fpath.exists():
        return None
    
    # Skip utterances that are too short
#     if len(wav) < hparams.utterance_min_duration * hparams.sample_rate:
#         return None
    
    # Compute the mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]
    
    # Skip utterances that are too long
#     if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
#         print(basename)
#         return None
    
    # Write the spectrogram, embed and audio to disk
    np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)
    
    # Return a tuple describing this training example
    return wav_fpath.name, mel_fpath.name, "embed-%s.npy" % basename.replace('.wav',''), len(wav), mel_frames, text


batch_size=5
pairs = []

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
args = parser.parse_args()
#with open('/NASdata/AudioData/mandarin/电子喉录音/EL5h_dn_src.txt','r') as f:
# with open('/Netdata/zhanghz/oneshot/fastspeech/aishell/taiwan_demo/train5.txt','r') as f:
#     lines=f.readlines()#[0:300]
if args.mode == 'train':
    lines = [os.path.basename(p) for p in glob.glob('training_data/*.wav')]
    subpairs = []
    begin = True
    for line in lines:
        items = line.strip()#.split('|')
        #print(items)

        subpairs.append([items[0:], items[0:]])
        if len(subpairs) % batch_size == 0 and not begin:
            pairs.append(subpairs)
            subpairs = []

        begin = False
    if len(subpairs) > 0:
        pairs.append(subpairs)
else:
    lines = [os.path.basename(p) for p in glob.glob('test_data/*.wav')]
    subpairs = []
    begin = True
    for line in lines:
        items = line.strip()#.split('|')
        #print(items)

        subpairs.append([items[0:], items[0:]])
        if len(subpairs) % batch_size == 0 and not begin:
            pairs.append(subpairs)
            subpairs = []

        begin = False
    if len(subpairs) > 0:
        pairs.append(subpairs)    


from synthesizer.hparams import hparams
from pathlib import Path
if args.mode=='train':
    rootpath = "./training_data/"
    gen_manifest = True
else:
    rootpath = "./test_data/"
    gen_manifest = False
outpath = Path("./")
outpath.mkdir(exist_ok=True, parents=True)

preprocess_aishell(rootpath, outpath, 20, False, hparams, pairs, gen_manifest=gen_manifest)
