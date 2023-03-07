import os

sav_post = ''
embed_scale = 3
lr = 0.001
back_weight = 3

# Dataset
base_dir = "./"
#"/Netdata/yangyg/dataset/cmu_data/vctk/"
embed_type = 'embeds'#"resnet_embeds"
num_speaker = 97

### multi-speaker
spk_embedding = True
gvector_dim = 256
dataset = "aishell_twostage"#"vctk_projection"

loss_type ='cos'

# Text
text_cleaners = ['english_cleaners']


# Speeaker encoder

sv_model_name = 'ECAPA-TDNN' #ECAPA-TDNN ResNet34SE
sv_model_type = 'ECAPA_TDNN'#ResNet34SEStatsPool'
in_planes = 80 # ResNet34SE in_planes=64,ECAPA-TDNN = 80
embd_dim = 256
spkembd_weight = 3

# Audio and mel
### for LJSpeech ###
'''
sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024
'''
### for Blizzard2020 ###
sampling_rate = 16000
filter_length = 800
hop_length = 200
win_length = 800

max_wav_value = 32768.0
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = 8000.0


bnf_size = 256
prenet_size = 256

# FastSpeech 2
encoder_layer = 4
encoder_head = 2
encoder_hidden = 256
decoder_layer = 4
decoder_head = 2
decoder_hidden = 256
fft_conv1d_filter_size = 1024
fft_conv1d_kernel_size = (9, 1)
encoder_dropout = 0.2
decoder_dropout = 0.2
embd_dropout = 0.2

variance_predictor_filter_size = 256
variance_predictor_kernel_size = 3
variance_predictor_dropout = 0.5

max_seq_len = 10000


# Quantization for F0 and energy
### for LJSpeech ###
f0_min = 71.0
f0_max = 795.8
energy_min = 0.0
energy_max = 315.0
### for Blizzard2013 ###
#f0_min = 71.0
#f0_max = 786.7
#energy_min = 21.23
#energy_max = 101.02

n_bins = 256


# Checkpoints and synthesis path
preprocessed_path = os.path.join(dataset,"preprocessed")
checkpoint_path = os.path.join(dataset,"ckpt")
synth_path = os.path.join(dataset,"synth")
eval_path = os.path.join(dataset,"eval")
log_path = os.path.join(dataset,"log")
test_path = "./results"


# Optimizer
batch_size = 10
epochs = 1000
n_warm_up_step = 4000
grad_clip_thresh = 1.0
acc_steps = 1

betas = (0.9, 0.98)
eps = 1e-9
weight_decay = 0.


# Vocoder
vocoder = 'melgan'  # 'waveglow' or 'melgan'


# Log-scaled duration
log_offset = 1.


# Save, log and synthesis
save_step = 3000
synth_step = 2000
eval_step = 1000
eval_size = 256
log_step = 10
clear_Time = 20




