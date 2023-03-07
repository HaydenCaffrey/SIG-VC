import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time

from classifier import ReversalClassifier
from bnf2mel import bnf2mel
from fastspeech2 import FastSpeech2
import sv_model.modules.model_spk as models_spk
from loss import FastSpeech2Loss
from dataset import Dataset
from optimizer import ScheduledOptim
#from evaluate import evaluate
import hparams as hp
import utils
import audio as Audio


def main(args):
    torch.manual_seed(0)

    # Get device
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')

    # Get dataset
    dataset = Dataset()
    loader = DataLoader(dataset, batch_size=hp.batch_size**2, shuffle=True,
                        collate_fn=dataset.collate_fn, drop_last=True, num_workers=0)

    # Define model
    model = nn.DataParallel(FastSpeech2(input_size = hp.bnf_size)).to(device)
#     bnf2mel_model = bnf2mel().to(device)

    # #load sv model 
#     model_sv = getattr(models_spk, hp.sv_model_type)(hp.in_planes, hp.embd_dim, dropout=0).cuda() #resnet
    model_sv = getattr(models_spk, hp.sv_model_type)(80, hidden_dim=1024, embedding_size=256).cuda() #tdnn

    
    model_sv  = nn.DataParallel(model_sv).to(device) 
    cos_loss_fun  = nn.CosineSimilarity(dim=1).to(device)
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model) #+ utils.get_param_num(model_b)
    print('Number of FastSpeech2 Parameters:', num_param)

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr = hp.lr, betas=hp.betas, eps=hp.eps, weight_decay=hp.weight_decay)
    scheduled_optim = ScheduledOptim(
        optimizer, hp.decoder_hidden, hp.n_warm_up_step, args.restore_step)
    
    Loss = FastSpeech2Loss().to(device)
    print("Optimizer and Loss Function Defined.")

#     bnf2mel_pre = torch.load(os.path.join('./bnf-mel_model/', 'bnf-mel.pth.tar'))#,map_location='cpu')
        ########################多gpu训练时需要
    # create new OrderedDict that does not contain `module.`
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in bnf2mel_pre['model'].items():
#         name = k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
#         new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。 
#     # load params
#     bnf2mel_model.load_state_dict(new_state_dict) # 从新加载这个模型。
    ##############################################################
#     bnf2mel_model.load_state_dict(bnf2mel_pre['model'])
    
#     checkpoint_sv = torch.load('./sv_model/save_model/%s/model_39.pkl' % hp.sv_model_name ) #resnet
    checkpoint_sv = torch.load('sv_model/save_model/%s/model_94.pkl' % hp.sv_model_name ) #tdnn

    
    ########################多gpu训练时需要
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint_sv['model'].items():
        name = 'module.'+k # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。 
    # load params
    model_sv.load_state_dict(new_state_dict) # 从新加载这个模型。
    ##############################################################
    
#     model_sv.load_state_dict(checkpoint_sv['model']) #load state dict sv model
    print("\n---Speker Encoder Model Restored ---\n") 
    # Load checkpoint if exists
    checkpoint_path = os.path.join(hp.checkpoint_path)
    if args.restore_step:
#         checkpoint_sv = torch.load('sv_model/save_model/%s/model_39.pkl' % hp.sv_model_name )
#         model_sv.load_state_dict(checkpoint_sv['model']) #load state dict sv model
#         model_sv  = nn.DataParallel(model_sv)
#         print("\n---Speker Encoder Model Restored ---\n") 
        #print(model_sv.state_dict()['module.front.conv1.weight'][0])
        checkpoint = torch.load(os.path.join(
            checkpoint_path, 'checkpoint_{}.pth.tar'.format(args.restore_step)))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])        
        print("\n---Model Restored at Step {}---\n".format(args.restore_step))
    else:
        print("\n---Start New Training---\n")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    # Load vocoder
    '''
    if hp.vocoder == 'melgan':
        melgan = utils.get_melgan()
    elif hp.vocoder == 'waveglow':
        waveglow = utils.get_waveglow()
    '''

    # Init logger
    log_path = hp.log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(os.path.join(log_path, 'train'))
        os.makedirs(os.path.join(log_path, 'validation'))
    train_logger = SummaryWriter(os.path.join(log_path, 'train'))
    val_logger = SummaryWriter(os.path.join(log_path, 'validation'))

    # Init synthesis directory
    synth_path = hp.synth_path
    if not os.path.exists(synth_path):
        os.makedirs(synth_path)

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()

    # Training
    model = model.train()
    model_sv.eval() # sv model state
    for epoch in range(hp.epochs):
        # Get Training Loader
        total_step = hp.epochs * len(loader) * hp.batch_size

        for i, batchs in enumerate(loader):
            for j, data_of_batch in enumerate(batchs):
                start_time = time.perf_counter()

                current_step = i*hp.batch_size + j + args.restore_step + \
                    epoch*len(loader)*hp.batch_size + 1

                # Get Data
                bnf = torch.from_numpy(
                    data_of_batch["bnf"]).float().to(device)
                mel_target = torch.from_numpy(
                    data_of_batch["mel_target"]).float().to(device)
                
                speaker = torch.from_numpy(data_of_batch["speaker"]).long().to(device)


                src_len = torch.from_numpy(
                    data_of_batch["src_len"]).long().to(device)
                mel_len = torch.from_numpy(
                    data_of_batch["mel_len"]).long().to(device)
                max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
                max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)
                
                tar_spk_embd = torch.from_numpy(data_of_batch["embed"]).to(device)
#                 print(bnf.size(),'%%%%%%%%%%%%%%%%%')
                bnf_mid, mel_mid, mel_output, mel_postnet_output, src_mask, mel_mask = model(bnf, src_len, mel_len, max_src_len, max_mel_len, tar_spk_embd, tar_spk_embd)
                
                pre_embd = model_sv(mel_postnet_output)
                
                mid_embd = model_sv(mel_mid)
                # Cal Loss
                # 最后的mel要和target尽量一样
                mel_loss, mel_postnet_loss, std_loss = Loss(mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask, max_src_len)
                
                zero_embd = tar_spk_embd-tar_spk_embd
                
                if hp.loss_type =='cos':
                    spk_loss_postnet = 1-torch.cosine_similarity(pre_embd, tar_spk_embd, dim=-1)
                    spk_loss_mid = 2*l1_loss(mid_embd, zero_embd)
                else:
                    spk_loss_postnet = mse_loss(pre_embd, tar_spk_embd)
                    spk_loss_mid = 2*l1_loss(mid_embd, zero_embd)

                spk_loss = hp.spkembd_weight * spk_loss_mid.mean()
                total_loss = spk_loss + mel_loss
#                 spk_loss = hp.spkembd_weight * spk_loss_mid.mean() + hp.spkembd_weight * spk_loss_postnet.mean()
#                 total_loss = std_loss + mel_postnet_loss + spk_loss + mel_loss
                
                # Logger
                t_l = total_loss.item()
                m_l = mel_loss.item()
                m_p_l = mel_postnet_loss.item()
                std_l = std_loss.item()
                s_l = spk_loss.mean().item() 

                with open(os.path.join(log_path, "total_loss.txt"), "a") as f_total_loss:
                    f_total_loss.write(str(t_l)+"\n")
                with open(os.path.join(log_path, "mel_loss.txt"), "a") as f_mel_loss:
                    f_mel_loss.write(str(m_l)+"\n")
                with open(os.path.join(log_path, "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
                    f_mel_postnet_loss.write(str(m_p_l)+"\n")
                with open(os.path.join(log_path, "std_loss.txt"), "a") as f_std_loss:
                    f_std_loss.write(str(std_l)+"\n")
                with open(os.path.join(log_path, "spk_loss.txt"), "a") as f_std_loss:
                    f_std_loss.write(str(s_l)+"\n")
                
                # Backward
                total_loss = total_loss / hp.acc_steps
                total_loss.backward()
                if current_step % hp.acc_steps != 0:
                    continue

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), hp.grad_clip_thresh)

                # Update weights
                scheduled_optim.step_and_update_lr()
                scheduled_optim.zero_grad()

                # Print
                if current_step % hp.log_step == 0:
                    Now = time.perf_counter()

                    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                        epoch+1, hp.epochs, current_step, total_step)
                    str2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Std Loss: {:.4f}, Spk embd loss:{:.4f}, Spk embd postnet loss:{:.4f}, Spk embd mid loss:{:.4f}".format(t_l, m_l, m_p_l, std_l, s_l, spk_loss_postnet.mean().item(), spk_loss_mid.mean().item())
                    
#                     str2 = "Total Loss: {:.4f}, bnf_deper_loss: {:.4f}, mel_deper_loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Std Loss: {:.4f}, Spk_embd_loss:{:.4f}, spk_loss1:{:.4f}, spk_loss_comp1:{:.4f}, spk_loss2:{:.4f}, spk_loss_comp2:{:.4f}, spk_loss_mid1:{:.4f},spk_loss_mid2:{:.4f}, Mel Loss1: {:.4f}, Mel Loss comp1: {:.4f}, Mel Loss2: {:.4f}, Mel Loss comp2: {:.4f}, Mel PostNet Loss1: {:.4f}, Mel PostNet Loss comp1: {:.4f}, Mel PostNet Loss2: {:.4f}, Mel PostNet Loss comp2: {:.4f}, class loss: {:.4f}".format(t_l, b_d_l, m_d_l, m_l, m_p_l, std_l, s_l, spk_loss1.mean().item(), spk_loss_comp1.mean().item(), spk_loss2.mean().item(), spk_loss_comp2.mean().item(), spk_loss_mid1.mean().item(), spk_loss_mid2.mean().item(), mel_loss1.item(),mel_loss_comp1.item(),mel_loss2.item(),mel_loss_comp2.item(),mel_postnet_loss1.item(),mel_postnet_loss_comp1.item(),mel_postnet_loss2.item(),mel_postnet_loss_comp2.item())
    
    
                    str3 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                        (Now-Start), (total_step-current_step)*np.mean(Time))

                    print("\n" + str1)
                    print(str2)
                    print(str3)

                    with open(os.path.join(log_path, "log.txt"), "a") as f_log:
                        f_log.write(str1 + "\n")
                        f_log.write(str2 + "\n")
                        f_log.write(str3 + "\n")
                        f_log.write("\n")

                    train_logger.add_scalar(
                        'Loss/total_loss', t_l, current_step)
                    train_logger.add_scalar('Loss/mel_loss', m_l, current_step)
                    train_logger.add_scalar(
                        'Loss/mel_postnet_loss', m_p_l, current_step)
                    train_logger.add_scalar('Loss/std_loss', std_l, current_step)
                    train_logger.add_scalar('Loss/spk_loss', s_l, current_step)
                
                
                
                if current_step % hp.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(checkpoint_path, 'checkpoint_{}{}.pth.tar'.format(current_step, hp.sav_post)))
                    print("save model at step {} ...".format(current_step))

#                 if current_step % hp.synth_step == 0:
#                     length = mel_len[0].item()
#                     #mel_target_torch = mel_target[0, :length].detach(
#                     #).unsqueeze(0).transpose(1, 2)
#                     mel_target = mel_target[0, :length].detach(
#                     ).cpu().transpose(0, 1)
#                     #mel_torch = mel_output[0, :length].detach(
#                     #).unsqueeze(0).transpose(1, 2)
#                     mel = mel_output[0, :length].detach().cpu().transpose(0, 1)
#                     #mel_postnet_torch = mel_postnet_output[0, :length].detach(
#                     #).unsqueeze(0).transpose(1, 2)
#                     mel_postnet = mel_postnet_output[0, :length].detach(
#                     ).cpu().transpose(0, 1)
#                     '''
#                     Audio.tools.inv_mel_spec(mel, os.path.join(
#                         synth_path, "step_{}_griffin_lim.wav".format(current_step)))
#                     Audio.tools.inv_mel_spec(mel_postnet, os.path.join(
#                         synth_path, "step_{}_postnet_griffin_lim.wav".format(current_step)))
                
#                     if hp.vocoder == 'melgan':
#                         utils.melgan_infer(mel_torch, melgan, os.path.join(
#                             hp.synth_path, 'step_{}_{}.wav'.format(current_step, hp.vocoder)))
#                         utils.melgan_infer(mel_postnet_torch, melgan, os.path.join(
#                             hp.synth_path, 'step_{}_postnet_{}.wav'.format(current_step, hp.vocoder)))
#                         utils.melgan_infer(mel_target_torch, melgan, os.path.join(
#                             hp.synth_path, 'step_{}_ground-truth_{}.wav'.format(current_step, hp.vocoder)))
#                     elif hp.vocoder == 'waveglow':
#                         utils.waveglow_infer(mel_torch, waveglow, os.path.join(
#                             hp.synth_path, 'step_{}_{}.wav'.format(current_step, hp.vocoder)))
#                         utils.waveglow_infer(mel_postnet_torch, waveglow, os.path.join(
#                             hp.synth_path, 'step_{}_postnet_{}.wav'.format(current_step, hp.vocoder)))
#                         utils.waveglow_infer(mel_target_torch, waveglow, os.path.join(
#                             hp.synth_path, 'step_{}_ground-truth_{}.wav'.format(current_step, hp.vocoder)))
                    
#                     f0 = f0[0, :length].detach().cpu().numpy()
#                     energy = energy[0, :length].detach().cpu().numpy()
#                     f0_output = f0_output[0, :length].detach().cpu().numpy()
#                     energy_output = energy_output[0,
#                                                :length].detach().cpu().numpy()
#                     '''

#                     utils.plot_data([(mel_postnet.numpy()), (mel_target.numpy())],
#                                     ['Synthetized Spectrogram', 'Ground-Truth Spectrogram'], filename=os.path.join(synth_path, 'step_{}.png'.format(current_step)))
                '''
                if current_step % hp.eval_step == 0:
                    model.eval()
                    with torch.no_grad():
                        d_l, f_l, e_l, m_l, m_p_l = evaluate(
                            model, current_step)
                        t_l = d_l + f_l + e_l + m_l + m_p_l

                        val_logger.add_scalar(
                            'Loss/total_loss', t_l, current_step)
                        val_logger.add_scalar(
                            'Loss/mel_loss', m_l, current_step)
                        val_logger.add_scalar(
                            'Loss/mel_postnet_loss', m_p_l, current_step)
                        val_logger.add_scalar(
                            'Loss/duration_loss', d_l, current_step)
                        val_logger.add_scalar(
                            'Loss/F0_loss', f_l, current_step)
                        val_logger.add_scalar(
                            'Loss/energy_loss', e_l, current_step)
                
                    model.train()
                '''
                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    args = parser.parse_args()

    main(args)
