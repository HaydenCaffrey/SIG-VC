CUDA_VISIBLE_DEVICES=7 python train_on_tacotron.py \
--load_path=../logs/finetune_cross_lingual \
--save_path=../logs/finetune_cross_lingual \
--data_path=../dataset/finetune_cross_lingual
