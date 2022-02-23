# SDITNet

### How to train recog_net
- enter ./recog_net/
 
- python main.py --data_root "dataset root" --gpu 0 --num_workers 1 --workers_idx 0 --ckpt_root ./pt/mask_root/ --patchsize 224 --loss mse --max_steps 200000 --num_valimages 50

### How to train recon_net

- python main.py --model sft_base --data_root "dataset root" --mask_root "mask root" --save_result --save_root ./output/folder/ --gpu 0 --num_workers 1 --patchsize 48 --workers_idx 0 --ckpt_root ./pt/folder/ --loss l1 --max_steps 300000 --decay 1500-3000 --num_valimages 50
