import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)

    # model
    parser.add_argument('--model', type=str, default='sft_base',
                        help='Select model')
    parser.add_argument('--pretrain', type=str,
                        help='path of the pretrained model')

    # dataset
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--mask_root', type=str, default='')

    # training setups
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default:1e-4)')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'l1'])
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--patchsize', type=int, default=48)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--num_valimages', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=300000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--workers_idx', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='Weight decay value in Adam optimizer')
    parser.add_argument('--decay', type=str,
                        default='1500-3000', help='Scheduler lr decay(default: at after 200th, 500th epochs [200-500])')

    # misc
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--save_result', action='store_true',
                        help='Save restored im in evaluation phase')
    parser.add_argument('--ckpt_root', type=str, default='./pt')
    parser.add_argument('--save_root', type=str, default='./output')

    return parser.parse_args()


def make_template(opt):
    opt.strict_dict = opt.test_only

    # model
    if "mepsnet" in opt.model.lower():
        # optimal settings for MEPSNet
        opt.kernelsize = 3
        opt.num_templates = 16
        opt.num_SRIRs = 3
        opt.num_SResidualBlocks = 12
        opt.extract_featsize = 256
        opt.num_experts = 3
        opt.expert_featsize = 64

    if "owan" in opt.model.lower():
        opt.loss = 'l1'
        opt.steps = 4
        opt.num_ch = 16
        opt.num_layer = 10

    if "edsr" in opt.model.lower():
        opt.num_blocks = 16
        opt.num_channels = 64
        opt.res_scale = 0.1

    if "base" in opt.model.lower():
        opt.kernelsize = 3
        opt.num_SRIRs = 2
        opt.num_SResidualBlocks = 10
        opt.extract_featsize = 256
        opt.expert_featsize = 64


def get_option():
    opt = parse_args()
    make_template(opt)
    return opt

