import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)

    # model

    parser.add_argument('--pretrain', type=str,
                        help='path of the pretrained model')

    # dataset
    parser.add_argument('--data_root', type=str, default='')

    # training setups
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default:1e-4)')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'l1'])
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--patchsize', type=int, default=48)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--num_valimages', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=200000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--workers_idx', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='Weight decay value in Adam optimizer')
    parser.add_argument('--decay', type=str,
                        default='1200-2400', help='Scheduler lr decay(default: at after 200th, 500th epochs [200-500])')

    # misc
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--ckpt_root', type=str, default='./pt')
    parser.add_argument('--save_root', type=str, default='./output')
    parser.add_argument('--save_result', action='store_true',
                        help='Save restored im in evaluation phase')
    return parser.parse_args()


def get_option():
    opt = parse_args()
    return opt
