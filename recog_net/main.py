import json
import numpy as np
import torch
from option import get_option
from solve import Solve


def main():
    opt = get_option()
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if not opt.test_only:
        print(json.dumps(vars(opt), indent=4))

    solver = Solve(opt)
    if opt.test_only:
        psnr = solver.evaluate(solver.test_loader, 0)
        print("{:.2f}".format(psnr))
    else:
        solver.fit()


if __name__ == "__main__":
    main()

