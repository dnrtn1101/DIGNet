import importlib
import json
import numpy as np

import torch

from option import get_option
from solve import Solve
from sft_solve import sft_Solve


def main():
    opt = get_option()
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    if not opt.test_only:
        print(json.dumps(vars(opt), indent=4))
    if "sft" in opt.model.lower():
        module = importlib.import_module("model.add_sft.{}".format(opt.model.lower()))
        solve = sft_Solve(module, opt)
        solve.fit()
    else:
        module = importlib.import_module("model.org.{}".format(opt.model.lower()))
        solve = Solve(module, opt)
        solve.fit()


if __name__ == "__main__":
    main()
