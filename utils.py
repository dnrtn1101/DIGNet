import numpy as np


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
        
    return 20 * np.log10(255.0 / np.sqrt(mse))