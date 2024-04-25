import os
import imageio
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from glob import glob


def preview_forward(diffusion_model, x_start, device="cuda"):
    plt.figure(figsize=(16, 5))
    for idx, t in enumerate([0, 100, 300, 400, 499]):
        x_noisy = diffusion_model.q_sample(x_start.to(device), t=torch.tensor([t]).to(device))
        noisy_image = (x_noisy.squeeze() + 1) * 127.5
        if idx==0:
            noisy_image = (x_start.squeeze() + 1) * 127.5
        noisy_image = noisy_image.cpu().numpy().astype(np.uint8)
        plt.subplot(1, 5, 1 + idx)
        plt.imshow(noisy_image, cmap='gray')
        plt.axis("off")
        plt.title(f"t={t}")


def get_imgs(x_seq, imgs_path, h, w, img_size):
    if not os.path.exists(imgs_path):
        os.mkdir(imgs_path)
    for i in tqdm(range(len(x_seq)), desc='generate gif time step', total=len(x_seq)):
        fig = plt.figure(figsize=(12, 5), constrained_layout=True)
        gs = fig.add_gridspec(h, w)
        imgs = x_seq[i].reshape(h, w, img_size, img_size)
        for n_row in range(h):
            for n_col in range(w):
                f_ax = fig.add_subplot(gs[n_row, n_col])
                f_ax.imshow((imgs[n_row, n_col] + 1.0) * 255 / 2, cmap="gray")
                f_ax.axis("off")
        plt.savefig('{}/{:04d}.jpg'.format(imgs_path, i), dpi=360)
        plt.close()


def compose_gif(img_paths, output_path, fps=10):
    print(img_paths[:12])
    gif_images = []
    for path in img_paths:
        gif_images.append(imageio.imread(path))
    imageio.mimsave(output_path, gif_images, fps=fps)


def generate_dif(x_seq, output_path, fps, h, w, img_size, delete_imgs=True):
    get_imgs(x_seq, './gif_temp', h, w, img_size)
    img_path = './gif_temp/*.jpg'
    img_ls = sorted(glob(img_path))
    compose_gif(img_ls, output_path, fps)
    print('start delete images')
    if delete_imgs:
        for i in img_ls:
            os.remove(i)