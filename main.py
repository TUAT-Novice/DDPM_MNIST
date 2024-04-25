import os
import time
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from model.unet import UnetModel
from model.diffusion_model import GaussianDiffusion
from utils import preview_forward, generate_dif

if __name__ == "__main__":
    batch_size = 128
    timesteps = 500
    epochs = 10
    p_uncound = 0.2  # dropout
    is_preview = False
    # is_conditional = True
    is_gif = True
    model_path = f'./saved_models/DDPM.h5'
    image_path = f'./saved_images/'
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    save_dir = '/'.join((model_path.split('/')[:-1]))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = datasets.MNIST(root='./dataset/mnist/', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UnetModel(
        in_channels=1,
        model_channels=96,
        out_channels=1,
        channel_mult=(1, 2, 2),
        attention_resolutions=[],
        class_num=10
    )
    model.to(device)
    gaussian_diffusion = GaussianDiffusion(timesteps=500, beta_schedule='linear')

    # preview
    if is_preview:
        image = next(iter(train_loader))[0][0].squeeze()
        label = next(iter(train_loader))[1][0].squeeze()
        x_start = image
        preview_forward(gaussian_diffusion, x_start, device)
        plt.savefig(os.path.join(image_path, 'forward.png'))
        plt.show()

    if model_path and os.path.exists(model_path):
        model = torch.load(model_path)
    else:
        # train
        len_data = len(train_loader)
        time_end = time.time()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for epoch in range(epochs):
            for step, (images, labels) in enumerate(train_loader):
                time_start = time_end
                optimizer.zero_grad()
                batch_size = images.shape[0]
                images = images.to(device)
                labels = labels.to(device)

                # random generate mask
                z_uncound = torch.rand(batch_size)
                batch_mask = (z_uncound > p_uncound).int().to(device)

                # sample t uniformally for every example in the batch
                t = torch.randint(0, timesteps, (batch_size,), device=device).long()

                loss = gaussian_diffusion.train_losses(model, images, t, labels, batch_mask)

                if step % 100 == 0:
                    time_end = time.time()
                    print("Epoch{}/{}\t  Step{}/{}\t Loss {:.4f}\t Time {:.2f}".format(epoch + 1, epochs, step + 1, len_data,
                                                                                       loss.item(), time_end - time_start))

                loss.backward()
                optimizer.step()
        save_dir = '/'.join((model_path.split('/')[:-1]))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(model, model_path)


    # eval
    for w in [0.0, 0.5, 2.0]:
        # DDPM random
        ddpm_random = gaussian_diffusion.sample(model, 28, batch_size=64, channels=1, n_class=10, w=w, mode='random',
                                                     clip_denoised=False)
        fig = plt.figure(figsize=(12, 12), constrained_layout=True)
        gs = fig.add_gridspec(8, 8)
        imgs = ddpm_random[-1].reshape(8, 8, 28, 28)
        for n_row in range(8):
            for n_col in range(8):
                f_ax = fig.add_subplot(gs[n_row, n_col])
                f_ax.imshow((imgs[n_row, n_col] + 1.0) * 255 / 2, cmap="gray")
                f_ax.axis("off")
        plt.savefig(os.path.join(image_path, f'DDPM_w={w}_random.png'))
        plt.show()
        # DDPM 0-9
        ddpm_all = gaussian_diffusion.sample(model, 28, batch_size=40, channels=1, n_class=10, w=w, mode='all',
                                                     clip_denoised=False)
        fig = plt.figure(figsize=(12, 5), constrained_layout=True)
        gs = fig.add_gridspec(4, 10)
        imgs = ddpm_all[-1].reshape(4, 10, 28, 28)
        for n_row in range(4):
            for n_col in range(10):
                f_ax = fig.add_subplot(gs[n_row, n_col])
                f_ax.imshow((imgs[n_row, n_col] + 1.0) * 255 / 2, cmap="gray")
                f_ax.axis("off")
        plt.savefig(os.path.join(image_path, f'DDPM_w={w}_all.png'))
        plt.show()

        # DDIM random
        ddim_random = gaussian_diffusion.ddim_sample(model, 28, batch_size=64, channels=1, ddim_timesteps=50,
                                                               n_class=10,
                                                               w=w, mode='random', ddim_discr_method='quad', ddim_eta=0.0,
                                                               clip_denoised=False)

        fig = plt.figure(figsize=(12, 12), constrained_layout=True)
        gs = fig.add_gridspec(8, 8)
        imgs = ddim_random[-1].reshape(8, 8, 28, 28)
        for n_row in range(8):
            for n_col in range(8):
                f_ax = fig.add_subplot(gs[n_row, n_col])
                f_ax.imshow((imgs[n_row, n_col] + 1.0) * 255 / 2, cmap="gray")
                f_ax.axis("off")
        plt.savefig(os.path.join(image_path, f'DDIM_w={w}_random.png'))
        plt.show()
        # DDIM 0-9
        ddim_all = gaussian_diffusion.ddim_sample(model, 28, batch_size=40, channels=1, ddim_timesteps=50,
                                                               n_class=10,
                                                               w=w, mode='all', ddim_discr_method='quad', ddim_eta=0.0,
                                                               clip_denoised=False)
        fig = plt.figure(figsize=(12, 5), constrained_layout=True)
        gs = fig.add_gridspec(4, 10)
        imgs = ddim_all[-1].reshape(4, 10, 28, 28)
        for n_row in range(4):
            for n_col in range(10):
                f_ax = fig.add_subplot(gs[n_row, n_col])
                f_ax.imshow((imgs[n_row, n_col] + 1.0) * 255 / 2, cmap="gray")
                f_ax.axis("off")
        plt.savefig(os.path.join(image_path, f'DDIM_w={w}_all.png'))
        plt.show()

        # gif image
        if is_gif:
            # too slow for the whole ddpm
            generate_dif(ddpm_random[::10], os.path.join(image_path, f'DDPM_w={w}_all.gif'), 10, 8, 8, 28)
            generate_dif(ddpm_all[::10], os.path.join(image_path, f'DDPM_w={w}_all.gif'), 10, 4, 10, 28)
            generate_dif(ddim_random, os.path.join(image_path, f'DDIM_w={w}_all.gif'), 10, 8, 8, 28)
            generate_dif(ddim_all, os.path.join(image_path, f'DDIM_w={w}_all.gif'), 10, 4, 10, 28)
