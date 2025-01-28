import torch
import os
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import config
from utils import save_checkpoint, load_checkpoint, seed_everything, get_device
from dataset import PotatoLeafDataset
from discriminator_model import Discriminator
from generator_model import Generator
from torchvision.utils import save_image
import torch.nn as nn

def train_fn(
    gen_bacterial, gen_healthy, disc_healthy, disc_bacterial, disc_early_blight, disc_late_blight, loader, opt_disc, opt_gen, l1_loss, mse_loss, d_scaler, g_scaler
):
    loop = tqdm(loader, leave=True)
    healthy_reals = 0
    healthy_fakes = 0

    for idx, (healthy, early_blight, late_blight) in enumerate(loop):
        healthy = healthy.to(config.DEVICE)
        early_blight = early_blight.to(config.DEVICE)
        late_blight = late_blight.to(config.DEVICE)

        # Train Discriminators (Healthy, Early Blight, Late Blight)
        with torch.cuda.amp.autocast():
            fake_bacterial = gen_bacterial(healthy)
            fake_healthy = gen_healthy(late_blight)
            # Discriminators for Healthy Domain
            D_healthy_real = disc_healthy(healthy)
            D_healthy_fake = disc_healthy(fake_bacterial.detach())
            healthy_reals += D_healthy_real.mean().item()
            healthy_fakes += D_healthy_fake.mean().item()
            D_healthy_real_loss = mse_loss(D_healthy_real, torch.ones_like(D_healthy_real))
            D_healthy_fake_loss = mse_loss(D_healthy_fake, torch.zeros_like(D_healthy_fake))
            D_healthy_loss = D_healthy_real_loss + D_healthy_fake_loss

            # Discriminators for Early Blight Domain
            D_early_blight_real = disc_early_blight(early_blight)
            D_early_blight_fake = disc_early_blight(fake_healthy.detach())
            D_early_blight_real_loss = mse_loss(D_early_blight_real, torch.ones_like(D_early_blight_real))
            D_early_blight_fake_loss = mse_loss(D_early_blight_fake, torch.zeros_like(D_early_blight_fake))
            D_early_blight_loss = D_early_blight_real_loss + D_early_blight_fake_loss

            # Discriminators for Late Blight Domain
            D_late_blight_real = disc_late_blight(late_blight)
            D_late_blight_fake = disc_late_blight(fake_healthy.detach())
            D_late_blight_real_loss = mse_loss(D_late_blight_real, torch.ones_like(D_late_blight_real))
            D_late_blight_fake_loss = mse_loss(D_late_blight_fake, torch.zeros_like(D_late_blight_fake))
            D_late_blight_loss = D_late_blight_real_loss + D_late_blight_fake_loss

            # Total Discriminator Loss
            D_loss = (D_healthy_loss + D_early_blight_loss + D_late_blight_loss) / 3

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators (Healthy and Bacterial)
        with torch.cuda.amp.autocast():
            D_healthy_fake = disc_healthy(fake_bacterial)
            D_early_blight_fake = disc_early_blight(fake_healthy)
            D_late_blight_fake = disc_late_blight(fake_healthy)
            loss_G_healthy = mse_loss(D_healthy_fake, torch.ones_like(D_healthy_fake))
            loss_G_bacterial = mse_loss(D_early_blight_fake, torch.ones_like(D_early_blight_fake))
            loss_G_late_blight = mse_loss(D_late_blight_fake, torch.ones_like(D_late_blight_fake))

            # Cycle Loss
            cycle_healthy = gen_healthy(fake_bacterial)
            cycle_bacterial = gen_bacterial(fake_healthy)
            cycle_healthy_loss = l1_loss(healthy, cycle_healthy)
            cycle_bacterial_loss = l1_loss(early_blight, cycle_bacterial)

            # Identity Loss
            identity_healthy = gen_healthy(healthy)
            identity_bacterial = gen_bacterial(early_blight)
            identity_healthy_loss = l1_loss(healthy, identity_healthy)
            identity_bacterial_loss = l1_loss(early_blight, identity_bacterial)

            # Total Generator Loss
            G_loss = (
                loss_G_healthy + loss_G_bacterial + loss_G_late_blight
                + cycle_healthy_loss * config.LAMBDA_CYCLE
                + cycle_bacterial_loss * config.LAMBDA_CYCLE
                + identity_healthy_loss * config.LAMBDA_IDENTITY
                + identity_bacterial_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Save images periodically
        if idx % 200 == 0:
            save_image(early_blight * 0.5 + 0.5, f"saved_images/real_early_blight_{idx}.png")
            save_image(fake_bacterial * 0.5 + 0.5, f"saved_images/healthy_to_bacterial_{idx}.png")
            save_image(healthy * 0.5 + 0.5, f"saved_images/real_healthy_{idx}.png")
            save_image(fake_healthy * 0.5 + 0.5, f"saved_images/early_blight_to_healthy_{idx}.png")

        loop.set_postfix(
            healthy_real=healthy_reals / (idx + 1),
            healthy_fake=healthy_fakes / (idx + 1),
        )


def main():
    seed_everything(config.SEED)

    # Load Dataset
    dataset = PotatoLeafDataset(config.TRAIN_DIR, domains=config.DOMAINS)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Initialize Models
    gen_bacterial = Generator(img_channels=3, num_features=64, num_residuals=9, num_domains=3).to(config.DEVICE)
    gen_healthy = Generator(img_channels=3, num_features=64, num_residuals=9, num_domains=3).to(config.DEVICE)
    disc_healthy = Discriminator(in_channels=3).to(config.DEVICE)
    disc_early_blight = Discriminator(in_channels=3).to(config.DEVICE)
    disc_late_blight = Discriminator(in_channels=3).to(config.DEVICE)

    # Optimizers
    opt_disc = optim.Adam(
        list(disc_healthy.parameters()) + list(disc_early_blight.parameters()) + list(disc_late_blight.parameters()),
        lr=config.LR_DISC,
        betas=(config.BETA1, config.BETA2),
    )
    opt_gen = optim.Adam(
        list(gen_bacterial.parameters()) + list(gen_healthy.parameters()),
        lr=config.LR_GEN,
        betas=(config.BETA1, config.BETA2),
    )

    # Loss functions
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    # Scalers for mixed precision training
    d_scaler = torch.cuda.amp.GradScaler()
    g_scaler = torch.cuda.amp.GradScaler()

    # Start Training
    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            gen_bacterial, gen_healthy, disc_healthy, disc_early_blight, disc_late_blight, loader, 
            opt_disc, opt_gen, l1_loss, mse_loss, d_scaler, g_scaler
        )
        save_checkpoint(gen_bacterial, opt_gen, filename=f"checkpoint_epoch_{epoch}.pth")


if __name__ == "__main__":
    main()
