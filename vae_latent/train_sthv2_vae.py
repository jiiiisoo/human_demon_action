import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from laq_model.vae import ConvVAE
from laq_model.data_single import SingleImageFrameDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', type=str, required=True, help='root of frame folders')
    p.add_argument('--image_size', type=int, default=256)
    p.add_argument('--dim', type=int, default=1024)
    p.add_argument('--latent_dim', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num_steps', type=int, default=100000)
    p.add_argument('--save_every', type=int, default=5000)
    p.add_argument('--results', type=str, default='results_vae')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = SingleImageFrameDataset(args.folder, args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    it = iter(dl)

    vae = ConvVAE(in_ch=3, dim=args.dim, latent_dim=args.latent_dim).to(device)
    opt = AdamW(vae.parameters(), lr=args.lr)

    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(1, args.num_steps + 1), desc='train-vae')
    vae.train()
    for step in pbar:
        try:
            x = next(it)
        except StopIteration:
            it = iter(dl)
            x = next(it)

        x = x.to(device)
        opt.zero_grad(set_to_none=True)
        x_hat, mu, logvar = vae(x)
        loss, logs = ConvVAE.loss_fn(x_hat, x, mu, logvar)
        loss.backward()
        opt.step()
        pbar.set_postfix(logs)

        if step % args.save_every == 0:
            torch.save(vae.state_dict(), results / f'vae_{step}.pt')

    torch.save(vae.state_dict(), results / 'vae_final.pt')


if __name__ == '__main__':
    main()


