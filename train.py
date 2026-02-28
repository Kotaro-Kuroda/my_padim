
import os

import torch
import tqdm

from dataset.mvtec_dataset import MVTecDataset
from models import padim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(dataloader, model):
    ev_count = 0
    ev_mean = None
    ev_ssmd = None
    ev_sum = None

    for x in tqdm.tqdm(dataloader):
        x = x.to(device)
        features = model(x)
        B, C, H, W = features.size()
        embedding_vectors = features.view(B, C, H * W)
        val = torch.sum(embedding_vectors, dim=0)
        ev_sum = ev_sum + val if ev_sum is not None else val
        d1 = ev_mean.clone() if ev_mean is not None else None
        mu2 = val / B
        ev_mean = ev_sum / (ev_count + B)
        d1 = (d1 - ev_mean).view(1, C, H * W) if d1 is not None else None
        d2 = (mu2 - ev_mean).view(1, C, H * W)
        val_ssmd = torch.einsum("bcx,bdx->cdx", embedding_vectors - mu2, embedding_vectors - mu2)
        ev_ssmd = (ev_ssmd + ev_count * torch.einsum("bcx,bdx->cdx", d1, d1) + val_ssmd + B * torch.einsum("bcx,bdx->cdx", d2, d2)) if ev_ssmd is not None else val_ssmd
        ev_count += B
    eye = torch.eye(C).view(C, C, 1).repeat(1, 1, H * W).to(device)
    ev_cov = ev_ssmd / (ev_count - 1) + 0.01 * eye
    cov_inv = torch.inverse(ev_cov.permute(2, 0, 1)).permute(1, 2, 0)
    return ev_mean, cov_inv


def main():
    backbone_name = "dinov2_vits14_reg"
    category = "bottle"
    model = padim.PaDiM(backbone_name=backbone_name, embed_dim=100).to(device)
    model.idx = model.idx.to(device)
    model.eval()
    dataset = MVTecDataset(root_dir="/home/localuser/data/mvtec/", categories=[category], input_size=672, pad_size=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    mean, cov_inv = train(dataloader, model)
    save_dir = './weights'
    os.makedirs(save_dir, exist_ok=True)
    torch.save({"mean": mean, "cov_inv": cov_inv, "idx": model.idx}, f"{save_dir}/{backbone_name}_{category}.pt")


if __name__ == "__main__":
    main()
