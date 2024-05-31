import os

import torch
import numpy as np


def construct_user_latent_state(u, V, item_idx, rating, lr=0.01, lbd=0.01):
    v_itemidx = V[:, item_idx][:, np.newaxis]
    return u - 2 * lr * (np.matmul(u.T, v_itemidx) - rating) * v_itemidx + lbd * u


def create_interaction_matrix_full(data, total_item_len):
    """creates a full (non-sparse) interaction matrix from the user/item pairs in the dataset"""
    user_ids = list(data['user_id'].unique())
    item_ids = list(data['item_id'].unique())
    user_posid_dict = {k: v for (k, v) in zip(user_ids, range(len(user_ids)))}
    arr = np.zeros(shape=(len(user_ids), total_item_len))
    for ix, row in data.iterrows():
        arr[user_posid_dict[row['user_id']], row['item_id']] = row['rating']
    return arr, (user_ids, item_ids)


def run_matrix_factorization(data, total_item_len, device, num_iter=4000, lbd=0.01, lr=0.001):
    """performs default matrix factorization"""
    arr, ids = create_interaction_matrix_full(data, total_item_len)
    device = device
    U = torch.rand(size=(len(ids[0]), 64), requires_grad=True, device=device).float()
    # factorize for all items even if certain items don't occur during training to prevent test time errors
    V = torch.rand(size=(64, total_item_len), requires_grad=True, device=device).float()
    opt = torch.optim.Adam(params=[U, V], lr=lr)
    lbd = torch.tensor(lbd, device=device).float()
    X = torch.tensor(arr, device=device).float()

    for s in range(num_iter):
        X_hat = torch.matmul(U, V)
        loss = torch.square(X - X_hat).sum() + lbd * (torch.norm(U, p='fro') + torch.norm(V, p='fro'))
        loss.backward()
        opt.step()
        opt.zero_grad()

    return U.cpu().detach().numpy(), V.cpu().detach().numpy(), ids
