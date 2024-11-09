import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
import torch.nn.functional as F

from lib.models.layers.attn import Attention

record_ratio  = []
class DynamicMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=768 // 2, output_dim=1):
        super(DynamicMLP, self).__init__()

        #
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.cuda()
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        ratio = 0.5 + 0.5 * torch.sigmoid(x)
        ratio = torch.mean(ratio)
        # record_ratio.append(ratio.cpu().numpy())
        # print(record_ratio, len(record_ratio))
        # print('ratio.....', ratio)
        return ratio

class MLP_score(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=768 // 2, output_dim=1):
        super(MLP_score, self).__init__()

        #
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.score = nn.Softmax(dim=1)
        self.k2 = nn.Sigmoid()

    def forward(self, x):
        x = x.cuda()
        x = torch.relu(self.fc1(x))
        k2 = torch.mean(0.05 * self.k2(x))
        x = self.fc2(x)
        score = self.score(x).squeeze(-1)
        return score, k2

def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float,
                          global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search re

        gion tokens
    """
    lens_s = attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape

    tokens_s = tokens[:, lens_t:]
    # if tokens_s.shape[1] == 256:
    test_pro = F.softmax(tokens_s, dim=1)
    test_entropy = -torch.sum(test_pro * torch.log2(test_pro + 1e-10), dim=1)  # compute entropy
    mlp1 = DynamicMLP(input_dim=768).cuda()
    keep_ratio = mlp1(test_entropy).cpu().float()

    # print('keep_ratio..............',keep_ratio)
    # keep_ratio = 0.613
    lens_keep = math.ceil(keep_ratio * lens_s)
    # if lens_keep == lens_s:
    #     print('into is, so error')
    #     print('lens_KEPP,', lens_keep, keep_ratio)
    #     return tokens, global_index, None
    # print(tokens_s.shape, test_entropy.shape)

    attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    mlp2 = MLP_score(input_dim=768).cuda()
    test_score, k2 = mlp2(attentive_tokens)
    # test_pro = F.softmax(attentive_tokens, dim=-1)
    # test_entropy = -torch.sum(test_pro * torch.log2(test_pro + 1e-10), dim=-1)  # compute entropy
    sort_index = torch.argsort(test_score, dim=-1)  # sort
    # topk_idx = topk_idx[sort_index]
    topk_idx = torch.gather(topk_idx, dim=1, index=sort_index)
    lens_keep_entropy = math.ceil(0.9 * topk_idx.size(1))
    start_point = math.ceil(k2 * topk_idx.size(1))
    #
    # topk_idx_entropy = topk_idx[:, :lens_keep_entropy]
    topk_idx_entropy = topk_idx[:, start_point:]
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx_entropy.unsqueeze(-1).expand(B, -1, C))
    #
    # # non_topk_idx = torch.cat([non_topk_idx,  topk_idx[:, 0:10], topk_idx[:, -10:]], dim=1)
    # non_topk_idx = torch.cat([non_topk_idx, topk_idx[:, lens_keep_entropy:]], dim=1)
    non_topk_idx = torch.cat([non_topk_idx, topk_idx[:, :start_point]], dim=1)
    keep_index = global_index.gather(dim=1, index=topk_idx_entropy)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index


class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search

    def forward(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None):
        x_attn, attn = self.attn(self.norm1(x), mask, True)
        x = x + self.drop_path(x_attn)
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, global_index_template, global_index_search, removed_index_search, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
