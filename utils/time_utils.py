# copied from deformable 3d gaussians

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_embedder(multires, i=1):
    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, xyz_multires=10, t_multires=6, sh_degree=3):
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.xyz_multires = xyz_multires
        self.t_multires = t_multires
        self.skips = [D // 2]

        self.embed_xyz_fn, self.xyz_input_ch = get_embedder(self.xyz_multires, 3)
        self.embed_t_fn, self.t_input_ch = get_embedder(self.t_multires, 1)
        
        self.num_shs = (1+sh_degree)**2
        self.linear = nn.ModuleList(
            [nn.Linear(self.xyz_input_ch + self.t_input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.xyz_input_ch + self.t_input_ch, W)
                for i in range(D - 1)]
        )

        self.xyz_warp = nn.Linear(W, 3)
        self.rot = nn.Linear(W, 4)
        self.r = nn.Linear(W, self.num_shs)
        self.g = nn.Linear(W, self.num_shs)
        self.b = nn.Linear(W, self.num_shs)
        self.a = nn.Linear(W, self.num_shs)

    def initialize_weights(self, args):
        self.isotropic = args.isotropic_gaussians
        for l in self.linear:
            torch.nn.init.xavier_normal_(l.weight)
            torch.nn.init.constant_(l.bias, val=0.0)
        if args.xavier_init_dxyz:
            torch.nn.init.xavier_normal_(self.xyz_warp.weight)
        else:
            torch.nn.init.normal_(self.xyz_warp.weight, mean=0.0, std=1e-5)
        torch.nn.init.constant_(self.xyz_warp.bias, val=0.0)
        torch.nn.init.normal_(self.rot.weight, mean=0.0, std=1e-5)
        torch.nn.init.constant_(self.rot.bias, val=0.0)

        torch.nn.init.normal_(self.r.weight, mean=0.0, std=1e-5)
        torch.nn.init.constant_(self.r.bias, val=0.0)
        torch.nn.init.normal_(self.g.weight, mean=0.0, std=1e-5)
        torch.nn.init.constant_(self.g.bias, val=0.0)
        torch.nn.init.normal_(self.b.weight, mean=0.0, std=1e-5)
        torch.nn.init.constant_(self.b.bias, val=0.0)
        torch.nn.init.normal_(self.a.weight, mean=0.0, std=1e-5)
        torch.nn.init.constant_(self.a.bias, val=0.0)

    def forward(self, x, t):
        x_emb = self.embed_xyz_fn(x)
        t_emb = self.embed_t_fn(t)

        h = torch.cat([x_emb, t_emb], dim=-1)
        
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)
        
        d_xyz = self.xyz_warp(h)
        d_sh_r = self.r(h)
        d_sh_g = self.g(h)
        d_sh_b = self.b(h)
        d_sh_a = self.a(h)
        d_sh_p = torch.zeros_like(d_sh_a).float().cuda()

        if self.isotropic:
            d_rot = 0.0
        else:
            d_rot = self.rot(h)
        return d_xyz, torch.zeros(d_rot.shape).cuda(), torch.stack([d_sh_r, d_sh_g, d_sh_b], dim=-1), torch.zeros(torch.stack([d_sh_p, d_sh_a], dim=-1).shape).cuda()