import torch
import torch.nn as nn

class PDN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, mode):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim

        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.core = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.vault_proj = nn.Linear(hidden_dim, hidden_dim)

        # Used only by pdn_no_skip, but always defined for clean loading/counting.
        self.vault_recur = nn.Linear(hidden_dim, hidden_dim)

        head_in = hidden_dim * 2

        if mode == "pdn_mlp_head":
            self.head = nn.Sequential(
                nn.Linear(head_in, head_in),
                nn.ReLU(),
                nn.Linear(head_in, output_dim),
            )
        else:
            self.head = nn.Linear(head_in, output_dim)

    def forward(self, x, return_stats=False):
        batch = x.shape[0]
        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        S = torch.zeros(batch, self.hidden_dim, device=x.device)

        gate_means = []
        diffuse_norms = []
        raw_norms = []

        actual_mode = "pdn_full" if self.mode == "pdn_mlp_head" else self.mode

        for t in range(x.shape[1]):
            xt = self.x_proj(x[:, t])
            z = h + xt

            h_raw = torch.tanh(self.core(z) + z)

            if actual_mode == "pdn_no_gate":
                g = torch.ones_like(h_raw)
            else:
                g = torch.sigmoid(self.gate(h_raw))

            D = g * h_raw

            if actual_mode == "pdn_no_subtraction":
                h = h_raw
            else:
                h = h_raw - D

            if actual_mode == "pdn_no_archive":
                S = torch.zeros_like(S)
            elif actual_mode == "pdn_no_skip":
                S = torch.tanh(self.vault_recur(S) + self.vault_proj(D))
            elif actual_mode == "pdn_leaky_vault":
                S = 0.95 * S + self.vault_proj(D)
            else:
                S = S + self.vault_proj(D)

            if return_stats:
                gate_means.append(g.mean().detach())
                diffuse_norms.append(D.norm(dim=-1).mean().detach())
                raw_norms.append(h_raw.norm(dim=-1).mean().detach())

        combined = torch.cat([h, S], dim=-1)
        logits = self.head(combined)

        if return_stats:
            stats = {
                "gate_mean": torch.stack(gate_means).mean().item(),
                "diffuse_norm": torch.stack(diffuse_norms).mean().item(),
                "raw_norm": torch.stack(raw_norms).mean().item(),
                "pipe_norm": h.norm(dim=-1).mean().item(),
                "vault_norm": S.norm(dim=-1).mean().item(),
            }
            return logits, stats

        return logits
