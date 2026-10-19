"""Label cardinality head: predicts how many labels k_i each node has, for top-k decoding.

The head reads the model's final node representation (model.all_emb, the fused module outputs
just before the output layer) and is trained jointly with the main loss:

    loss = main_loss + lambda_card * cardinality_loss

  ce       classification over k in {1, ..., K_max} with cross-entropy; k_pred is the argmax
  poisson  Poisson regression on k with a log-rate output; k_pred is the rounded rate
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CARD_HEADS = ["none", "ce", "poisson"]


def cardinality_kmax(y_train, cap=None):
    """K_max: the largest number of labels of a training node, optionally capped."""
    k_max = int(y_train.sum(dim=1).max().item())
    return max(1, min(k_max, cap) if cap is not None else k_max)


class CardinalityHead(nn.Module):
    def __init__(self, input_dim, k_max, kind="ce", hidden_dim=64, detach=False):
        super(CardinalityHead, self).__init__()
        self.kind = kind
        self.k_max = k_max
        self.detach = detach
        # the fused representation concatenates modules of very different scales, hence the LayerNorm
        self.mlp = nn.Sequential(nn.LayerNorm(input_dim),
                                 nn.Linear(input_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, k_max if kind == "ce" else 1))

    def forward(self, h):
        # with detach=True the cardinality loss does not change the shared representation
        return self.mlp(h.detach() if self.detach else h)

    def targets(self, y):
        """True cardinalities clipped to [1, K_max] (top-k decoding always predicts at least one label)."""
        return y.sum(dim=1).round().long().clamp(1, self.k_max)

    def loss(self, out, y):
        k = self.targets(y)
        if self.kind == "ce":
            return F.cross_entropy(out, k - 1)
        return F.poisson_nll_loss(out.squeeze(1), k.float(), log_input=True)

    def predict(self, out):
        if self.kind == "ce":
            return out.argmax(dim=1) + 1
        return out.squeeze(1).exp().round().long().clamp(1, self.k_max)
