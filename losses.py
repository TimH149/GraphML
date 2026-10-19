"""Multi-label losses on logits: BCE, BCE with per-label pos_weight, and the asymmetric loss (ASL).

All losses average over nodes and labels, so their scale is comparable to plain BCE and the
same learning rate can be used for all of them.
"""
import torch
import torch.nn.functional as F

LOSSES = ["bce", "bce_weighted", "asl"]


def label_pos_weight(y_train):
    """#negatives / #positives per label on the training nodes (labels without positives get weight 1)."""
    num_pos = y_train.sum(dim=0)
    num_neg = y_train.shape[0] - num_pos
    return torch.where(num_pos > 0, num_neg / num_pos.clamp(min=1), torch.ones_like(num_pos))


def asymmetric_loss(logits, labels, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
    """Asymmetric loss (Ridnik et al., 2021).

    positive term:  (1 - p)^gamma_pos * log(p)
    negative term:  p_m^gamma_neg * log(1 - p_m),  with the shifted probability p_m = max(p - clip, 0)
    """
    p = torch.sigmoid(logits)
    p_m = (p - clip).clamp(min=0) if clip > 0 else p

    loss_pos = labels * (1 - p).pow(gamma_pos) * F.logsigmoid(logits)
    loss_neg = (1 - labels) * p_m.pow(gamma_neg) * torch.log((1 - p_m).clamp(min=eps))
    return -(loss_pos + loss_neg).mean()


class MultiLabelLoss:
    """The loss selected with --loss, set up once from the training labels."""

    def __init__(self, name, y_train=None, gamma_pos=0.0, gamma_neg=4.0, clip=0.05):
        self.name = name
        self.gamma_pos, self.gamma_neg, self.clip = gamma_pos, gamma_neg, clip
        self.pos_weight = label_pos_weight(y_train) if name == "bce_weighted" else None

    def __call__(self, logits, labels):
        if self.name == "bce":
            return F.binary_cross_entropy_with_logits(logits, labels)
        elif self.name == "bce_weighted":
            return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.pos_weight)
        elif self.name == "asl":
            return asymmetric_loss(logits, labels, self.gamma_pos, self.gamma_neg, self.clip)
        raise ValueError(f"Unknown loss '{self.name}', expected one of {LOSSES}")
