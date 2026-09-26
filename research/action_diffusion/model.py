"""KAD-MD: conditional absorbing-mask discrete diffusion over action chunks.

New model trained from scratch, inspired by masked discrete diffusion (D3PM),
NOT a DiffusionGemma checkpoint or port. No pretrained language weights.

With Config.condition_dim > 0 the model also reads the game it comes from (CONDITIONS: the date,
the team's rating below the day's best, the result, the margin above expectation), so it learns
how play differs between days, teams and results instead of averaging them, and can be asked for
one kind of play (a prompt such as "the newest day, the top team, a win").
"""
from dataclasses import asdict, dataclass
import datetime
import torch
from torch import nn
from torch.nn import functional as F

# A conditioned model's inputs about the game, in this order (encode_condition).
CONDITIONS = ('date', 'rating_gap', 'win', 'excess_margin')
EPOCH = datetime.date(2026, 7, 30)   # the first published replay day


def encode_condition(day=None, rating_gap=None, win=None, excess_margin=None):
    """One game's conditions as [values..., known flags...] (2 * len(CONDITIONS) floats); None is
    unknown (value 0, flag 0), and so is every condition of an unconditioned prediction.

    date: the day (YYYY-MM-DD or a date), as days after EPOCH / 60. rating_gap: the team's rating
    minus the day's best team rating (<= 0), / 100, from -3. win: 1 won, 0.5 tie, 0 lost, as
    +1 / 0 / -1. excess_margin: the seat's cash lead over its opponent minus the lead its rating
    edge predicts, both as fractions of the game's mean cash (data.add_ratings), / 0.1, within +-3.
    """
    if isinstance(day, str):
        day = datetime.date.fromisoformat(day)
    values = [None if day is None else (day - EPOCH).days / 60,
              None if rating_gap is None else max(min(rating_gap, 0.0), -300.0) / 100,
              None if win is None else 2 * win - 1,
              None if excess_margin is None else max(min(excess_margin / 0.1, 3.0), -3.0)]
    return [0.0 if v is None else float(v) for v in values] + [0.0 if v is None else 1.0 for v in values]


@dataclass
class Config:
    feature_dim: int
    field_sizes: list
    context: int = 64
    horizon: int = 24
    width: int = 384
    layers: int = 10
    heads: int = 6
    field_embed: int = 8
    context_patch: int = 4
    dropout: float = 0.1
    condition_dim: int = 0   # len(CONDITIONS) for a conditioned model, 0 for an unconditioned one


class ActionDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        assert c.context % c.context_patch == 0
        self.register_buffer('sizes', torch.tensor(c.field_sizes))
        offsets = torch.tensor([0] + [sum(n + 1 for n in c.field_sizes[:i]) for i in range(1, len(c.field_sizes))])
        self.register_buffer('offsets', offsets)
        self.embedding = nn.Embedding(sum(n + 1 for n in c.field_sizes), c.field_embed)
        self.action_proj = nn.Linear(len(c.field_sizes) * c.field_embed, c.width)
        self.context_proj = nn.Linear(c.feature_dim, c.width)
        self.time = nn.Sequential(nn.Linear(1, c.width), nn.SiLU(), nn.Linear(c.width, c.width))
        n = c.context // c.context_patch + c.horizon
        self.position = nn.Parameter(torch.randn(1, n, c.width) * .02)
        layer = nn.TransformerEncoderLayer(c.width, c.heads, 4*c.width, c.dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, c.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(c.width)
        self.output = nn.Linear(c.width, sum(c.field_sizes))
        self.register_buffer('vocab_mask', torch.arange(max(c.field_sizes))[None, :] < self.sizes[:, None])
        if c.condition_dim:
            # The conditions shift every token. The last layer starts at zero, so a conditioned model
            # grown from an unconditioned checkpoint (train.load_weights) starts out predicting as it.
            self.condition = nn.Sequential(nn.Linear(2 * c.condition_dim, c.width), nn.SiLU(),
                                           nn.Linear(c.width, c.width))
            nn.init.zeros_(self.condition[2].weight)
            nn.init.zeros_(self.condition[2].bias)

    def forward(self, context, context_mask, noisy_actions, noise_fraction, condition=None):
        """`condition`: (batch, 2 * condition_dim) from encode_condition; None = all unknown."""
        c = self.config
        b = context.shape[0]
        # Pool only known past observations, never future states or padded frames.
        ctx = context.reshape(b, c.context // c.context_patch, c.context_patch, c.feature_dim)
        mask = context_mask.reshape(b, c.context // c.context_patch, c.context_patch)
        ctx = (ctx * mask[..., None]).sum(2) / mask.sum(2).clamp_min(1)[..., None]
        ctx = self.context_proj(ctx)
        act = self.embedding(noisy_actions + self.offsets).flatten(-2)
        act = self.action_proj(act) + self.time(noise_fraction[:, None])[:, None, :]
        tokens = torch.cat([ctx, act], dim=1) + self.position
        if c.condition_dim:
            if condition is None:
                condition = torch.zeros(b, 2 * c.condition_dim, device=tokens.device)
            tokens = tokens + self.condition(condition.float())[:, None, :]
        padding =torch.cat([~mask.any(-1), torch.zeros(b, c.horizon, dtype=torch.bool, device=ctx.device)], 1)
        h = self.encoder(tokens, src_key_padding_mask=padding)[:, -c.horizon:]
        raw = self.output(self.norm(h))
        chunks = raw.split(c.field_sizes, dim=-1)
        logits = torch.stack([F.pad(x, (0, max(c.field_sizes) - n), value=-1e4)
                              for x, n in zip(chunks, c.field_sizes)], dim=-2)
        return logits

    def config_dict(self):
        return asdict(self.config)


def corrupt(actions, sizes, full_mask_probability=.25):
    b = actions.shape[0]
    t = torch.rand(b, device=actions.device) * .95 + .05
    t = torch.where(torch.rand_like(t) < full_mask_probability, torch.ones_like(t), t)
    mask = torch.rand(actions.shape, device=actions.device) < t[:, None, None]
    noisy = torch.where(mask, sizes[None, None, :], actions)
    return noisy, mask, t


def masked_loss(logits, targets, corruption_mask, action_mask):
    loss = F.cross_entropy(logits.float().flatten(0, 2), targets.flatten(), reduction='none').view_as(targets)
    valid = corruption_mask & action_mask[:, :, None]
    # Empty commands still learn their operation, but irrelevant args/counts
    # must not dominate the objective with easy zeros. NONE op is codec index 0.
    ops = targets[:, :, 0::3]
    active = (ops != 0).repeat_interleave(3, -1)
    active[:, :, 0::3] = True
    valid = valid & active
    return (loss * valid).sum() / valid.sum().clamp_min(1)


@torch.no_grad()
def predict(model, context, context_mask, steps=1, condition=None, guidance=1.0):
    """One-pass readout or confidence-ordered iterative mask removal.

    Per-slot marginals are not a calibrated joint distribution. Iterative
    prediction lets later denoising passes condition on earlier decisions.
    A conditioned model predicts the play `condition` describes (None: unconditioned); guidance > 1
    pushes further in that direction, logits = unconditioned + guidance * (conditioned -
    unconditioned) (classifier-free guidance: training hides the conditions at times, so the model
    knows both).
    """
    if steps < 1:
        raise ValueError('steps must be >= 1')
    c = model.config
    b = len(context)
    x = model.sizes[None, None, :].expand(b, c.horizon, -1).clone()
    probs = None
    for k in range(steps):
        masked = x == model.sizes
        t = masked.float().mean((1, 2))
        logits = model(context, context_mask, x, t, condition)
        if condition is not None and guidance != 1.0:
            free = model(context, context_mask, x, t)
            logits = free + guidance * (logits - free)
        probs = logits.float().softmax(-1)
        confidence, labels = probs.max(-1)
        if k == steps - 1:
            x = torch.where(masked, labels, x)
        else:
            score = confidence.masked_fill(~masked, -1).flatten(1)
            count = max(1, (c.horizon * len(c.field_sizes) + steps - 1) // steps)
            indices = score.topk(min(count, score.shape[1]), dim=1).indices
            update = torch.zeros_like(score, dtype=torch.bool).scatter_(1, indices, True).view_as(masked) & masked
            x = torch.where(update, labels, x)
    assert probs is not None
    return x, probs
