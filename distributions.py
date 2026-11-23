import torch
import torch.distributions as D


def sample_dist(name, n, dim=1, device="cpu"):
    """Shared sampler for p0 (noise), q(x1) (dataset), or p1(.|x1) (conditional blob)."""
    zero = torch.tensor(0.0, device=device)
    one = torch.tensor(1.0, device=device)

    if name == "normal":
        return D.Normal(zero, one).sample((n, dim))
    elif name == "laplace":
        return D.Laplace(zero, one).sample((n, dim))
    elif name.startswith("student_t_"):
        df = float(name.split("_")[-1])  # e.g. "student_t_5" or "student_t_3"
        return D.StudentT(df=torch.tensor(df, device=device)).sample((n, dim))
    elif name == "cauchy":
        return D.Cauchy(zero, one).sample((n, dim))
    elif name == "exponential":
        return D.Exponential(one).sample((n, dim))
    elif name == "uniform":
        return D.Uniform(-one, one).sample((n, dim))
    elif name == "gamma":
        return D.Gamma(torch.tensor(2.0, device=device), one).sample((n, dim))
    elif name == "bounded_ball":
        # Uniform inside ball of radius R = 3 (satisfies ||X1|| <= 3)
        z = torch.randn(n, dim, device=device)
        z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        r = 3.0 * (torch.rand(n, 1, device=device) ** (1.0 / dim))
        return r * z
    elif name == "bimodal":
        signs = torch.randint(0, 2, (n, 1), device=device).float() * 2.0 - 1.0
        centers = torch.zeros(n, dim, device=device)
        centers[:, 0:1] = 2.0 * signs
        return centers + 0.4 * torch.randn(n, dim, device=device)
    else:
        raise ValueError(f"Unknown distribution: {name}")
