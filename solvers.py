import torch


@torch.no_grad()
def solve_euler(model, x0, n_steps):
    x = x0.clone()
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((x.shape[0], 1), k * dt, device=x.device, dtype=x.dtype)
        x = x + dt * model(x, t)
    return x


@torch.no_grad()
def solve_midpoint(model, x0, n_steps):
    x = x0.clone()
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((x.shape[0], 1), k * dt, device=x.device, dtype=x.dtype)
        t_mid = t + 0.5 * dt
        v1 = model(x, t)
        x_mid = x + 0.5 * dt * v1
        v2 = model(x_mid, t_mid)
        x = x + dt * v2
    return x
