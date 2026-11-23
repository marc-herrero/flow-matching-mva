import os
import torch
import matplotlib.pyplot as plt

from distributions import sample_dist
from model import VectorFieldMLP
from solvers import solve_euler, solve_midpoint


def sample_time(batch_size, device, schedule="uniform", beta_alpha=3.0, beta_beta=1.0):
    if schedule == "uniform":
        return torch.rand(batch_size, 1, device=device)
    elif schedule == "beta":
        return torch.distributions.Beta(beta_alpha, beta_beta).sample((batch_size, 1)).to(device)


def run_experiment(
    p0="normal",              # Starting noise distribution p0
    dataset="bounded_ball",   # Synthetic dataset distribution q(x1)
    p1_cond="dirac",          # Conditional blob shape p1(. | x1)
    sigma_min=0.0,            # Spread of p1(. | x1); 0.0 means Dirac at x1
    dim=1,                    # Dimension (1 or 2 for built-in plots)
    time_schedule="uniform",  # "uniform" or "beta"
    beta_alpha=3.0,
    beta_beta=1.0,
    solver="euler",           # "euler" or "midpoint"
    nfe=100,                  # Total network evaluations at inference
    steps=3000,
    batch_size=256,
    width=128,
    depth=4,
    lr=1e-3,
    radius=3.0,               # Threshold R for tail escape P(||x|| > R)
    seed=0,
):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VectorFieldMLP(dim=dim, width=width, depth=depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_means = []
    loss_vars = []

    print(f"\n=== p0={p0} | q={dataset} | p1_cond={p1_cond} (sigma={sigma_min}) | t={time_schedule} ===")

    #Training
    for step in range(1, steps + 1):
        optimizer.zero_grad()

        x0 = sample_dist(p0, batch_size, dim, device)
        x1 = sample_dist(dataset, batch_size, dim, device)

        if p1_cond == "dirac" or sigma_min == 0.0:
            y1 = x1
        else:
            y1 = x1 + sigma_min * sample_dist(p1_cond, batch_size, dim, device)

        t = sample_time(batch_size, device, time_schedule, beta_alpha, beta_beta)
        xt = (1.0 - t) * x0 + t * y1
        ut = y1 - x0

        vt = model(xt, t)
        per_sample_loss = ((vt - ut) ** 2).sum(dim=-1)

        loss = per_sample_loss.mean()
        loss.backward()
        optimizer.step()

        loss_means.append(loss.item())
        loss_vars.append(per_sample_loss.var().item())

        if step % 500 == 0:
            print(f"Step {step:5d} | mean loss: {loss.item():10.4f} | loss var: {loss_vars[-1]:12.4f}")

    
    # TODO: rehacer experimentos con lo que queremos y sin IA xd

    # --- 2. Diagnostics & Sampling ---
    model.eval()
    n_test = 5000
    x0_test = sample_dist(p0, n_test, dim, device)
    x1_true = sample_dist(dataset, n_test, dim, device)
    mean_x1 = x1_true.mean(dim=0, keepdim=True)

    # 1-step Euler collapse + median metric D
    with torch.no_grad():
        t0 = torch.zeros((n_test, 1), device=device)
        v0 = model(x0_test, t0)
        D_metric = (x0_test - mean_x1 + v0).norm(dim=-1).median().item()

    x1_euler1 = solve_euler(model, x0_test, n_steps=1)

    if solver == "euler":
        x1_gen = solve_euler(model, x0_test, n_steps=nfe)
    elif solver == "midpoint":
        x1_gen = solve_midpoint(model, x0_test, n_steps=max(1, nfe // 2))
    else:
        raise ValueError(f"Unknown solver: {solver}")

    outlier_rate = (x1_gen.norm(dim=-1) > radius).float().mean().item()
    print("--- Evaluation ---")
    print(f"1-step Euler median error D:       {D_metric:.4f}")
    print(f"1-step Euler std (collapse check): {x1_euler1.std(dim=0).mean().item():.4f}")
    print(f"Tail escape P(||x1_gen|| > {radius}):  {100 * outlier_rate:.2f}%")

    # --- 3. Save vector PDF plot ---
    os.makedirs("results", exist_ok=True)
    tag = f"p0-{p0}_q-{dataset}_p1c-{p1_cond}_sig-{sigma_min}_t-{time_schedule}_{solver}_d{dim}"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(loss_means, label="Batch Mean Loss", alpha=0.8)
    axes[0].plot(loss_vars, label="Batch Loss Variance", alpha=0.6)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Training step")
    axes[0].set_title(f"Training Stability ({p0} -> {dataset})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    x1_true_np = x1_true.cpu().numpy()
    x1_gen_np = x1_gen.cpu().numpy()
    x1_e1_np = x1_euler1.cpu().numpy()

    if dim == 1:
        axes[1].hist(x1_true_np[:, 0], bins=60, range=(-8, 8), density=True, alpha=0.5, label="True q(x1)")
        axes[1].hist(x1_gen_np[:, 0], bins=60, range=(-8, 8), density=True, alpha=0.5, label=f"Generated ({solver}, NFE={nfe})")
        axes[1].axvline(mean_x1.item(), color="k", linestyle="--", label="E[X1]")
        axes[1].set_title(f"1D Output | Escape (|x|>{radius}): {100 * outlier_rate:.2f}%")
        axes[1].legend()
    elif dim == 2:
        axes[1].scatter(x1_true_np[:1500, 0], x1_true_np[:1500, 1], s=5, alpha=0.35, label="True q(x1)")
        axes[1].scatter(x1_gen_np[:1500, 0], x1_gen_np[:1500, 1], s=5, alpha=0.35, label=f"Generated ({solver}, NFE={nfe})")
        axes[1].scatter(x1_e1_np[:300, 0], x1_e1_np[:300, 1], s=8, alpha=0.6, label="1-step Euler")
        axes[1].set_xlim(-6, 6)
        axes[1].set_ylim(-6, 6)
        axes[1].set_aspect("equal")
        axes[1].set_title(f"2D Output | Escape (||x||>{radius}): {100 * outlier_rate:.2f}%")
        axes[1].legend()

    plt.tight_layout()
    out_path = f"results/{tag}.pdf"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    # 1. Heavy-tailed priors -> bounded dataset
    for prior in ["normal", "laplace", "student_t_5", "student_t_3", "cauchy"]:
        run_experiment(p0=prior, dataset="bounded_ball", p1_cond="dirac", sigma_min=0.0, dim=1)

    # 2. Skewed time sampling (weighting t -> 1) on Cauchy
    run_experiment(p0="cauchy", dataset="bounded_ball", time_schedule="beta", beta_alpha=3.0, beta_beta=1.0, dim=1)

    # 3. Reverse direction (Gaussian prior -> Cauchy dataset)
    run_experiment(p0="normal", dataset="cauchy", dim=1)

    # 4. Non-Gaussian conditional blob p1(. | x1): tiny vs. large sigma_min
    run_experiment(p0="normal", dataset="bimodal", p1_cond="laplace", sigma_min=0.001, dim=2)
    run_experiment(p0="normal", dataset="bimodal", p1_cond="laplace", sigma_min=0.5, dim=2)
