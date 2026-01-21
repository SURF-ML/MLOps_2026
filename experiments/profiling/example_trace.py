import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# Simple model: 2 matmul ops
# -------------------------
class SimpleMatmulModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(dim, dim))
        self.W2 = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        x = x @ self.W1          # matmul 1
        x = torch.relu(x)
        x = x @ self.W2          # matmul 2
        return F.log_softmax(x, dim=-1)


# -------------------------
# Training loop with profiler
# -------------------------
def train_profiling(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    log_interval,
    logdir,
):
    model.train()

    WAIT, WARMUP, ACTIVE, REPEAT = 10, 11, 10, 2

    # Initializing profiler
    with torch.profiler.profile(
        # Track CPU & GPU (CUDA) activity
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=WAIT,
            warmup=WARMUP,
            active=ACTIVE,
            repeat=REPEAT,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            logdir, worker_name="worker0"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as p:

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.6f}"
                )

            p.step()

            # Stop early to keep profiling short and deterministic
            if batch_idx == (WAIT + WARMUP + ACTIVE) * REPEAT:
                break


# -------------------------
# Main
# -------------------------
def main(logdir: str):
    assert torch.cuda.is_available(), "CUDA not available"

    device = torch.device("cuda")

    os.makedirs(logdir, exist_ok=True)

    # Fake data (kept intentionally simple)
    batch_size = 256
    dim = 4096
    num_batches = 200

    x = torch.randn((num_batches * batch_size, dim))
    y = torch.randint(0, dim, (num_batches * batch_size,))

    dataset = TensorDataset(x, y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    model = SimpleMatmulModel(dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_profiling(
        model=model,
        device=device,
        train_loader=loader,
        optimizer=optimizer,
        epoch=0,
        log_interval=10,
        logdir=logdir,
    )

    print(f"\nTensorBoard traces written to: {logdir}")
    print("Launch with:")
    print(f"  tensorboard --logdir {logdir}")


if __name__ == "__main__":
    logdir = "logs/simple_test"
    main(logdir)
