import torch
import torch.nn as nn
import torch.nn.functional as F

class ElephantNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class RiderNet(nn.Module):
    def __init__(self, input_dim, elephant_dim, hidden_dim=64, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + elephant_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, elephant_output):
        combined = torch.cat([x, elephant_output], dim=-1)
        return self.net(combined)


class GatingNetwork(nn.Module):
    def __init__(self, elephant_dim):
        super().__init__()
        self.linear = nn.Linear(elephant_dim, 1)

    def forward(self, elephant_output):
        return torch.sigmoid(self.linear(elephant_output))


class ElephantRiderModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.elephant = ElephantNet(input_dim)
        self.rider = RiderNet(input_dim, elephant_dim=64)
        self.gating = GatingNetwork(elephant_dim=64)

    def forward(self, x):
        e_out = self.elephant(x)
        r_out = self.rider(x, e_out)
        alpha = self.gating(e_out)

        alpha_expanded = alpha.expand_as(e_out)
        y = alpha_expanded * e_out + (1 - alpha_expanded) * r_out

        return y, alpha


if __name__ == "__main__":
    model = ElephantRiderModel(input_dim=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for step in range(50):
        x = torch.randn(32, 10)
        target = torch.randn(32, 64)

        output, alpha = model(x)
        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}, Alpha mean: {alpha.mean().item():.4f}")
