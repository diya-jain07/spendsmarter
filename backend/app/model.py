"""
Budget Risk Predictor - Neural Network Model (PyTorch)

Architecture:
  Input:  10 features [5x last_month_spend + 5x planned_budget]
  Layer1: Linear(10 -> 8) + ReLU
  Layer2: Linear(8 -> 5)  + Sigmoid
  Output: 5 probabilities [one per spending category]

Trained on synthetic behavioral data to recognize overspending patterns.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

CATEGORIES = ['groceries', 'rent', 'utilities', 'entertainment', 'misc']


class BudgetRiskNet(nn.Module):
    """Neural network classifier for per-category overspend risk."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 8, output_dim: int = 5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def generate_synthetic_data(n_samples: int = 10_000, seed: int = 42):
    """
    Simulate realistic spending behavioral data for training.

    Behavioral patterns modeled:
    - Consistent overspenders in entertainment/misc
    - Seasonal utility spikes
    - Rent as near-fixed cost (low variance)
    - Budget anchoring bias (people underestimate)
    """
    rng = np.random.default_rng(seed)

    # Base monthly budgets per category (realistic distributions)
    budget_means = np.array([400, 1400, 130, 250, 150])  # groceries, rent, utilities, entertainment, misc
    budget_stds =  np.array([80,  200,  30,  100, 60])

    budgets = rng.normal(budget_means, budget_stds, (n_samples, 5)).clip(min=50)

    # Spending patterns with behavioral biases
    # spend_ratio = actual_spend / budget, >1.0 = overspend
    spend_ratios = np.ones((n_samples, 5))

    # Groceries: slight upward drift, moderate variance
    spend_ratios[:, 0] = rng.normal(1.05, 0.20, n_samples)

    # Rent: very stable, rarely overspent
    spend_ratios[:, 1] = rng.normal(0.98, 0.05, n_samples)

    # Utilities: seasonal variance
    spend_ratios[:, 2] = rng.normal(1.00, 0.25, n_samples)

    # Entertainment: highest overspend tendency
    spend_ratios[:, 3] = rng.normal(1.20, 0.35, n_samples)

    # Misc: high variance
    spend_ratios[:, 4] = rng.normal(1.08, 0.30, n_samples)

    # Actual spending
    spending = budgets * spend_ratios.clip(min=0.1)

    # Labels: 1 if overspent (spend > budget), 0 if not
    labels = (spending > budgets).astype(np.float32)

    # Feature vector: [last_month_spend x5, planned_budget x5]
    X = np.concatenate([spending, budgets], axis=1).astype(np.float32)
    y = labels

    return X, y


def train_model(n_samples: int = 10_000, epochs: int = 200, lr: float = 0.001) -> tuple:
    """Train the risk prediction model on synthetic behavioral data."""
    print("Generating synthetic training data...")
    X, y = generate_synthetic_data(n_samples)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y)

    model = BudgetRiskNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print(f"Training for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    print("Training complete.")
    return model, scaler


def save_model(model: BudgetRiskNet, scaler: StandardScaler, path: str = "model"):
    """Save model weights and scaler."""
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), f"{path}/risk_model.pt")
    with open(f"{path}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Model saved to {path}/")


def load_model(path: str = "model") -> tuple:
    """Load model and scaler from disk."""
    model = BudgetRiskNet()
    model.load_state_dict(torch.load(f"{path}/risk_model.pt", map_location="cpu"))
    model.eval()

    with open(f"{path}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def predict(
    model: BudgetRiskNet,
    scaler: StandardScaler,
    last_month_spend: list[float],
    next_budget: list[float]
) -> dict[str, float]:
    """
    Run inference for a single user.

    Args:
        last_month_spend: [groceries, rent, utilities, entertainment, misc]
        next_budget:       [groceries, rent, utilities, entertainment, misc]

    Returns:
        dict of {category: probability_of_overspend}
    """
    features = np.array([last_month_spend + next_budget], dtype=np.float32)
    features_scaled = scaler.transform(features)
    x = torch.FloatTensor(features_scaled)

    with torch.no_grad():
        probs = model(x).squeeze().numpy()

    return {cat: float(round(probs[i], 4)) for i, cat in enumerate(CATEGORIES)}


if __name__ == "__main__":
    # Train and save model
    model, scaler = train_model()
    save_model(model, scaler)

    # Test inference
    test_spend  = [480, 1200, 140, 320, 180]
    test_budget = [420, 1200, 130, 200, 150]
    result = predict(model, scaler, test_spend, test_budget)
    print("\nTest prediction:")
    for cat, prob in result.items():
        bar = "█" * int(prob * 20)
        print(f"  {cat:<14} {prob:.2%}  {bar}")
