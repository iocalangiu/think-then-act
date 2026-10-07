import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. ENVIRONMENT & ADVERSARY RANDOMIZER TOOL
# ==========================================
class RandomizedEnv:
    def __init__(self):
        # Using a standard robotic/control env as a stand-in (e.g., Reacher or custom MuJoCo)
        self.env = gym.make("Reacher-v4", render_mode=None)
        
    def reset(self):
        obs, info = self.env.reset()
        # Adversary Tool: Perturb physics/conditions on reset
        # For demonstration, we simulate mass/friction tweak or observation noise
        self.current_perturbation = np.random.uniform(0.5, 2.0) # e.g., mass multiplier
        return obs, info

    def step(self, action):
        # Apply perturbation effect in dynamics (simplified here via action scaling/noise)
        perturbed_action = action * self.current_perturbation
        obs, reward, terminated, truncated, info = self.env.step(perturbed_action)
        return obs, reward, terminated, truncated, info

# ==========================================
# 2. NEURAL NETWORK MODELS (BC & World Model)
# ==========================================
class BehavioralCloningPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Tanh()
        )
    def forward(self, x): return self.net(x)

class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64), nn.ReLU(),
            nn.Linear(64, state_dim)
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

# ==========================================
# 3. PIPELINE EXECUTION SCRIPT
# ==========================================
def main():
    env = RandomizedEnv()
    state_dim = env.env.observation_space.shape[0]
    action_dim = env.env.action_space.shape[0]

    print("Step 1: Collecting data with randomized conditions...")
    states, actions, next_states = [], [], []
    for episode in range(50):
        obs, _ = env.reset()
        for _ in range(50):
            action = env.env.action_space.sample() # Expert / Random collector placeholder
            next_obs, _, terminated, truncated, _ = env.step(action)
            
            states.append(obs)
            actions.append(action)
            next_states.append(next_obs)
            obs = next_obs
            if terminated or truncated: break

    S = torch.tensor(np.array(states), dtype=torch.float32)
    A = torch.tensor(np.array(actions), dtype=torch.float32)
    S_next = torch.tensor(np.array(next_states), dtype=torch.float32)

    print("Step 2: Training Behavioral Cloning Policy & World Model...")
    bc_policy = BehavioralCloningPolicy(state_dim, action_dim)
    world_model = WorldModel(state_dim, action_dim)
    
    optimizer_bc = optim.Adam(bc_policy.parameters(), lr=1e-3)
    optimizer_wm = optim.Adam(world_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(100): # Quick training loop
        # Train BC
        pred_actions = bc_policy(S)
        loss_bc = loss_fn(pred_actions, A)
        optimizer_bc.zero_grad(); loss_bc.backward(); optimizer_bc.step()

        # Train World Model
        pred_next_s = world_model(S, A)
        loss_wm = loss_fn(pred_next_s, S_next)
        optimizer_wm.zero_grad(); loss_wm.backward(); optimizer_wm.step()

    print("Step 3: Validation of Expectation (Detecting Failures/Surprise)...")
    obs, _ = env.reset()
    state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    
    for step in range(20):
        with torch.no_grad():
            action = bc_policy(state_tensor)
            # World Model predicts what *should* happen
            predicted_next_state = world_model(state_tensor, action)
            
        # Real environment step
        next_obs, _, _, _, _ = env.step(action.numpy().squeeze())
        real_next_state = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        
        # Validation Metric: Prediction Error (Surprise)
        prediction_error = torch.mean(torch.abs(predicted_next_state - real_next_state)).item()
        
        print(f"Step {step}: Adversary Perturbation Level = {env.current_perturbation:.2f} | World Model Error = {prediction_error:.4f}")
        
        if prediction_error > 0.1:
            print(f" -> [ALERT] High surprise detected! The adversary's perturbation broke the world model's expectation.")

        state_tensor = real_next_state

if __name__ == "__main__":
    main()