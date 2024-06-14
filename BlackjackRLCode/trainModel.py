import torch
import torch.nn as nn
import torch.nn.functional as F
from BlackjackEnv import BlackjackEnv


class BlackjackPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BlackjackPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs


input_dim = 4  # Number of state features (player_hand_value, dealer_visible_card, usable_ace, usable_pair)
output_dim = 4  # Number of actions (split, hit, stand, double)

policy = BlackjackPolicy(input_dim, output_dim)

# Example function to interact with the external Blackjack environment
def step_environment(state, action):
    # Implement your external environment step logic here
    # Example: calculate next_state, reward, done based on state and action
    next_state = state  # Placeholder; replace with actual logic
    reward = 0  # Placeholder; replace with actual logic
    done = False  # Placeholder; replace with actual logic
    return next_state, reward, done

# Example function to encode the state for the neural network
def encode_state(state):
    return torch.tensor(state, dtype=torch.float32)

# Custom VecEnv for Blackjack (assumed external environment)
class BlackjackVecEnv:
    def __init__(self, num_envs):
        super().__init__(num_envs)
        self.state = [(18, 6, False, False)] * num_envs  # Example initial states

    def reset(self):
        return [encode_state(s) for s in self.state]

    def step(self, actions):
        next_states, rewards, dones = [], [], []
        for i, action in enumerate(actions):
            next_state, reward, done = step_environment(self.state[i], action)
            self.state[i] = next_state
            next_states.append(encode_state(next_state))
            rewards.append(reward)
            dones.append(done)
        return next_states, rewards, dones, {}


def evaluate_policy(policy, env, num_episodes=100):
    total_rewards = 0

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            state_tensor = torch.tensor(encode_state(state)).unsqueeze(0)  # Convert state to tensor
            action_probs = policy(state_tensor)  # Get action probabilities
            action = torch.multinomial(action_probs, num_samples=1).item()  # Sample action

            next_state, reward, done = env.step(action)  # Take action in the environment
            total_rewards += reward
            state = next_state

    average_reward = total_rewards / num_episodes
    return average_reward

# Evaluate the trained policy
average_reward = evaluate_policy(policy, env)
print(f"Average Reward: {average_reward}")

# Initialize environment and policy
env = BlackjackVecEnv(num_envs=1)
algo = PPO(policy, env)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    algo.train()
    avg_reward = evaluate_policy(policy, env)
    print(f"Epoch {epoch}: Average Reward = {avg_reward}")

# Save trained policy
torch.save(policy.state_dict(), 'blackjack_policy.pth')