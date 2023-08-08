import gym
import torch
import torch.nn as nn
import random



class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.99, device="cpu"):
        self.device = torch.device(device)
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.loss_fn = nn.MSELoss()

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                q_values = self.model(torch.tensor(state, dtype=torch.float32).to(self.device))
                return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        action = torch.tensor(action).to(self.device)

        if done:
            target = reward
        else:
            with torch.no_grad():
                target = reward + self.gamma * torch.max(self.model(next_state))

        prediction = self.model(state_tensor)[action]

        loss = self.loss_fn(target, prediction)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print('Model saved!', path, sep='\n')

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        print('Model loaded!', path, sep='\n')


def train(agent, env, episodes, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    epsilon = epsilon_start
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.get_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    env.close()


device = "cuda" if torch.cuda.is_available() else "cpu"
env = gym.make('CartPole-v1')
agent = DQNAgent(state_size=4, action_size=2, device=device)
train(agent, env, episodes=10000)

print('Training finished!')

agent.save('model/model.pth')


