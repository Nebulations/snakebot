import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
from collections import deque

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Create the first layer of the model, between the input
        # and the middle layers
        # There are 15 inputs:
        # | The snake's direction (4): up, down, left, right
        # | The apple's position relative to the snake's head (4): up, down, left, right
        # | If theres danger (wall/snake body) (3): in front, left, right
        # | Quadrants based from the snake's head. (top left, right. bottom left, right)
        # | Ray casting from the head to left, right, in front and behind the head.
        self.layer_1 = nn.Linear(19, 256)
        # Second layer of the model: between the middle layer and
        # the output layer.
        # 3 outputs: go forward, turn left or turn right.
        self.layer_2 = nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x

    def save(self, file_name="model.pth"):
        if not os.path.exists("./models"):
            os.makedirs("./models")

        file_name = os.path.join("./models", file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        file_name = os.path.join("./models", file_name)
        self.load_state_dict(torch.load(file_name))
        self.eval()

class Trainer():
    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        prediction = self.model(state)

        target = prediction.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()

MAX_MEMORY = 100000
BATCH_SIZE = 1000

class Agent:
    def __init__(self, training: bool = True, model_file: str | None = None):
        self.games_played = 0
        self.epsilon = 1.0 if training else 0.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Network()
        if not training:
            self.model.load(model_file)
        self.trainer = Trainer(self.model, learning_rate=0.001, gamma=0.9)

    def get_action(self, state):
        action = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            with torch.no_grad():
                q_values = self.model(torch.tensor(state, dtype=torch.float))
            move = torch.argmax(q_values).item()
        action[move] = 1
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short(self, state, action, reward, next_state, done):
        self.trainer.train(state, action, reward, next_state, done)
    
    def train_long(self):
        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE)
        else:
            batch = list(self.memory)

        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train(states, actions, rewards, next_states, dones)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)