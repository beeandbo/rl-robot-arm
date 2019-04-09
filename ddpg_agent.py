import numpy as np
import random
from critic_network import CriticNetwork
from actor_network import ActorNetwork
import torch
from replay_buffer import ReplayBuffer
import torch.optim as optim
import torch.nn.functional as F
from ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess

REPLAY_BUFFER_SIZE = 10**6
BATCH_SIZE = 64
STEPS_BETWEEN_TRAINING = 20 * 20 # 20 agents for 20 steps
ITERATIONS_PER_TRAINING = 10
GAMMA = 0.99
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 3e-4
TAU = 0.001 # Rate at which target networks are updated
CRITIC_WEIGHT_DECAY = 0.0001

# Random process parameters
RANDOM_THETA = 0.15
RANDOM_SIGMA = 0.2


class DDPGAgent():
    def __init__(self, state_size, action_size, num_agents):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        # Actor
        self.local_actor_network = ActorNetwork(state_size, action_size)
        self.target_actor_network = ActorNetwork(state_size, action_size)
        self.actor_optimizer = optim.Adam(self.local_actor_network.parameters(), lr=ACTOR_LEARNING_RATE)

        # Critic
        self.local_critic_network = CriticNetwork(state_size, action_size)
        self.target_critic_network = CriticNetwork(state_size, action_size)
        self.critic_optimizer = optim.Adam(self.local_critic_network.parameters(), lr=CRITIC_LEARNING_RATE, weight_decay=CRITIC_WEIGHT_DECAY)

        self.replay_buffer = ReplayBuffer(action_size, REPLAY_BUFFER_SIZE, None)
        self.steps = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.random_process = OrnsteinUhlenbeckProcess((num_agents, action_size), sigma=RANDOM_SIGMA, theta=RANDOM_THETA)


    def act(self, states, eval = False):
        self.local_actor_network.eval()
        with torch.no_grad():
            actions = self.local_actor_network(torch.tensor(states, dtype=torch.float32)).detach().numpy()
        self.local_actor_network.train()
        if not eval:
            actions = actions + self.random_process.sample()
        actions = np.clip(actions, -1, 1)
        return actions

    def vectorize_experiences(self, experiences):
        """Vectorizes experience objects for use by pytorch

        Params
        ======
            experiences (array_like of Experience objects): Experiences to
                vectorize
        """
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def normalize(self, to_normalize):
        std = to_normalize.std(0)
        mean = to_normalize.mean(0)
        return (to_normalize - mean)/std

    def soft_update(self, target_parameters, local_parameters):
        for target, local in zip(target_parameters, local_parameters):
            target.data.copy_(TAU*local.data + (1.0-TAU)*target.data)

    def train(self, experiences):
        print("Training")
        states, actions, rewards, next_states, dones = self.vectorize_experiences(experiences)
        #states = self.normalize(states)
        #next_states = self.normalize(next_states)

        # Use the target critic network to calculate a target q value
        next_actions = self.target_actor_network(next_states)
        q_target = rewards + GAMMA * self.target_critic_network(next_states, next_actions) * (1-dones)

        # Calculate the predicted q value
        q_predicted = self.local_critic_network(states, actions)

        # Update critic network
        critic_loss = F.mse_loss(q_predicted, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.local_critic_network.parameters(), 1)
        self.critic_optimizer.step()

        # Update predicted action using policy gradient
        actions_predicted = self.local_actor_network(states)
        policy_loss = -self.local_critic_network(states, actions_predicted).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.target_actor_network.parameters(), self.local_actor_network.parameters())
        self.soft_update(self.target_critic_network.parameters(), self.local_critic_network.parameters())

    def learn(self, experience):
        self.replay_buffer.add(experience)
        self.steps += 1
        if self.steps % STEPS_BETWEEN_TRAINING == 0 and len(self.replay_buffer) >= BATCH_SIZE:
            for i in range(ITERATIONS_PER_TRAINING):
                self.train(self.replay_buffer.sample(BATCH_SIZE))

    def end_episode(self):
        self.random_process.reset()
        self.steps = 0
