import numpy as np
from experience import Experience

class Coach():

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]

    def to_experiences(self, states, actions, next_states, rewards, dones):
        experiences = []
        for (state, action, next_state, reward, done) in zip(states, actions, next_states, rewards, dones):
            experiences.append(Experience(state, action, reward, next_state, done))
        return experiences

    def run_episode(self):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations
        reward_history = []
        while True:
            actions = self.agent.act(states)
            env_info = self.env.step(actions)[self.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            reward_history.append(rewards)
            for experience in self.to_experiences(states, actions, next_states, rewards, dones):
                self.agent.learn(experience)
            if dones[0]:
                break
            states = next_states

        self.agent.end_episode()
        return np.array(reward_history).sum(0).mean()

    def diagnostic(self, episode, rewards, average_rewards_over):
        reward_window = rewards[-average_rewards_over:]
        mean_reward = np.mean(reward_window)
        max_reward = np.max(reward_window)
        if (episode + 1) % 20 == 0:
            end = "\n"
        else:
            end = ""
        print("\rEpisode: {}, Mean: {}, Max: {}, Last: {}".format(episode, mean_reward, max_reward, rewards[-1]), end=end)


    def run_episodes(self, num_episodes):
        rewards = []
        for i in range(num_episodes):
            rewards.append(self.run_episode())
            self.diagnostic(i, rewards, 20)
