import numpy as np
from experience import Experience

class Coach():

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]

    def to_experiences(self, states, actions, rewards, next_states, dones):
        experiences = []
        for (state, action, reward, next_state, done) in zip(states, actions, rewards, next_states, dones):
            experiences.append(Experience(state, action, reward, next_state, done))
        return experiences

    def run_episode(self, max_steps):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations
        scores = np.zeros(len(states))
        for i in range(max_steps):
            actions = self.agent.act(states)
            env_info = self.env.step(actions)[self.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += rewards
            for experience in self.to_experiences(states, actions, rewards, next_states, dones):
                self.agent.learn(experience)
            if dones[0]:
                break
            states = next_states

        self.agent.end_episode()
        return scores.mean()

    def diagnostic(self, episode, scores, average_scores_over):
        score_window = scores[-average_scores_over:]
        mean_score = np.mean(score_window)
        max_score = np.max(score_window)
        if (episode + 1) % 10 == 0:
            end = "\n"
        else:
            end = ""
        print("\rEpisode: {}, Mean: {}, Max: {}, Last: {}".format(episode, mean_score, max_score, scores[-1]), end=end)


    def run_episodes(self, num_episodes, max_steps):
        scores = []
        for i in range(num_episodes):
            scores.append(self.run_episode(max_steps))
            self.diagnostic(i, scores, 20)
