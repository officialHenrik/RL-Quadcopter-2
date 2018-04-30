from agents.actor import Actor
from agents.critic import Critic
from agents.replay_buffer import ReplayBuffer
from agents.ou_noise import OUNoise
import numpy as np

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    """        Deep Determini, stic Policy Gradients         """
    def __init__(self, task, 
                 gamma = 0.99, 
                 tau = 0.01, 
                 exploration_mu=0, 
                 exploration_theta=0.15, 
                 exploration_sigma=0.2,
                 lr_critic=0.001,
                 lr_actor =0.0001):
        
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        
        self.lr_critic = lr_critic
        self.lr_actor  = lr_actor 
        
        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, self.lr_actor, lr_decay=0)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, self.lr_actor, lr_decay=0)
        self.actor_best = Actor(self.state_size, self.action_size, self.action_low, self.action_high, self.lr_actor, lr_decay=0)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, self.lr_critic, lr_decay=0)
        self.critic_target = Critic(self.state_size, self.action_size, self.lr_critic, lr_decay=0)

        self.actor_best_score = -np.inf
        
        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = exploration_mu #0
        self.exploration_theta = exploration_theta # 0.15
        self.exploration_sigma = exploration_sigma # 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 1000000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = gamma  # discount factor
        self.tau = tau #0.01 # for soft update of target parameters

        # Score tracker?
        self.total_reward = -np.inf
        self.loss = 0
        
        print("Actor model:")
        self.actor_target.model.summary()
        print("Critic model:")
        self.critic_target.model.summary()
        
    def save_actor(self, mean_reward):
        if self.actor_best_score < mean_reward:
            self.actor_best_score = mean_reward
            self.soft_update(self.actor_target.model, self.actor_best.model, 1)
        
    def reset_episode(self, new_runtime=5.):
        
        self.total_reward = 0
        self.noise.reset()
        state = self.task.reset(new_runtime)
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward               
        self.memory.add(self.last_state, action, reward, next_state, done)

        self.total_reward += np.sum(reward)
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
             
        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        actions = list(action + self.noise.sample())  # add some noise for exploration
        #actions = list(action)  # no noise for exploration
        actions = np.maximum(actions, self.action_low)
        actions = np.minimum(actions, self.action_high)
        return actions
    
    def act_target(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_target.model.predict(state)[0]
        #actions = list(action + self.noise.sample())  # add some noise for exploration
        actions = list(action)  # no noise for exploration
        return actions
    
    def act_best_explore(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_best.model.predict(state)[0]
        actions = list(action + self.noise.sample())  # add some noise for exploration
        actions = np.maximum(actions, self.action_low)
        actions = np.minimum(actions, self.action_high)
        return actions
    
    def act_best(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_best.model.predict(state)[0]
        #actions = list(action + self.noise.sample())  # add some noise for exploration
        actions = list(action)  # no noise for exploration
        return actions
    
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        
        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.loss = self.actor_local.train_fn([states, action_gradients, 1])  # custom training function
        
        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model, self.tau)
        self.soft_update(self.actor_local.model, self.actor_target.model, self.tau) 
           
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = tau * local_weights + (1 - tau) * target_weights
        target_model.set_weights(new_weights)
        