import random
from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.rewards = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.rewards.append(reward)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        #from numpy.random import choice
        #return choice(self.memory, size=batch_size, replace=False, p=(self.rewards/sum(self.rewards)))
        return random.sample(self.memory,batch_size )
        

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)