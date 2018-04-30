import random
from collections import namedtuple, deque
import numpy as np

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
        self.buffer_size = buffer_size
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.rewards.append(np.sum(reward))

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        from numpy.random import choice
        
        return random.sample(self.memory,batch_size )
        
        # Prioritized Experience Replay
        #  http://pemami4911.github.io/paper-summaries/deep-rl/2016/01/26/prioritizing-experience-replay.html
        p = np.array(self.rewards)
        if(min(p) < 0):
            p -= 2*min(p)
        p_sum = p.sum()
        if(p_sum == 0):
            return random.sample(self.memory,batch_size )
        
        p = np.divide(p, p_sum)
        selIdx = choice(len(self.memory), size=batch_size, replace=False, p=p)       
        batch = [val for i, val in enumerate(self.memory) if i in selIdx]
        return batch
        

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)