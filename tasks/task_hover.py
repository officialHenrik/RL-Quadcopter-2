import numpy as np
from physics_sim import PhysicsSim
import math as m
    
class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, action_size=4):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation 
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 5

        self.state_size = self.action_repeat * len(self.sim.pose)
        self.action_low = 0
        self.action_high = 900
        self.action_delta_abs = 2
        self.nof_rotors = 4
        self.action_size = action_size
        self.runtime = runtime
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
                
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        
        def sigmoid(x):
            import math
            return 1 / (1 + math.exp(-x))

        reward = sigmoid(np.linalg.norm(self.sim.pose[:3] - self.target_pos[:3]))
        reward = (0.5 - sigmoid(reward))
        return reward
        

    def step(self, actions):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
                
        if self.action_size == 1:
            rotor_speeds = np.array([actions[0]] * self.nof_rotors)
        elif self.action_size == 5:
            rotor_speeds = np.array([actions[4]] * self.nof_rotors)
            for i in range(self.nof_rotors):
                rotor_speeds[i] += (actions[i]/self.action_high) * 2.*self.action_delta_abs - self.action_delta_abs
        elif self.action_size == 4:
            rotor_speeds = actions
        else:
            print("WARNING: Unknown actions size")
        
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            
            # Change pose x,y,z to target error
            pose = self.sim.pose[:]*1.
            
            pose[:3] -= self.target_pos
            pose[3:]  = self.sim.v
            
            pose_all.append(pose)
        
        # Stop if high above
        #if self.sim.pose[2] > self.target_pos[2]*2:
        #    done = True

        # Stop when time runs out
        if self.sim.time >= self.runtime:
            done = True
        
        # Stop if over-rotating
        #if(abs(pose[3]) > m.pi) or (abs(pose[4]) > m.pi):
        #   done = True
           
        rotor_speeds_used = rotor_speeds
        next_state = np.concatenate(pose_all)
        return next_state, reward, done, rotor_speeds_used

    def reset(self, runtime=5.):
        """Reset the sim to start a new episode."""
        self.runtime = runtime
        self.sim.reset(self.runtime)
                
        reset_pose = self.sim.pose[:]*1.
        reset_pose[:3] -= self.target_pos
        reset_pose[3:] -= self.sim.v
        state = np.concatenate([reset_pose] * self.action_repeat) 
        
        return state