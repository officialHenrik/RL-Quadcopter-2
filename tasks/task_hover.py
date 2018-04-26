import numpy as np
from physics_sim import PhysicsSim

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
        self.action_repeat = 3

        self.state_size = self.action_repeat * len(self.sim.pose)
        self.action_low = 0 # 350
        self.action_high = 900 #550
        self.nof_rotors = 4
        self.action_size = action_size
        self.runtime = runtime
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
                
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        norm= np.linalg.norm([self.sim.pose[:3] - self.target_pos])
        z_norm= np.linalg.norm([self.sim.pose[2] - self.target_pos[2]])
        #reward = np.tanh(1 - 0.004*(norm))
        #return reward
    
        #reward = -min(norm**2, 100)
        reward = -(min(norm, 20)/20)*0.2
        reward -= min(z_norm, 20)/20
            
        reward -= (min(abs(self.sim.v[0]),20)/20)*0.2
        reward -= (min(abs(self.sim.v[1]),20)/20)*0.2
        reward -= (min(abs(self.sim.v[2]),20)/20)*0.2
        
        reward -= (min(abs(self.sim.angular_v[0]),20)/20)*0.2
        reward -= (min(abs(self.sim.angular_v[1]),20)/20)*0.2
        reward -= (min(abs(self.sim.angular_v[2]),20)/20)*0.2
        
        #reward = np.tanh(0.01*(reward))
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
                
        if self.action_size == 1:
            rotor_speeds = np.array([rotor_speeds[0]] * self.nof_rotors)
        
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose = self.sim.pose[:]*1.
            pose[:3] -= self.target_pos
            #pose[:3] /= self.target_pos[2]
            pose_all.append(pose)
        
        if self.sim.pose[2] > self.target_pos[2]*2:
            done = True

        if self.sim.time >= self.runtime:
            done = True
            
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self, runtime=5.):
        """Reset the sim to start a new episode."""
        self.runtime = runtime
        self.sim.reset(self.runtime)
                
        reset_pose = self.sim.pose[:]*1.
        reset_pose[:3] -= self.target_pos
        #reset_pose[:3] /= self.target_pos[2]*2.
        state = np.concatenate([reset_pose] * self.action_repeat) 
        
        return state