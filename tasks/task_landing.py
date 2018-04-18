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
        self.action_low = 300
        self.action_high = 500
        self.nof_rotors = 4
        self.action_size = action_size
        self.runtime = runtime

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
    def get_reward(self):
        """Uses current pose of sim to return reward."""
       
        ######################
        reward=0
        penalty = 0
        
        state = self.sim.pose
        vel = self.sim.v
        goal = self.target_pos
        pos = state[:3]
        ang  = state[4:6]
        angv  = state[7:9]
        reward = 0
        
        distance = np.linalg.norm(goal-pos)
        z_distance = abs(self.sim.pose[2] - self.target_pos[2])
        
        # Penalties
        penalty = 10*z_distance**2                      # vertical distance from target
        penalty += distance                      # distance from target
        #penalty += abs(self.sim.pose[3:6]).sum()   # euler angles
        
        z_speed_penalty_gain = ((100-z_distance)/10)*10.0
            
        penalty += z_speed_penalty_gain*abs(self.sim.v).sum()           # velocities
        penalty += 0.03*abs(self.sim.angular_v).sum()   # euler angle velocities
                    
        #if distance < 5:
        #    reward += 100
        #    if self.sim.time >= self.runtime:
        #        reward +=100
        
        #reward = self.sim.time*100
        reward -= penalty
        #reward = np.maximum(reward, -6000.0)
        #print(reward)
        #reward = np.maximum(reward, -100.0)
        #reward = np.minimum(reward, +100.0)
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
            pose_all.append(self.sim.pose)
        
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state