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
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #return reward
    
        state = self.sim.pose
        vel = self.sim.v
        goal = self.target_pos
        pos = state[:3]
        ang  = state[4:6]
        angv  = state[7:9]
        reward = 0
        
        
        distance = np.linalg.norm(goal-pos)
        #print(distance)
        distance = np.minimum(distance, 20)
        #distance = np.minimum(distance, 10)
        sum_vel = np.linalg.norm(vel)
        sum_ang = np.linalg.norm(ang)
        
        reward_gain = 1.
        #if(pos[2] > 10):
        #    reward_gain = 0.2
        reward = (10.-(distance**2)*reward_gain) - sum_ang*0.5 - sum_vel*2.0
        
        if(distance < 5):
            reward += 100
            
        #return reward
        
        ######################
        # https://github.com/dbagaev/RL-Quadcopter-2/blob/solution/tasks/hover.py
        reward=0
        
        def apply_lower_threshold(x, threshold):
            return np.maximum(x, threshold)

        def apply_higher_threshold(x, threshold):
            return max(0.0, threshold - x)
        
        target_z_distance = abs(goal[2] - pos[2])
        height_penalty = target_z_distance
        
        reward += apply_higher_threshold(target_z_distance, 5.0) * 10
        
        ######################
        reward = 1 - .0005*(abs(self.sim.pose[2] - self.target_pos[2]))
        
        
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
        penalty += 0.03*abs(self.sim.v).sum()           # velocities
        penalty += 0.03*abs(self.sim.angular_v).sum()   # euler angle velocities
                    
        #if distance < 5:
        #    reward += 100
        #    if self.sim.time >= self.runtime:
        #        reward +=100
        
        reward = self.sim.time*100
        reward -= penalty
        reward = np.maximum(reward, -6000.0)
        #print(reward)
        #reward = np.maximum(reward, -100.0)
        #reward = np.minimum(reward, +100.0)
        return reward
        
        ################################
        reward =0 
        if distance < 1.0:
            reward=1 
        elif distance < 2.0:
            reward=0.5
            
        if abs(self.sim.v).sum() < 1:
            reward+=1
        if abs(self.sim.angular_v).sum() < 0.02:
            reward+=1
            
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
        
        if self.sim.pose[2] > 2*self.target_pos[2]:
            done = True
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state