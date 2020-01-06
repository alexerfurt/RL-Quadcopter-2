import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
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

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 40.]) 

    #def _tanh(self, x):
        #return (2 / (1 + np.exp(-2*x))) - 1
    
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        goal = False
        crashed = False
        # Introduce positional error w.r.t. target position
        pos_error = (abs(self.sim.pose[:3] - np.float32(self.target_pos))).sum() / 3
        # Normalize positional error with hyberbolic tangent to range -1-1
        reward = 1. -0.4*((2 / (1 + np.exp(-2*pos_error))) - 1)
        # ToDo: what to do if self.sim.pose[2] (=z-value) == 0 --> Crash the floor --> huge negative reward --> end the episode
        if self.sim.pose[2] < 0:
        #if self.sim.done and self.sim.runtime > self.sim.time:
            reward -= 10
            crashed = True
        # ToDO: what to do if self.sim.pose[2] == 50 --> Target position reached --> huge positive reward --> confirm goal is achieved
        if self.sim.pose[2] > 30 and pos_error < 20:
            reward += 5
            goal = True
        
        return reward, crashed, goal

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        success = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            new_reward, crashed, goal = self.get_reward()
            reward += new_reward
            pose_all.append(self.sim.pose)
            # Count if positional goal was reached
            if goal:
                success += 1
            # Make sure the episode ends if agent crashes, i.e. goes below 0 in z
            if crashed or reward <0:
                done=True
        next_state = np.concatenate(pose_all)
        return next_state, reward, done, success

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    
    