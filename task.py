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
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

        # For distance normalization
        try:
            self.norm=[1.0 if (target_pos[i] - init_pose[i]) == 0 else \
                       np.linalg.norm([init_pose[i], target_pos[i]]) for i in range(3)]
            self.norm_target = self.target_pos / self.norm
        except TypeError:
            self.norm = [1.0, 1.0, 1.0]

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def get_reward_2(self):
        """Uses current pose and velocity of sim to return reward."""
        def dist_reward(xyz):
            normalized = xyz / self.norm
            # normalized diff of x, y, and z
            diff = abs(normalized - self.norm_target)
            if min(diff) < 0.03:
                # close enough to the target
                return 1.0
            # normalized average diff of x, y and z
            av_diff = sum(diff) / 3.0
            return max(1 - av_diff**0.4, -1.0)

        # crush
        if self.sim.pose[2] < 0:
            return -1.0    
        reward = dist_reward(self.sim.pose[:3])
        # going up
        if self.sim.v[2] > 0:
            reward += 0.1
        return min(reward, 1.0)

    def step(self, rotor_speeds, opt=False):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() if not opt else self.get_reward_2()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state