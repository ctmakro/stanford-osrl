import traceback

# get PPO to run for L2R
import tensorflow as tf
import numpy as np
from canton import *
import gym

import threading as th
import math, time

from ppo import MultiCategoricalContinuous
from ppo2 import ppo_agent2, SingleEnvSampler, flatten
from triggerbox import TriggerBox
from farmer import farmer as farmer_class

# instead of repeatedly using one environment instance, obtain a new one everytime current environment is done.
class DisposingSampler(SingleEnvSampler):
    def get_env(self): # obtain a new environment on demand
        global farmer
        while 1:
            remote_env = farmer.acq_env()
            if remote_env == False: # no free environment
                time.sleep(0.1)
            else:
                if hasattr(self, 'remote_env'):
                    del self.remote_env # release previous before allocate new

                self.remote_env = remote_env
                from multi import fastenv
                fenv = fastenv(remote_env,2)
                # a skip of 2; also performs observation processing
                return fenv

    def __init__(self, agent):
        super().__init__(env=None, agent=agent)

# policy for L2R.
class AwesomePolicy(Can):
    def __init__(self, ob_space, ac_space):
        super().__init__()

        # 1. assume probability distribution is continuous
        assert len(ac_space.shape) == 1
        self.ac_dims = ac_dims = ac_space.shape[0]
        self.ob_dims = ob_dims = ob_space.shape[0]

        # 2. build our action network
        rect = Act('tanh')
        # apparently John doesn't give a fuck about ReLUs. Change the rectifiers as you wish.
        rect = Act('lrelu',alpha=0.2)
        magic = 1/(0.5+0.5*0.2) # stddev factor for lrelu(0.2)

        c = Can()
        c.add(Dense(ob_dims, 128, stddev=magic))
        c.add(rect)
        c.add(Dense(128, 64, stddev=magic))
        c.add(rect)
        c.add(Dense(64, 64, stddev=magic))
        c.add(rect)
        c.add(Dense(64, ac_dims*3, stddev=1))
        # self.dist = c.add(Bernoulli())
        self.dist = c.add(MultiCategoricalContinuous(ac_dims, 3))
        c.chain()
        self.actor = self.add(c)

        # 3. build our value network
        c = Can()
        c.add(Dense(ob_dims, 128, stddev=magic))
        c.add(rect)
        c.add(Dense(128, 64, stddev=magic))
        c.add(rect)
        c.add(Dense(64, 64, stddev=magic))
        c.add(rect)
        c.add(Dense(64, 1, stddev=1))
        c.chain()
        self.critic = self.add(c)


if __name__ == '__main__':
    farmer = farmer_class()

    stopsimflag = False
    def stopsim():
        global stopsimflag
        print('stopsim called')
        stopsimflag = True

    tb = TriggerBox('Press a button to do something.',
        ['stop simulation'],
        [stopsim])

    from osim.env import RunEnv
    runenv = RunEnv(visualize=False)
    from gym.spaces import Box

    from observation_processor import processed_dims
    ob_space = Box(-1.0, 1.0, (processed_dims,))

    agent = ppo_agent2(
        ob_space, runenv.action_space,
        horizon=128, # minimum steps to collect before policy update
        gamma=0.99, # discount factor for reward
        lam=0.95, # smooth factor for advantage estimation
        train_epochs=10, # how many epoch over data for one update
        batch_size=128, # batch size for training
        buffer_length=16,

        policy=AwesomePolicy
    )

    get_session().run(gvi()) # init global variables for TF

    # parallelized
    process_count = 16 # total horizon = process_count * agent.horizon
    samplers = [DisposingSampler(agent) for i in range(process_count)]

    def r(iters=2):
        global stopsimflag
        print('start running')
        for i in range(iters):
            if stopsimflag:
                stopsimflag = False
                print('(run) stop signal received, stop at iter',i+1)
                break
            print('optimization iteration {}/{}'.format(i+1, iters))
            # agent.iterate_once(env)
            agent.iterate_once_on_samplers(samplers)

    def save():
        agent.current_policy.save_weights('ppo_pol.npz')
        agent.old_policy.save_weights('ppo_old.npz')

    def load():
        agent.current_policy.load_weights('ppo_pol.npz')
        agent.old_policy.load_weights('ppo_old.npz')
