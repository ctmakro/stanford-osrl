import tensorflow as tf
import numpy as np
from canton import *
import gym

from ppo import ppo_agent
import threading as th
import math, time

# given an environment, start stepping through it when signaled. yield trajectories every [horizon] steps.
class SingleEnvSampler:
    def __init__(self,env,agent):
        self.env = env
        self.agent = agent
        self.running = False
        self.horizon = 0
        self.lock = th.Lock()

        # long-running version of agent.collect_trajectories()
        def collect_trajectories_longrunning():
            policy = self.agent.current_policy
            # length we are going to collect
            # horizon = self.agent.horizon

            env = self.env
            print('collecting trajectory...')

            # things we have to collect
            s1 = [] # observations before action
            a1 = [] # action taken
            r1 = [] # reward received
            _done = [] # is the episode done after a1

            # counters
            ep = 0
            steps = 0

            while 1:
                # episode start
                episode_total_reward = 0
                episode_length = 0

                # initial observation
                ob = env.reset()
                while 1:
                    # sample action from given policy
                    mean, sto, val_pred = self.agent.act(ob)
                    # policy_out, val_pred = self.act(ob)
                    # sto_action = 1.0*(policy_out > noise_sample())
                    sto_action = sto
                    # sto_action = noise_sample() * std + mean
                    # sto_limited = self.action_limiter(sto_action)
                    mean_limited, sto_limited = self.agent.action_limiter(mean), self.agent.action_limiter(sto_action)

                    # logging actions. comment out if you don't have opencv
                    if True:
                        # mean_limited = self.action_limiter(mean)
                        disp_mean = mean_limited*5. + np.arange(policy.ac_dims)*12 + 30
                        disp_sto = sto_limited*5. - np.flipud(np.arange(policy.ac_dims))*12 - 30
                        self.agent.loggraph(np.hstack([disp_mean, disp_sto, val_pred]))

                    # step environment with action and obtain reward
                    new_ob, reward, done, info = env.step(sto_limited)

                    # if steps%100==0:
                    #     env.render()

                    # append data into collection
                    s1.append(ob)
                    a1.append(sto_action)
                    r1.append(reward)
                    _done.append(1 if done else 0)

                    ob = new_ob # assign new_ob to prev ob

                    # counting
                    episode_total_reward+=reward
                    episode_length+=1
                    steps+=1

                    # if episode is done, either natually or forcifully
                    if done or episode_length >= 1600:
                        done = 1
                        _done[-1] = 1
                        print('episode {} done in {} steps, total reward:{}'.format(
                            ep+1, episode_length, episode_total_reward,
                        ))
                        self.agent.plotter.pushys([ep,episode_total_reward])
                        # break

                    if steps % self.horizon == 0: # if enough steps collected
                        s2 = new_ob
                        s1.append(s2)
                        yield [s1,a1,r1,_done]
                        s1,a1,r1,_done = [],[],[],[] # clear collection
                        ep = 0

                    if done:
                        break

                ep+=1

        def loop():
            traj_generator = collect_trajectories_longrunning()
            while 1:
                if self.running == True:
                    self.running = False
                    self.collected = traj_generator.__next__()
                else:
                    time.sleep(0.2)

        t = th.Thread(target=loop, daemon=True)
        t.start()

    # start collecting [horizon] samples from env
    def start_collecting(self, horizon):
        self.horizon = horizon # set horizon
        self.collected = None
        self.running = True

    # block until enough samples collected from env
    def get_result(self):
        while 1:
            c = self.collected
            if c is None:
                time.sleep(0.1)
            else:
                return c

flatten = lambda l: [item for sublist in l for item in sublist]

class ppo_agent2(ppo_agent):

    # perform one policy iteration w/ a sampler
    def iterate_once_on_sampler(self, sampler):
        # 0. assign new to old
        self.assign_old_eq_new()

        # # 1. collect trajectories w/ current policy
        # collected = self.collect_trajectories(env)

        sampler.start_collecting(self.horizon) # signal start
        collected = sampler.get_result() # blocking

        # 2. push the trajectories into buffer
        self.traj_buffer.push(collected)

        # 3. load historic trajectories from buffer
        collected = self.traj_buffer.get_all_raw()

        # 4. estimate advantage and TD(lambda) return
        collected = [self.append_vtarg_and_adv(c) for c in collected] # process each individually

        collected = self.chain_list_of_trajectories(collected)

        # 5. processing, numpyization
        collected = self.usual_data_processing(collected)
        # s1,a1,r1,done,advantage,tdlamret = self.usual_data_processing(collected)

        # 6. train for some epochs
        self.usual_feed_training(collected)

        print('iteration done.')

    # [[[s1],[a1],[r1]...],[[s1],[a1],[r1]...]...] => [[s1],[a1],[r1]...]
    def chain_list_of_trajectories(self, collected_list):
        # this function was designed with extreme carefulness to prevent data corruption...
        collected = collected_list

        # join all collected together
        c0 = list(collected[0])
        for i in range(1, len(collected)):
            for j in range(len(collected[i])):
                c0[j] = c0[j] + collected[i][j]

        return c0

    # perform one policy iteration w/ an array of samplers
    def iterate_once_on_samplers(self, samplers):
        self.assign_old_eq_new()

        [s.start_collecting(self.horizon) for s in samplers] # signal start
        collected = [s.get_result() for s in samplers] # blocking

        # buffer
        self.traj_buffer.push(collected)
        collected = self.traj_buffer.get_all_raw()

        collected = flatten(collected) # reduce 1 layer of lists

        collected = [self.append_vtarg_and_adv(c) for c in collected] # process each individually

        collected = self.chain_list_of_trajectories(collected)

        # 5. ready to train
        collected = self.usual_data_processing(collected)

        # 6. train for some epochs
        self.usual_feed_training(collected)

        print('iteration done.')

# what environment to use in this training
def get_env():
    # get swingy
    envname = 'Pendulum-v0'
    envname = 'BipedalWalker-v2'
    return gym.make(envname)

def remote_env_loop(conn):
    env = get_env()
    while 1:
        msg = conn.recv()
        if msg[0] == 'step':
            conn.send(env.step(msg[1]))
        elif msg[0] == 'reset':
            conn.send(env.reset())
        else:
            return

from ipc import ipc
class remote_env(ipc):
    def __init__(self):
        super().__init__(remote_env_loop)

    def step(self,actions):
        actions = [float(i) for i in actions]
        self.send(('step', actions))
        return self.recv()

    def reset(self):
        self.send(('reset',))
        return self.recv()

if __name__ == '__main__':
    env = get_env()

    agent = ppo_agent2(
        env.observation_space, env.action_space,
        horizon=512, # minimum steps to collect before policy update
        gamma=0.99, # discount factor for reward
        lam=0.95, # smooth factor for advantage estimation
        train_epochs=10, # how many epoch over data for one update
        batch_size=128, # batch size for training
        buffer_length=16,
    )

    get_session().run(gvi()) # init global variables for TF

    go_parallel = True
    if go_parallel:
        # parallelized
        process_count = 4
        samplers = [SingleEnvSampler(remote_env(), agent) for i in range(process_count)]

        def r(iters=2):
            print('start running')
            for i in range(iters):
                print('optimization iteration {}/{}'.format(i+1, iters))
                # agent.iterate_once(env)
                agent.iterate_once_on_samplers(samplers)
    else:
        # parallelized infrastructure, but test with one instance only
        sampler = SingleEnvSampler(env, agent)

        def r(iters=2):
            print('start running')
            for i in range(iters):
                print('optimization iteration {}/{}'.format(i+1, iters))
                # agent.iterate_once(env)
                agent.iterate_once_on_sampler(sampler)
    r(1000)
