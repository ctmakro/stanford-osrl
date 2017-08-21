from __future__ import print_function

# gym boilerplate
import numpy as np
import gym
from gym import wrappers
from gym.spaces import Discrete, Box

from math import *
import random
import time

from winfrey import wavegraph

from rpm import rpm # replay memory implementation

from noise import one_fsq_noise

import tensorflow as tf
import canton as ct
from canton import *

from observation_processor import process_observation as po
from observation_processor import generate_observation as go

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    ex = np.exp(x)
    return ex / np.sum(ex, axis=0)

from triggerbox import TriggerBox

import traceback

from plotter import interprocess_plotter as plotter

class nnagent(object):
    def __init__(self,
    observation_space_dims,
    action_space,
    stack_factor=1,
    discount_factor=.99, # gamma
    # train_skip_every=1,
    train_multiplier=1,
    ):
        self.rpm = rpm(1000000) # 1M history
        self.plotter = plotter(num_lines=3)
        self.render = True
        self.training = True
        self.noise_source = one_fsq_noise()
        self.train_counter = 0
        # self.train_skip_every = train_skip_every
        self.train_multiplier = train_multiplier
        self.observation_stack_factor = stack_factor

        self.inputdims = observation_space_dims * self.observation_stack_factor
        # assume observation_space is continuous

        self.is_continuous = True if isinstance(action_space,Box) else False

        if self.is_continuous: # if action space is continuous

            low = action_space.low
            high = action_space.high

            num_of_actions = action_space.shape[0]

            self.action_bias = high/2. + low/2.
            self.action_multiplier = high - self.action_bias

            # say high,low -> [2,7], then bias -> 4.5
            # mult = 2.5. then [-1,1] multiplies 2.5 + bias 4.5 -> [2,7]

            def clamper(actions):
                return np.clip(actions,a_max=action_space.high,a_min=action_space.low)

            self.clamper = clamper
        else:
            num_of_actions = action_space.n

            self.action_bias = .5
            self.action_multiplier = .5 # map (-1,1) into (0,1)

            def clamper(actions):
                return np.clip(actions,a_max=1.,a_min=0.)

            self.clamper = clamper

        self.outputdims = num_of_actions
        self.discount_factor = discount_factor
        ids,ods = self.inputdims,self.outputdims
        print('inputdims:{}, outputdims:{}'.format(ids,ods))

        self.actor = self.create_actor_network(ids,ods)
        self.critic = self.create_critic_network(ids,ods)
        self.actor_target = self.create_actor_network(ids,ods)
        self.critic_target = self.create_critic_network(ids,ods)

        # print(self.actor.get_weights())
        # print(self.critic.get_weights())

        self.feed,self.joint_inference,sync_target = self.train_step_gen()

        sess = ct.get_session()
        sess.run(tf.global_variables_initializer())

        sync_target()

        import threading as th
        self.lock = th.Lock()

        if not hasattr(self,'wavegraph'):
            num_waves = self.outputdims*2+1
            def rn():
                r = np.random.uniform()
                return 0.2+r*0.4
            colors = []
            for i in range(num_waves-1):
                color = [rn(),rn(),rn()]
                colors.append(color)
            colors.append([0.2,0.5,0.9])
            self.wavegraph = wavegraph(num_waves,'actions/noises/Q',np.array(colors))

    # the part of network that the input and output shares architechture
    def create_common_network(self,inputdims,outputdims):
        # timesteps = 8
        # dim_per_ts = int(inputdims/timesteps)
        # rect = Act('relu')
        # c = Can()
        #
        # if not hasattr(self,'gru'):
        #     # share parameters between actor and critic
        #     self.common_gru = GRU(dim_per_ts,128)
        #
        # gru = c.add(self.common_gru)
        #
        # # d1 = c.add(Dense(128,128))
        # d2 = c.add(Dense(128,outputdims))
        #
        # def call(i):
        #     # shape i: [Batch Dim*Timesteps]
        #
        #     batchsize = tf.shape(i)[0]
        #
        #     reshaped = tf.reshape(i,[batchsize,timesteps,dim_per_ts])
        #     # [Batch Timesteps Dim]
        #
        #     o = gru(reshaped)
        #     # [Batch Timesteps Dim]
        #
        #     ending = o[:,timesteps-1,:]
        #
        #     # l1 = rect(d1(ending))
        #     l2 = rect(d2(ending))
        #     return l2
        #
        # c.set_function(call)
        # return c

        c = Can()
        rect = Act('lrelu',0.05)
        d1 = c.add(Dense(inputdims,128))
        d1a = c.add(Dense(128,128))
        d2 = c.add(Dense(128,outputdims))

        def call(i):
            l1 = rect(d1(i))
            l1a = rect(d1a(l1))
            l2 = rect(d2(l1a))
            return l2
        c.set_function(call)
        return c

    # a = actor(s) : predict actions given state
    def create_actor_network(self,inputdims,outputdims):
        # add gaussian noise.

        c = Can()
        c.add(self.create_common_network(inputdims,64))
        c.add(Dense(64,outputdims))

        if self.is_continuous:
            c.add(Act('tanh'))
            c.add(Lambda(lambda x: x*self.action_multiplier + self.action_bias))
        else:
            c.add(Act('softmax'))

        c.chain()
        return c

    # q = critic(s,a) : predict q given state and action
    def create_critic_network(self,inputdims,actiondims):
        c = Can()
        concat = Lambda(lambda x:tf.concat(x,axis=1))

        # concat state and action
        den0 = c.add(self.create_common_network(inputdims,64))
        # den1 = c.add(Dense(256, 256))
        den2 = c.add(Dense(64+actiondims, 128))
        den3 = c.add(Dense(128,64))
        den4 = c.add(Dense(64,1))

        rect = Act('lrelu',0.05)

        def call(i):
            state = i[0]
            action = i[1]
            i = den0(state)

            i = concat([i,action])
            i = den2(i)
            i = rect(i)
            i = den3(i)
            i = rect(i)
            i = den4(i)

            q = i
            return q
        c.set_function(call)
        return c

    def train_step_gen(self):
        s1 = tf.placeholder(tf.float32,shape=[None,self.inputdims])
        a1 = tf.placeholder(tf.float32,shape=[None,self.outputdims])
        r1 = tf.placeholder(tf.float32,shape=[None,1])
        isdone = tf.placeholder(tf.float32,shape=[None,1])
        s2 = tf.placeholder(tf.float32,shape=[None,self.inputdims])

        # 1. update the critic
        a2 = self.actor_target(s2)
        q2 = self.critic_target([s2,a2])
        q1_target = r1 + (1-isdone) * self.discount_factor * q2
        q1_predict = self.critic([s1,a1])
        critic_loss = tf.reduce_mean((q1_target - q1_predict)**2)
        # produce better prediction

        # 2. update the actor
        a1_predict = self.actor(s1)
        q1_predict = self.critic([s1,a1_predict])
        actor_loss = tf.reduce_mean(- q1_predict)
        # maximize q1_predict -> better actor

        # 3. shift the weights (aka target network)
        tau = tf.Variable(1e-3) # original paper: 1e-3. need more stabilization
        aw = self.actor.get_weights()
        atw = self.actor_target.get_weights()
        cw = self.critic.get_weights()
        ctw = self.critic_target.get_weights()

        one_m_tau = 1-tau

        shift1 = [tf.assign(atw[i], aw[i]*tau + atw[i]*(one_m_tau))
            for i,_ in enumerate(aw)]
        shift2 = [tf.assign(ctw[i], cw[i]*tau + ctw[i]*(one_m_tau))
            for i,_ in enumerate(cw)]

        # 4. inference
        a_infer = self.actor(s1)
        q_infer = self.critic([s1,a_infer])

        # 5. L2 weight decay on critic
        decay_c = tf.reduce_sum([tf.reduce_sum(w**2) for w in cw])* 0.0001
        # decay_a = tf.reduce_sum([tf.reduce_sum(w**2) for w in aw])* 0.0001

        # optimizer on
        # actor is harder to stabilize...
        opt_actor = tf.train.AdamOptimizer(1e-4)
        opt_critic = tf.train.AdamOptimizer(2e-4)
        # opt_actor = tf.train.RMSPropOptimizer(1e-3)
        # opt_critic = tf.train.RMSPropOptimizer(1e-3)
        cstep = opt_critic.minimize(critic_loss, var_list=cw)
        astep = opt_actor.minimize(actor_loss, var_list=aw)

        self.feedcounter=0
        def feed(memory):
            [s1d,a1d,r1d,isdoned,s2d] = memory # d suffix means data
            sess = ct.get_session()
            res = sess.run([critic_loss,actor_loss,
                cstep,astep,shift1,shift2],
                feed_dict={
                s1:s1d,a1:a1d,r1:r1d,isdone:isdoned,s2:s2d,tau:1e-3
                })

            #debug purposes
            self.feedcounter+=1
            if self.feedcounter%10==0:
                print(' '*30, 'closs: {:6.4f} aloss: {:6.4f}'.format(
                    res[0],res[1]),end='\r')

            # return res[0],res[1] # closs, aloss

        def joint_inference(state):
            sess = ct.get_session()
            res = sess.run([a_infer,q_infer],feed_dict={s1:state})
            return res

        def sync_target():
            sess = ct.get_session()
            sess.run([shift1,shift2],feed_dict={tau:1.})

        return feed,joint_inference,sync_target

    def train(self,verbose=1):
        memory = self.rpm
        batch_size = 64
        total_size = batch_size
        epochs = 1

        self.lock.acquire()
        if memory.size() > total_size * 128:

            #if enough samples in memory
            for i in range(self.train_multiplier):
                # sample randomly a minibatch from memory
                [s1,a1,r1,isdone,s2] = memory.sample_batch(batch_size)
                # print(s1.shape,a1.shape,r1.shape,isdone.shape,s2.shape)

                self.feed([s1,a1,r1,isdone,s2])

        self.lock.release()

    def feed_one(self,tup):
        self.rpm.add(tup)

    # gymnastics
    def play(self,env,max_steps=-1,realtime=False,noise_level=0.): # play 1 episode
        timer = time.time()
        noise_source = one_fsq_noise()

        for j in range(10):
            noise_source.one((self.outputdims,),noise_level)

        max_steps = max_steps if max_steps > 0 else 50000
        steps = 0
        total_reward = 0
        episode_memory = []

        # removed: state stacking
        # moved: observation processing

        try:
            observation = env.reset()
        except Exception as e:
            print('(agent) something wrong on reset(). episode terminates now')
            traceback.print_exc()
            print(e)
            return

        while True and steps <= max_steps:
            steps +=1

            observation_before_action = observation # s1

            exploration_noise = noise_source.one((self.outputdims,),noise_level)
            # exploration_noise -= noise_level * 1

            # self.lock.acquire() # please do not disrupt.
            action = self.act(observation_before_action, exploration_noise) # a1
            # self.lock.release()

            if self.is_continuous:
                # add noise to our actions, since our policy by nature is deterministic
                exploration_noise *= self.action_multiplier
                # print(exploration_noise,exploration_noise.shape)
                action += exploration_noise
                action = self.clamper(action)
                action_out = action
            else:
                raise RuntimeError('this version of ddpg is for continuous only.')

            # o2, r1,
            try:
                observation, reward, done, _info = env.step(action_out) # take long time
            except Exception as e:
                print('(agent) something wrong on step(). episode teminates now')
                traceback.print_exc()
                print(e)
                return

            # d1
            isdone = 1 if done else 0
            total_reward += reward

            # feed into replay memory
            if self.training == True:
                episode_memory.append((
                    observation_before_action,action,reward,isdone,observation
                ))

                # don't feed here since you never know whether the episode will complete without error.
                # self.feed_one((
                #     observation_before_action,action,reward,isdone,observation
                # )) # s1,a1,r1,isdone,s2
                # self.lock.acquire()
                self.train(verbose=2 if steps==1 else 0)
                # self.lock.release()

            # if self.render==True and (steps%30==0 or realtime==True):
            #     env.render()
            if done :
                break

        # print('episode done in',steps,'steps',time.time()-timer,'second total reward',total_reward)
        totaltime = time.time()-timer
        print('episode done in {} steps in {:.2f} sec, {:.4f} sec/step, got reward :{:.2f}'.format(
        steps,totaltime,totaltime/steps,total_reward
        ))
        self.lock.acquire()

        for t in episode_memory:
            self.feed_one(t)

        self.plotter.pushys([total_reward,noise_level,(time.time()%3600)/3600-2])
        # self.noiseplotter.pushy(noise_level)
        self.lock.release()

        return

    # one step of action, given observation
    def act(self,observation,curr_noise=None):
        actor,critic = self.actor,self.critic
        obs = np.reshape(observation,(1,len(observation)))

        # actions = actor.infer(obs)
        # q = critic.infer([obs,actions])[0]
        self.lock.acquire()
        [actions,q] = self.joint_inference(obs)
        self.lock.release()

        actions,q = actions[0],q[0]

        if curr_noise is not None:
            disp_actions = (actions-self.action_bias) / self.action_multiplier
            disp_actions = disp_actions * 5 + np.arange(self.outputdims) * 12.0 + 30

            noise = curr_noise * 5 - np.arange(self.outputdims) * 12.0 - 30

            self.lock.acquire()
            self.loggraph(np.hstack([disp_actions,noise,q]))
            self.lock.release()
            # temporarily disabled.
        return actions

    def loggraph(self,waves):
        wg = self.wavegraph
        wg.one(waves.reshape((-1,)))

    def save_weights(self):
        networks = ['actor','critic','actor_target','critic_target']
        for name in networks:
            network = getattr(self,name)
            network.save_weights('ddpg_'+name+'.npz')

    def load_weights(self):
        networks = ['actor','critic','actor_target','critic_target']
        for name in networks:
            network = getattr(self,name)
            network.load_weights('ddpg_'+name+'.npz')

from osim.env import RunEnv

if __name__=='__main__':
    # p = playground('LunarLanderContinuous-v2')
    # p = playground('Pendulum-v0')
    # p = playground('MountainCar-v0')BipedalWalker-v2
    # p = playground('BipedalWalker-v2')
    # e = p.env

    e = RunEnv(visualize=False)
    from observation_processor import processed_dims

    agent = nnagent(
    processed_dims,
    e.action_space,
    discount_factor=.985,
    # .99 = 100 steps = 4 second lookahead
    # .985 = somewhere in between.
    # .98 = 50 steps = 2 second lookahead
    # .96 = 25 steps = 1 second lookahead
    stack_factor=1,
    train_multiplier=4,
    )

    noise_level = 2.
    noise_decay_rate = 0.002
    noise_floor = 0.2
    noiseless = 0.005

    # show_sim = False
    #
    # from multi import eipool # multiprocessing driven simulation pool
    # epl = None
    # def newpool(count=6):
    #     global epl,show_sim
    #     tepl = epl
    #     epl = eipool(count,showfirst=show_sim)
    #     del tepl
    #
    # newpool()

    from farmer import farmer as farmer_class
    # from multi import fastenv

    # one and only
    farmer = farmer_class()

    def refarm():
        global farmer
        del farmer
        farmer = farmer_class()

    stopsimflag = False
    def stopsim():
        global stopsimflag
        print('stopsim called')
        stopsimflag = True

    tb = TriggerBox('Press a button to do something.',
        ['stop simulation'],
        [stopsim])

    def playonce(nl,env):
        from multi import fastenv

        # global noise_level
        # env = farmer.acq_env()
        fenv = fastenv(env,4)
        agent.play(fenv,realtime=False,max_steps=-1,noise_level=nl)
        # epl.rel_env(env)
        env.rel()
        del fenv

    def play_ignore(nl,env):
        import threading as th
        t = th.Thread(target=playonce,args=(nl,env),daemon=True)
        t.start()
        # ignore and return.

    def playifavailable(nl):
        while True:
            remote_env = farmer.acq_env()
            if remote_env == False: # no free environment
                # time.sleep(0.1)
                pass
            else:
                play_ignore(nl,remote_env)
                break

    def r(ep,times=1):
        global noise_level,stopsimflag
        # agent.render = True
        # e = p.env
        for i in range(ep):
            if stopsimflag:
                stopsimflag = False
                print('(run) stop signal received, stop at ep',i+1)
                break

            noise_level *= (1-noise_decay_rate)
            noise_level = max(noise_floor, noise_level)

            nl = noise_level if np.random.uniform()>0.05 else noiseless

            print('ep',i+1,'/',ep,'times:',times,'noise_level',nl)
            # playtwice(times)
            playifavailable(nl)

            time.sleep(0.01)

            if (i+1) % 200 == 0:
                # global epl
                #
                # while True: # wait until all free
                #     if epl.all_free():
                #         break
                #     else:
                #         print('wait until all of epl free..')
                #         time.sleep(0.5)

                # # reset the env to prevent memory leak.
                # newpool()

                # save the training result.
                save()

            # following is commented out. dont renew() every so often, impacts performance.
            # if (i+1) % 1000 == 0:
            #     farmer.renew()

    def test():
        # e = p.env
        agent.render = True
        agent.play(e,realtime=True,max_steps=-1,noise_level=1e-11)

    def save():
        agent.save_weights()
        agent.rpm.save('rpm.pickle')

    def load():
        agent.load_weights()
        agent.rpm.load('rpm.pickle')

    def up():
        # uploading to CrowdAI

        # global _stepsize
        # _stepsize = 0.01

        apikey = open('apikey.txt').read().strip('\n')
        print('apikey is',apikey)

        import opensim as osim
        from osim.http.client import Client
        from osim.env import RunEnv

        # Settings
        remote_base = "http://grader.crowdai.org:1729"
        crowdai_token = apikey

        client = Client(remote_base)

        # Create environment
        observation = client.env_create(crowdai_token)
        # old_observation = None
        stepno= 0
        epino = 0
        total_reward = 0
        old_observation = None
        def obg(plain_obs):
            nonlocal old_observation, stepno
            processed_observation, old_observation = go(plain_obs, old_observation, step=stepno)
            return np.array(processed_observation)


        print('environment created! running...')
        # Run a single step
        while True:
            proc_observation = obg(observation)

            [observation, reward, done, info] = client.env_step(
                [float(i) for i in list(agent.act(proc_observation))],
                True
            )
            stepno+=1
            total_reward+=reward
            print('step',stepno,'total reward',total_reward)
            # print(observation)
            if done:
                observation = client.env_reset()
                old_observation = None

                print('>>>>>>>episode',epino,' DONE after',stepno,'got_reward',total_reward)
                total_reward = 0
                stepno = 0
                epino+=1

                if not observation:
                    break

        print('submitting...')
        client.submit()

        # _stepsize = 0.04
