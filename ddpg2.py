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

def ResDense(nip):
    c = Can()
    nbp = int(nip/2)
    d0 = c.add(Dense(nip,nbp))
    d1 = c.add(Dense(nbp,nip))
    def call(i):
        inp = i
        i = d0(i)
        i = Act('elu')(i)
        i = d1(i)
        i = Act('elu')(i)
        return i+inp
    c.set_function(call)
    return c

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    ex = np.exp(x)
    return ex / np.sum(ex, axis=0)


import matplotlib.pyplot as plt
class plotter:
    def __init__(self):
        plt.ion()

        self.x = []
        self.y = []
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)

    def pushy(self,y):
        self.y.append(y)
        if len(self.x)>0:
            self.x.append(self.x[-1]+1)
        else:
            self.x.append(0)

    def show(self):
        self.ax.clear()
        self.ax.plot(self.x,self.y)
        plt.draw()

class nnagent(object):
    def __init__(self,
    observation_space,
    action_space,
    stack_factor=1,
    discount_factor=.99, # gamma
    train_skip_every=1,
    ):
        self.rpm = rpm(1000000) # 1M history
        self.plotter = plotter()
        self.render = True
        self.training = True
        self.noise_source = one_fsq_noise()
        self.train_counter = 0
        self.train_skip_every = train_skip_every
        self.observation_stack_factor = stack_factor

        self.inputdims = observation_space.shape[0] * self.observation_stack_factor
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

    # a = actor(s) : predict actions given state
    def create_actor_network(self,inputdims,outputdims):
        # add gaussian noise.
        c = Can()
        c.add(Dense(inputdims,128))
        c.add(Act('lrelu'))
        c.add(Dense(128,128))
        c.add(Act('lrelu'))
        c.add(Dense(128,64))
        c.add(Act('elu'))
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
        concat = Lambda(lambda x:tf.concat([x[0],x[1]],axis=1))
        # concat state and action
        den1 = c.add(Dense(inputdims,256))
        den2 = c.add(Dense(256+actiondims,256))
        den3 = c.add(Dense(256, 64))
        den4 = c.add(Dense(64,1))

        def call(i):
            state = i[0]
            action = i[1]
            h1 = den1(state)
            h1 = Act('lrelu')(h1)
            h2 = den2(concat([h1,action]))
            h2 = Act('lrelu')(h2)

            h2 = den3(h2)
            h2 = Act('elu')(h2)

            q = den4(h2)
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
        opt_critic = tf.train.AdamOptimizer(3e-4)
        # opt_actor = tf.train.MomentumOptimizer(1e-1,momentum=0.9)
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
        total_size = batch_size * self.train_skip_every
        epochs = 1

        self.train_counter+=1
        self.train_counter %= self.train_skip_every

        if self.train_counter != 0: # train every few steps
            return

        if memory.size() > total_size * 64:
            #if enough samples in memory
            for i in range(self.train_skip_every):
                # sample randomly a minibatch from memory
                [s1,a1,r1,isdone,s2] = memory.sample_batch(batch_size)
                # print(s1.shape,a1.shape,r1.shape,isdone.shape,s2.shape)

                self.feed([s1,a1,r1,isdone,s2])

    def feed_one(self,tup):
        self.rpm.add(tup)

    # gymnastics
    def play(self,env,max_steps=-1,realtime=False,noise_level=0.): # play 1 episode
        timer = time.time()
        max_steps = max_steps if max_steps > 0 else 50000
        steps = 0
        total_reward = 0

        # stack a little history to ensure markov property
        # LSTM will definitely be used here in the future...
        # global que # python 2 quirk
        self.que = np.zeros((self.inputdims,),dtype='float32') # list of recent history actions

        if self.observation_stack_factor>1:
            def quein(observation):
                # global que # python 2 quirk
                length = len(observation)
                self.que[0:-length] = self.que[length:] # left shift
                self.que[-length:] = np.array(observation)

            def quecopy():
                return self.que.copy()
        else:
            def quein(observation): self.que = np.array(observation)
            def quecopy():return self.que.view()

        # what the agent see as state is a stack of history observations.

        observation = env.reset(difficulty=0)
        quein(observation) # quein o1

        while True and steps <= max_steps:
            steps +=1

            thisque = quecopy() # s1

            action = self.act(thisque) # a1

            if self.is_continuous:
                # add noise to our actions, since our policy by nature is deterministic
                exploration_noise = self.noise_source.one((self.outputdims,),noise_level)
                exploration_noise *= self.action_multiplier
                # print(exploration_noise,exploration_noise.shape)
                action += exploration_noise
                action = self.clamper(action)
                action_out = action
            else:
                exploration_noise = self.noise_source.one((self.outputdims,),noise_level)
                exploration_noise *= self.action_multiplier
                action += exploration_noise
                # action = self.clamper(action)
                action = softmax(action)
                # discretize our actions
                probabilities = action
                csprob = np.cumsum(probabilities)
                action_index = (csprob > np.random.rand()).argmax()
                action_out = action_index

            # o2, r1,
            observation, reward, done, _info = env.step(action_out)

            # d1
            isdone = 1 if done else 0
            total_reward += reward

            quein(observation) # quein o2
            nextque = quecopy() # s2

            # feed into replay memory
            if self.training == True:
                self.feed_one((thisque,action,reward,isdone,nextque)) # s1,a1,r1,isdone,s2

            if self.render==True and (steps%30==0 or realtime==True):
                env.render()
            if done :
                break

            verbose= 2 if steps==1 else 0

            if self.training == True:
                self.train(verbose=verbose)

        # print('episode done in',steps,'steps',time.time()-timer,'second total reward',total_reward)
        totaltime = time.time()-timer
        print('episode done in {} steps in {:.2f} sec, {:.4f} sec/step, got reward :{:.2f}'.format(
        steps,totaltime,totaltime/steps,total_reward
        ))

        self.plotter.pushy(total_reward)
        self.plotter.show()
        return

    # one step of action, given observation
    def act(self,observation):
        actor,critic = self.actor,self.critic
        obs = np.reshape(observation,(1,len(observation)))

        # actions = actor.infer(obs)
        # q = critic.infer([obs,actions])[0]
        [actions,q] = self.joint_inference(obs)
        actions,q = actions[0],q[0]

        disp_actions = (actions-self.action_bias) / self.action_multiplier
        disp_actions = disp_actions * 5 + np.arange(self.outputdims) * 12.0 + 30

        noise = self.noise_source.ask() * 5 - np.arange(self.outputdims) * 12.0 - 30

        self.loggraph(np.hstack([disp_actions,noise,q]))
        # temporarily disabled.
        return actions

    def loggraph(self,waves):
        if not hasattr(self,'wavegraph'):
            def rn():
                r = np.random.uniform()
                return 0.2+r*0.4
            colors = []
            for i in range(len(waves)-1):
                color = [rn(),rn(),rn()]
                colors.append(color)
            colors.append([0.2,0.5,0.9])
            self.wavegraph = wavegraph(len(waves),'actions/noises/Q',np.array(colors))

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

class playground(object):
    def __init__(self,envname):
        self.envname=envname
        env = gym.make(envname)
        self.env = env

        self.monpath = './experiment-'+self.envname

    def wrap(self):
        from gym import wrappers
        self.env = wrappers.Monitor(self.env,self.monpath,force=True)

    def up(self):
        self.env.close()
        gym.upload(self.monpath, api_key='sk_ge0PoVXsS6C5ojZ9amTkSA')

from osim.env import RunEnv

if __name__=='__main__':
    # p = playground('LunarLanderContinuous-v2')
    # p = playground('Pendulum-v0')
    # p = playground('MountainCar-v0')BipedalWalker-v2
    # p = playground('BipedalWalker-v2')
    # e = p.env

    e = RunEnv(visualize=False)

    agent = nnagent(
    e.observation_space,
    e.action_space,
    discount_factor=.99,
    stack_factor=1,
    train_skip_every=1,
    )

    noise_level = 2.
    def r(ep):
        global noise_level
        # agent.render = True
        # e = p.env
        for i in range(ep):
            noise_level *= .99
            noise_level = max(3e-6, noise_level)
            print('ep',i,'/',ep,'noise_level',noise_level)
            agent.play(e,realtime=False,max_steps=-1,noise_level=noise_level)

    def test():
        # e = p.env
        agent.render = True
        agent.play(e,realtime=True,max_steps=-1,noise_level=1e-11)
