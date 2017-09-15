# Proximal Policy Optimization algorithm
# implemented by Qin Yongliang
# rewritten with openai/baselines as a reference

# this is intended to be run on py3.5

# terminology
# ob, obs, input, observation, state, s1 : observation in RL
# ac, action, a1, output : action in RL
# vp, vf, val_pred, value_prediction : value function predicted by critic
# reward, return, ret, r1, tdlamret : (per step / expected future) reward in RL
# adv, advantage, adv_target, atarg : advantage (estimated / target) in RL

import tensorflow as tf
import numpy as np
from canton import *
import gym

# low-passed gaussian noise to help with exploration.
# from gaussian import lowpassgaussian as lpgs

# To improve our policy via PPO, we must be able to parametrize and sample from it as a probabilistic distribution. A typical choice for continuous domain problems is the Diagonal Gaussian Distribution. It's simpler (and thus less powerful) than a full Multivariate Gaussian, but should work just fine.

# Knowledge of probabilistics is required to read and comprehend following code.

tsq = lambda x:tf.square(x)
tsum = lambda x:tf.reduce_sum(x)
tmean = lambda x:tf.reduce_mean(x)
tsumlast = lambda x:tf.reduce_sum(x, axis=[-1])

class Dist(Can):
    def logp(self, a):
        return - self.neglogp(a)

# Bernoulli layer
class Bernoulli(Dist):
    def __call__(self, x):
        self.logits = x
        self.probs = Act('sigmoid')(self.logits)
        self.mean = self.probs
        return self.sample()
    def sample(self):
        return tf.to_float(tf.random_uniform(tf.shape(self.probs)) < self.probs)

    def neglogp(self, a): # action be either of [0,1]
        # a.shape [num, dims]
        # def loge(i):
        #     eps = 1e-13
        #     return tf.log(i+eps)
        # logp_each = loge(self.probs) * a + loge(1.-self.probs)*(1.-a)
        # logp_total = tsumlast(logp_each)
        # return logp_total # shape [num]

        return tsumlast(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.to_float(a)))

class Categorical(Dist):
    def __call__(self,x):
        self.logits = x # [batch, categories]
        return self.sample()

    def sample(self):
        self.mean = tf.argmax(self.logits, axis=-1) # [batch, ]
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)
        # returns a number indicating category

    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        one_hot_actions = tf.one_hot(x, tf.shape(self.logits)[1])
        return tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=one_hot_actions)

class MultiCategorical(Dist):
    def __init__(self, dims, cats):
        super().__init__()
        self.dims = dims # how many categoricals
        self.cats = cats # how many category within one categorical
        self.categoricals = [Categorical() for _ in range(dims)]

    def __call__(self,x): #[batch, dim*categories]
        self.multilogits = x
        listlogits = tf.split(self.multilogits,num_or_size_splits=self.dims, axis=1)
        for c,l in zip(self.categoricals,listlogits):
            c(l) # feed every categorical
        return self.sample()

    def sample(self):
        self.mean = tf.cast(tf.stack([p.mean for p in self.categoricals],axis=-1), tf.int32)
        return tf.cast(tf.stack([p.sample() for p in self.categoricals], axis=-1), tf.int32)

    def neglogp(self, x): #[batch, dims]
        return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x, axis=1))])

class MultiCategoricalContinuous(MultiCategorical):
    def __init__(self,*args):
        super().__init__(*args)
        self.scaler = 1/(self.cats-1)

    def sample(self):
        ret = super().sample()
        ret = tf.to_float(ret) * self.scaler
        self.mean = tf.to_float(self.mean) * self.scaler
        return ret

    def neglogp(self, x):
        x = tf.cast(x/self.scaler+1e-8, tf.int32) # integerization
        return super().neglogp(x)

# a simple MLP policy.
class Policy(Can):
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
        c.add(Dense(64, ac_dims*10, stddev=1))
        # self.dist = c.add(Bernoulli())
        self.dist = c.add(MultiCategoricalContinuous(ac_dims, 10))
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

# don't discard trajectories after one iteration;
# keep them around in the buffer to increase sample efficiency.
class traj_buffer:
    def __init__(self, length):
        self.maxlen = length
        self.buf = []

    def push(self, collected):
        # collected is a tuple of (s1,a1...)
        self.buf.append(collected)
        while len(self.buf)>self.maxlen:
            self.buf.pop(0) # remove first

    def get_all(self):
        collected = [[] for i in range(len(self.buf[0]))]
        for c in self.buf:
            for i in range(len(c)):
                collected[i] += c[i]
        return collected

    def get_all_raw(self):
        return list(self.buf)

# our PPO agent.
class ppo_agent:
    def __init__(
        self, ob_space, ac_space,
        horizon=2048,
        gamma=0.99, lam=0.95,
        train_epochs=10, batch_size=64,
        buffer_length=10,
        policy=None
        ):
        if policy is None:
            policy = Policy
        self.current_policy = policy(ob_space, ac_space)
        self.old_policy = policy(ob_space, ac_space)
        self.current_policy.actor.summary()
        self.current_policy.critic.summary()

        self.gamma, self.lam, self.horizon = gamma, lam, horizon
        self.train_epochs, self.batch_size = train_epochs, batch_size
        self.traj_buffer = traj_buffer(buffer_length)

        self.act, self.predict_value, self.train_for_one_step, self.assign_old_eq_new = self.build_functions()

        low, high = ac_space.low, ac_space.high
        self.action_bias = (high + low)/2.
        self.action_multiplier = high - self.action_bias

        # limit action into the range specified by environment.
        def action_limiter(action): # assume input mean 0 std 1
            return np.tanh(action) * self.action_multiplier + self.action_bias
        def action_limiter(action): # assume input uniform [0,1]
            return (action * 2 - 1) * self.action_multiplier + self.action_bias
        self.action_limiter = action_limiter

        # logging of episodic reward.
        from plotter import interprocess_plotter as plotter
        self.plotter = plotter(2)

        # logging of actions. comment out if you don't have opencv
        if not hasattr(self,'wavegraph'):
            from winfrey import wavegraph
            # num_waves = self.outputdims*2+1
            num_waves = self.current_policy.ac_dims*2+1
            def rn():
                r = np.random.uniform()
                return 0.3+r*0.4
            colors = []
            for i in range(num_waves-1):
                color = [rn(),rn(),rn()]
                colors.append(color)
            colors.append([0.2,0.5,0.9])
            self.wavegraph = wavegraph(num_waves,'ac_mean/ac_sto/vf',np.array(colors))

            def loggraph(waves):
                wg = self.wavegraph
                wg.one(waves.reshape((-1,)))

            self.loggraph = loggraph

    # build graph and actions for training with tensorflow.
    def build_functions(self):
        # the 'lrmult' parameter is not implemented.

        # improve policy w.r.t. old_policy
        policy, old_policy = self.current_policy, self.old_policy

        # Input Placeholders
        states, actions = ph([policy.ob_dims]), ph([policy.ac_dims]) # you know these two
        adv_target = ph([1]) # Target advantage function, estimated
        ret = ph([1]) # Empirical return, estimated

        # feed observation thru the networks
        # policy_mean, policy_std = policy.actor(states)
        policy_sample = policy.actor(states)
        policy_val_pred = policy.critic(states)

        # old_policy_mean, old_policy_std = old_policy.actor(states)
        old_policy_sample = old_policy.actor(states)
        old_policy_val_pred = old_policy.critic(states)

        # ratio = P_now(state, action) / P_old(state, action)
        # state was previously fed so we will pass in actions only
        # # IMPORTANT: add epsilon to guarantee numerical stability
        # eps = 1e-8
        # pn,po = policy.dg.p(actions)+eps, old_policy.dg.p(actions)+eps
        # ratio = pn/po
        # ratio = tf.reduce_prod(ratio, axis=1) #temp
        logpn,logpo = policy.dist.logp(actions), old_policy.dist.logp(actions)
        ratio = tf.exp(logpn - logpo)

        # surr1 -> policy gradient
        surr1 = ratio * adv_target

        # surr2 -> policy deviation
        clip_param = 0.2 # magical epsilon in paper
        surr2 = tf.clip_by_value(ratio, 1.0-clip_param, 1.0+clip_param) * adv_target

        # together they form the L^CLIP loss in PPO paper.
        policy_surrogate = - tf.reduce_mean(tf.minimum(surr1,surr2))

        # how far is our critic's prediction from estimated return?
        value_prediction = policy_val_pred
        value_loss = tf.reduce_mean(tsq(value_prediction-ret))

        # optimizer
        opt = tf.train.AdamOptimizer(1e-4)
        opt_a = tf.train.AdamOptimizer(3e-4)
        opt_c = tf.train.AdamOptimizer(1e-3)

        # sum of two losses used in original implementation
        total_loss = policy_surrogate + value_loss
        combined_trainstep = opt.minimize(total_loss, var_list=policy.get_weights())

        # If you want different learning rates, go with the following
        actor_trainstep = opt_a.minimize(policy_surrogate, var_list=policy.actor.get_weights())
        critic_trainstep = opt_c.minimize(value_loss, var_list=policy.critic.get_weights())

        # # gradient clipping test
        # grads_vars = opt_a.compute_gradients(policy_surrogate, policy.actor.get_weights())
        # capped = [(tf.clip_by_value(grad, -1., 1.), var) for grad,var in grads_vars]
        # actor_trainstep = opt_a.apply_gradients(capped)

        # # weight decay if needed
        # decay_factor = 1e-5
        # decay_step = [tf.assign(w, w * (1-decay_factor)) for w in policy.actor.get_only_weights()]

        def insane(k):
            for x in k:
                s = np.sum(x)
                if np.isnan(s) or np.isinf(s):
                    print(k)
                    raise Exception('inf/nan')

        # 1. build our action sampler: given observation, generate action
        def act(state, stochastic=True):
            # assume state is ndarray of shape [dims]
            state = state.view()
            state.shape = (1,) + state.shape
            insane(state)
            res = get_session().run([
                # policy_mean,
                # policy_std,
                # policy_out,
                policy.dist.mean,
                policy_sample,
                value_prediction,
            ], feed_dict={states: state})

            pm, ps, vp = res
            # po, vp = res
            # [batch, dims] [batch, dims] [batch, 1]

            pm,ps,vp = pm[0], ps[0], vp[0,0]
            return pm,ps,vp
            # po, vp = po[0], vp[0,0]
            # return po,vp

        # 2. value prediction
        def predict_value(_states):
            # assume _states is ndarray of shape [batch, dims]
            res = get_session().run([value_prediction],feed_dict={states:_states})
            return res[0]

        # 3. trainer. update current policy given processed trajectories.
        def train_for_one_step(_states, _actions, _adv_target, _ret):
            # insane([_states,_actions,_adv_target,_ret])
            res = get_session().run(
                [ # perform training and collect losses in one go
                    policy_surrogate, value_loss,
                    # ratio,logpn,logpo,
                    actor_trainstep, critic_trainstep,
                    # combined_trainstep,
                    # decay_step,
                ],
                feed_dict = {
                    states:_states, actions:_actions,
                    adv_target:_adv_target, ret:_ret,
                }
            )
            # insane([(x if x is not None else 0) for x in res])
            # res[0] is ploss, res[1] is val_loss
            return res

        # 4. assigner. assign old_policy's weights with current policy's weights.
        assign_ops = [tf.assign(o,n) for o,n in zip(old_policy.get_weights(), policy.get_weights())]
        print('total of {} weights to assign from new to old'.format(len(assign_ops)))
        def assign_old_eq_new():
            get_session().run([assign_ops])

        return act, predict_value, train_for_one_step, assign_old_eq_new

    # run a bunch of episodes with current_policy on env and collect some trajectories.
    def collect_trajectories(self, env):
        policy = self.current_policy
        # minimum length we are going to collect
        horizon = self.horizon
        print('collecting trajectory...')

        # things we have to collect
        s1 = [] # observations before action
        a1 = [] # action taken
        r1 = [] # reward received
        _done = [] # is the episode done after a1

        # counters
        ep = 0
        steps = 0
        sum_reward = 0

        while 1:
            episode_total_reward = 0
            episode_length = 0

            # from gaussian import lowpassuniform
            # noises = [lowpassuniform() for _ in range(self.current_policy.ac_dims)]
            # def noise_sample():
            #     return np.array([n.sample() for n in noises])
            #     # return np.random.uniform(size=(self.current_policy.ac_dims,))

            # initial observation
            ob = env.reset()
            while 1:
                # sample action from given policy
                mean, sto, val_pred = self.act(ob)
                # policy_out, val_pred = self.act(ob)
                # sto_action = 1.0*(policy_out > noise_sample())
                sto_action = sto
                # sto_action = noise_sample() * std + mean
                # sto_limited = self.action_limiter(sto_action)
                mean_limited, sto_limited = self.action_limiter(mean), self.action_limiter(sto_action)

                # logging actions. comment out if you don't have opencv
                if True:
                    # mean_limited = self.action_limiter(mean)
                    disp_mean = mean_limited*5. + np.arange(policy.ac_dims)*12 + 30
                    disp_sto = sto_limited*5. - np.flipud(np.arange(policy.ac_dims))*12 - 30
                    self.loggraph(np.hstack([disp_mean, disp_sto, val_pred]))

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
                    _done[-1] = 1
                    print('episode {} done in {} steps, total reward:{}'.format(
                        ep+1, episode_length, episode_total_reward,
                    ))
                    self.plotter.pushys([ep,episode_total_reward])
                    break

            sum_reward += episode_total_reward
            print('{}/{} steps collected in {} episode(s)'.format(steps,horizon,ep+1), end='\r')
            if steps>= horizon:
                break
            else:
                ep+=1

        print('mean reward per episode:{}'.format(sum_reward/(ep+1)))
        return s1,a1,r1,_done

    # estimate target value (which we are trying to make our critic to fit) via TD(lambda), and advantage using GAE(lambda), from collected trajectories.
    def append_vtarg_and_adv(self, collected):
        # you know what these mean, don't you?
        gamma = self.gamma # 0.99
        lam = self.lam # 0.95

        [s1,a1,r1,done] = collected
        vp1 = self.predict_value(np.array(s1))

        T = len(a1) # [s1] might be longer than the others by one.
        # since to predict vp1 for T+1 step, we might append last s2 into [s1]
        advantage = [None]*T

        last_adv = 0
        for t in reversed(range(T)): # step T-1, T-2 ... 0
            # delta = (reward_now) + (predicted_future_t+1) - (predicted_future_t)
            delta = r1[t] + (0 if done[t] else gamma * vp1[t+1]) - vp1[t]

            advantage[t] = delta + gamma * lam * (1-done[t]) * last_adv
            last_adv = advantage[t]

        if len(s1) > len(a1):
            s1 = list(s1) # create a new list without intefereing original
            s1.pop() # remove last

        tdlamret = [a+v for a,v in zip(advantage, vp1)]
        return [s1,a1,r1,done, advantage,tdlamret]

    # perform one policy iteration
    def iterate_once(self, env):

        # 0. assign new to old
        self.assign_old_eq_new()

        # 1. collect trajectories w/ current policy
        collected = self.collect_trajectories(env)

        # 2. push the trajectories into buffer
        self.traj_buffer.push(collected)

        # 3. load historic trajectories from buffer
        collected = self.traj_buffer.get_all()

        # 4. estimate advantage and TD(lambda) return
        collected = self.append_vtarg_and_adv(collected)

        # 5. processing, numpyization
        collected = self.usual_data_processing(collected)
        # s1,a1,r1,done,advantage,tdlamret = self.usual_data_processing(collected)

        # 6. train for some epochs
        self.usual_feed_training(collected)

        print('iteration done.')

    # data processing moved here
    def usual_data_processing(self,collected):
        s1,a1,r1,done,advantage,tdlamret = collected

        l = len(s1)
        assert l==len(a1)and l==len(r1)and l==len(done)and l==len(advantage)and l==len(tdlamret) # all inputs must be of the same length

        # 5. data processing
        # shuffling
        indices = np.arange(len(a1))
        np.random.shuffle(indices)

        # numpyization
        s1, a1, advantage, tdlamret = [
            np.take(np.array(k).astype('float32'), indices, axis=0)
            for k in [s1, a1, advantage, tdlamret]
        ]

        # expand dimension for minibatch training
        for nd in [s1,a1,advantage,tdlamret]:
            if nd.ndim == 1:
                nd.shape += (1,)

        # standarize/normalize
        advantage = (advantage - advantage.mean())/(advantage.std()+1e-3)

        return s1,a1,r1,done,advantage,tdlamret

    # feed-train moved here
    def usual_feed_training(self,collected):
        s1,a1,r1,done,advantage,tdlamret = collected
        # 6. train for some epochs
        train_epochs = self.train_epochs
        batch_size = self.batch_size
        data_length = len(s1)
        import time
        lasttimestamp = time.time()

        print('training network on {} datapoints for {} epochs'.format(data_length,train_epochs))
        for e in range(train_epochs):
            for j in range(0, data_length-batch_size+1, batch_size):
                # ignore tail
                res = self.train_for_one_step(
                    s1[j:j+batch_size],
                    a1[j:j+batch_size],
                    advantage[j:j+batch_size],
                    tdlamret[j:j+batch_size],
                )
                ploss, vloss = res[0],res[1]
                if time.time() - lasttimestamp > 0.2:
                    lasttimestamp = time.time()
                    print(' '*30, 'epoch: {}/{} ploss: {:6.4f} vloss: {:6.4f}'.format(
                        e+1,train_epochs,ploss, vloss),end='\r')

if __name__ == '__main__':
    # get swingy
    envname = 'Pendulum-v0'
    envname = 'BipedalWalker-v2'
    env = gym.make(envname)

    agent = ppo_agent(
        env.observation_space, env.action_space,
        horizon=1024, # minimum steps to collect before policy update
        gamma=0.99, # discount factor for reward
        lam=0.95, # smooth factor for advantage estimation
        train_epochs=10, # how many epoch over data for one update
        batch_size=64, # batch size for training
        buffer_length=16, # how may iteration of trajectories to keep around for training. Set to 1 for 'original' PPO (higher variance).
    )

    get_session().run(gvi()) # init global variables for TF

    def r(iters=2):
        print('start running')
        for i in range(iters):
            print('optimization iteration {}/{}'.format(i+1, iters))
            agent.iterate_once(env)
    r(1000)
