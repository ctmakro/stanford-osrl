from multiprocessing import Process, Pipe

# FAST ENV

# this is a environment wrapper. it wraps the RunEnv and provide interface similar to it. The wrapper do a lot of pre and post processing (to make the RunEnv more trainable), so we don't have to do them in the main program.

from observation_processor import generate_observation as go
import numpy as np

class fastenv:
    def __init__(self,e,skipcount):
        self.e = e
        self.stepcount = 0

        self.old_observation = None
        self.skipcount = skipcount # 4

    def obg(self,plain_obs):
        # observation generator
        # derivatives of observations extracted here.
        processed_observation, self.old_observation = go(plain_obs, self.old_observation, step=self.stepcount)
        return np.array(processed_observation)

    def step(self,action):
        action = [float(action[i]) for i in range(len(action))]

        import math
        for num in action:
            if math.isnan(num):
                print('NaN met',action)
                raise RuntimeError('this is bullshit')

        sr = 0
        for j in range(self.skipcount):
            self.stepcount+=1
            oo,r,d,i = self.e.step(action)
            o = self.obg(oo)
            sr += r

            if d == True:
                break

        # # alternative reward scheme
        # delta_x = oo[1] - self.lastx
        # sr = delta_x * 1
        # self.lastx = oo[1]

        return o,sr,d,i

    def reset(self):
        self.stepcount=0
        self.old_observation = None

        oo = self.e.reset()
        # o = self.e.reset(difficulty=2)
        self.lastx = oo[1]
        o = self.obg(oo)
        return o


def standalone(conn,visualize=True):
    from osim.env import RunEnv
    re = RunEnv(visualize=visualize)
    e = fastenv(re,4)

    while True:
        msg = conn.recv()

        # messages should be tuples,
        # msg[0] should be string

        if msg[0] == 'reset':
            obs = e.reset()
            conn.send(obs)
        elif msg[0] == 'step':
            four = e.step(msg[1])
            conn.send(four)
        else:
            conn.close()
            del e
            return

class ei: # Environment Instance
    def __init__(self,visualize=True):
        self.pc, self.cc = Pipe()
        self.p = Process(target = standalone, args=(self.cc, visualize), daemon=True)
        self.p.start()

        self.occupied = False

    def reset(self):
        self.pc.send(('reset',))
        return self.pc.recv()

    def step(self,actions):
        self.pc.send(('step',actions,))
        return self.pc.recv()

    def __del__(self):
        self.pc.send(('exit',))
        print('(ei)waiting for join...')
        self.p.join()

class eipool: # Environment Instance Pool
    def __init__(self,n=1,showfirst=True):
        import threading as th
        self.pool = [ei(visualize=(True if i==0 and showfirst else False)) for i in range(n)]
        self.lock = th.Lock()

    def acq_env(self):
        self.lock.acquire()
        for e in self.pool:
            if e.occupied == False:
                e.occupied = True # occupied
                self.lock.release()
                return e # return the envinstance

        self.lock.release()
        return False # no available ei

    def rel_env(self,ei):
        self.lock.acquire()
        for e in self.pool:
            if e == ei:
                e.occupied = False # freed
        self.lock.release()

    def num_free(self):
        return sum([0 if e.occupied else 1 for e in self.pool])

    def num_total(self):
        return len(self.pool)

    def all_free(self):
        return self.num_free()==self.num_total()

    def __del__(self):
        for e in self.pool:
            del e

if __name__ == '__main__':
    from osim.env import RunEnv
    grande = RunEnv(visualize=False)

    ep = eipool(5)

    def run():
        env = ep.acq_env()
        if env ==False:
            print('shi!!!!')
            return

        observation = env.reset()
        for i in range(500):
            observation, reward, done, info = env.step(grande.action_space.sample())
            # print(observation)
            if done:
                break;

        ep.rel_env(env)

    def para():
        import threading as th
        ts = [th.Thread(target=run,daemon=True) for i in range(4)]
        for i in ts:
            i.start()
        for i in ts:
            i.join()

    para()
    para()
    del ep
