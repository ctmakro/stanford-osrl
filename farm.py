# farm.py
# a single instance of a farm.

# a farm should consist of a pool of instances
# and expose those instances as one giant callable class

import multiprocessing
from multiprocessing import Process, Pipe
from osim.env import RunEnv

ncpu = multiprocessing.cpu_count()

# separate process that holds a separate RunEnv instance.
# This has to be done since RunEnv() in the same process result in interleaved running of simulations.
def standalone_headless_isolated(conn):
    from osim.env import RunEnv
    e = RunEnv(visualize=False)

    while True:
        msg = conn.recv()

        # messages should be tuples,
        # msg[0] should be string

        if msg[0] == 'reset':
            o = e.reset(difficulty=2)
            conn.send(o)
        elif msg[0] == 'step':
            ordi = e.step(msg[1])
            conn.send(ordi)
        else:
            conn.close()
            del e
            return

eid = 0
# class that manages the interprocess communication and expose itself as a RunEnv.
class ei: # Environment Instance
    def __init__(self):
        self.pc, self.cc = Pipe()
        self.p = Process(
            target = standalone_headless_isolated,
            args=(self.cc,)
        )
        self.p.daemon = True
        self.p.start()
        self.occupied = False

        global eid
        self.id = eid
        eid+=1
        print('(ei)instance created, id '+str(self.id))

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

# class that other classes acquires and releases EIs from.
class eipool: # Environment Instance Pool
    def __init__(self,n=1):
        import threading as th
        print('(eipool)starting '+str(ncpu)+' instance(s)...')
        self.pool = [ei() for i in range(n)]
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

    def get_env_by_id(self,id):
        for e in self.pool:
            if e.id == id:
                return e
        return False

    def __del__(self):
        for e in self.pool:
            del e

# farm
# interface with eipool via eids.
class farm:
    def __init__(self):
        # on init, create a pool
        # self.renew()
        pass

    def acq(self,n=None):
        self.renew_if_needed(n)
        result = self.eip.acq_env()
        if result == False:
            return False
        else:
            print('(farm) acq '+str(result.id))
            return result.id

    def rel(self,id):
        e = self.eip.get_env_by_id(id)
        self.eip.rel_env(e)
        print('(farm) rel '+str(id))

    def step(self,id,actions):
        e = self.eip.get_env_by_id(id)
        if e == False: return e

        return e.step(actions)

    def reset(self,id):
        e = self.eip.get_env_by_id(id)
        if e == False: return e

        return e.reset()

    def renew_if_needed(self,n=None):
        if not hasattr(self,'eip'):
            print('(farm) renew because no eipool present')
            self.renew(n)

    # recreate the pool
    def renew(self,n=None):
        global ncpu
        print('(farm) natural pool renew')

        if hasattr(self,'eip'): # if eip exists
            while not self.eip.all_free(): # wait until all free
                print('(farm) wait until all of self.eip free..')
                time.sleep(0.5)
            del self.eip

        self.eip = eipool(ncpu if n is None else n)

    def forcerenew(self,n=None):
        print('(farm) forced pool renew')

        if hasattr(self,'eip'): # if eip exists
            del self.eip
        self.eip = eipool(ncpu if n is None else n)


# expose the farm via Pyro4
def main():
    from pyro_helper import pyro_expose
    pyro_expose(farm,20099,'farm')

if __name__ == '__main__':
    main()
