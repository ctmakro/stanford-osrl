import multiprocessing as mp
# from multiprocessing import Process, Queue
import time, os

class pretty:
    def pretty(self,s):
        print('({}) {}'.format(self.__class__.__name__, s))

# connection object built on two Queue()s
class conn(pretty):
    def __init__(self, sq, rq): # rq, sq => receive queue, send queue
        self.rq = rq
        self.sq = sq
    def recv(self):
        return self.rq.get()
    def send(self,x):
        return self.sq.put(x)

class conn_slave(conn):
    def recv(self):
        msg = super().recv()
        if msg[0] == 'msg':
            return msg[1]
        else:
            # suicide
            self.pretty('received msg other than \'msg\', quitting')
            import os
            os._exit(1)
            time.sleep(0.1)
            return

class conn_master(conn):
    def send(self,x):
        return super().send(('msg',x))
    def kill_slave(self):
        return super().send(('kill',))

class ipc(pretty): # base for all interprocess communicating classes
    def __init__(self, f):
        ctx = mp.get_context('spawn') # eliminate problems with fork().
        pq,cq = ctx.Queue(1), ctx.Queue(1)
        self.pc, self.cc = conn_master(pq, cq), conn_slave(cq,pq)

        self.p = ctx.Process(target=f, args=(self.cc,), daemon=True)
        self.pretty('starting process')
        self.p.start()

    def send(self,x):
        return self.pc.send(x)
    def recv(self):
        return self.pc.recv()

    def __del__(self):
        self.pc.kill_slave()
        self.pretty('waiting for join()')
        self.p.join()
