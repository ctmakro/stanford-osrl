import time, os
from ipc import ipc

def remote_plotter_callback(conn):
    import matplotlib.pyplot as plt
    import time
    import threading as th

    class plotter:
        def __init__(self,num_lines=1):
            self.lock = th.Lock()
            self.x = []
            self.y = []
            self.num_lines = num_lines
            self.ys = [[] for i in range(num_lines)]

            self.time = time.time()

            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

            plt.show(block=False)
            # plt.show()

            self.something_new = True

        def show(self):
            self.lock.acquire()
            if self.anything_new():
                self.ax.clear()
                for y in self.ys:
                    self.ax.plot(self.x,y)
            self.lock.release()
            plt.pause(0.2)
            # plt.draw()

        def pushy(self,y):
            self.lock.acquire()
            self.y.append(y)
            if len(self.x)>0:
                self.x.append(self.x[-1]+1)
            else:
                self.x.append(0)
            self.something_new = True
            self.lock.release()

        def pushys(self,ys):
            self.lock.acquire()
            for idx in range(self.num_lines):
                self.ys[idx].append(ys[idx])

            if len(self.x)>0:
                self.x.append(self.x[-1]+1)
            else:
                self.x.append(0)
            self.something_new = True
            self.lock.release()

        def anything_new(self):
            # self.lock.acquire()
            n = self.something_new
            self.something_new = False
            # self.lock.release()
            return n

    p = None
    endflag = False

    # wait for init parameters
    while 1:
        msg = conn.recv()
        if p is None:
            if msg[0] == 'init':
                p = plotter(msg[1])
                break

    def receive_loop():
        while 1:
            msg = conn.recv()
            if msg[0] == 'pushys':
                p.pushys(msg[1])
            else:
                return

    th.Thread(target = receive_loop, daemon = True).start()

    while 1:
        p.show()

class interprocess_plotter(ipc):
    def __init__(self,num_lines=1):
        super().__init__(remote_plotter_callback)
        self.send(('init',num_lines))

    def pushys(self,ys):
        self.send(('pushys', ys))

if __name__=='__main__':
    ip = interprocess_plotter(2)
    import math,time
    for i in range(100):
        ip.pushys([math.sin(i/10), math.sin(i/10+2)])
        time.sleep(0.05)

    time.sleep(5)
