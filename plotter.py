# interactive plotter that runs in a separate process.

import time, os
from ipc import ipc

def remote_plotter_callback(conn):
    import matplotlib
    import time
    import threading as th

    from sys import platform as _platform

    if _platform == "darwin":
        # MAC OS X
        matplotlib.use('qt5Agg')
        # avoid using cocoa or TK backend, since they require using a framework build.
        # conda install pyqt

    import matplotlib.pyplot as plt

    class plotter:
        def __init__(self,num_lines=1):
            self.lock = th.Lock()
            self.x = []
            self.y = []
            self.num_lines = num_lines
            self.ys = [[] for i in range(num_lines)]

            self.colors = [
                [
                    (i*i*7.9+i*19/2.3+17/3.1)%0.5+0.2,
                    (i*i*9.1+i*23/2.9+31/3.7)%0.5+0.2,
                    (i*i*11.3+i*29/3.1+37/4.1)%0.5+0.2,
                ]
                for i in range(num_lines)]

            self.time = time.time()

            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

            plt.show(block=False)
            # plt.show()

            self.something_new = True

        def custom_pause(self,interval):
            # per https://stackoverflow.com/a/44352761
            # and https://stackoverflow.com/a/45734500
            # June 2017 sucks!
            # calling plot() or pause() with newer matplotlib versions always focus the plot to foreground. to prevent this from happening, you have to use alternative methods to update the plot.

            # also per https://github.com/matplotlib/matplotlib/pull/9061/commits/1f083e45fa5c022112967d2bfd966f073ffb42b0
            if self.fig.canvas.figure.stale:
                self.fig.canvas.draw()
            self.fig.canvas.start_event_loop(interval)

        def show(self):
            self.lock.acquire()
            if self.anything_new():
                self.ax.clear()
                self.ax.grid(color='#f0f0f0', linestyle='solid', linewidth=1)

                x = self.x

                for idx in range(len(self.ys)):
                    y = self.ys[idx]
                    c = self.colors[idx]
                    self.ax.plot(x,y,color=tuple(c))

                for idx in range(len(self.ys)):
                    y = self.ys[idx]
                    c = self.colors[idx]
                    init = 5
                    if len(y)>init:
                        ysmooth = [sum(y[0:init])/init]*init
                        for i in range(init,len(y)): # first order
                            ysmooth.append(ysmooth[-1]*0.9+y[i]*0.1)
                        for i in range(init,len(y)): # second order
                            ysmooth[i] = ysmooth[i-1]*0.9+ysmooth[i]*0.1

                        self.ax.plot(x,ysmooth,lw=2,color=tuple([cp**0.3 for cp in c]),alpha=0.5)

            self.lock.release()
            # plt.pause(0.2)
            # plt.draw()
            self.custom_pause(0.2)

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
