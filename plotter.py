from multiprocessing import Process, Pipe

def remote_plotter(conn):
    import matplotlib.pyplot as plt
    import time
    import threading as th

    class plotter:
        def __init__(self):
            self.lock = th.Lock()
            self.x = []
            self.y = []

            self.time = time.time()

            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

            plt.show(block=False)

        def show(self):
            self.ax.clear()
            self.lock.acquire()
            self.ax.plot(self.x,self.y)
            self.lock.release()
            plt.draw()

        def pushy(self,y):
            self.lock.acquire()
            self.y.append(y)
            if len(self.x)>0:
                self.x.append(self.x[-1]+1)
            else:
                self.x.append(0)
            self.lock.release()

    p = plotter()

    endflag = False
    def msgloop():
        while True:
            msg = conn.recv()

            # messages should be tuples,
            # msg[0] should be string

            if msg[0] == 'pushy':
                p.pushy(msg[1])
            elif msg[0] == 'show':
                p.show()
            else:
                conn.close()
                endflag=True
                return

    t = th.Thread(target = msgloop, daemon = True)
    t.start()

    last_xlen = 0
    def showable():
        nonlocal last_xlen
        p.lock.acquire()
        xlen = len(p.x)
        if xlen> last_xlen:
            last_xlen = xlen
            canshow = True
        else:
            canshow = False
        p.lock.release()
        return canshow

    while True:
        if showable():
            p.show()
        plt.pause(0.2)
        if endflag:
            return

class interprocess_plotter:
    def __init__(self):
        self.pc, cc = Pipe()
        self.p = Process(target = remote_plotter, args=(cc,), daemon=True)
        self.p.start()

    def pushy(self,y):
        self.pc.send(('pushy', y))

    def show(self):
        self.pc.send(('show',))

    def __del__(self):
        self.pc.send(('exit',))
        print('(ip)waiting for join...')
        self.p.join()

if __name__=='__main__':
    ip = interprocess_plotter()
    import math,time
    for i in range(100):
        ip.pushy(math.sin(i/10))
        time.sleep(0.05)

    time.sleep(5)
