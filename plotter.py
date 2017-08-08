from multiprocessing import Process, Pipe

def remote_plotter(conn,num_lines):
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

        def show(self):
            self.ax.clear()
            self.lock.acquire()
            for y in self.ys:
                self.ax.plot(self.x,y)
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

        def pushys(self,ys):
            self.lock.acquire()
            for idx in range(self.num_lines):
                self.ys[idx].append(ys[idx])

            if len(self.x)>0:
                self.x.append(self.x[-1]+1)
            else:
                self.x.append(0)
            self.lock.release()

    p = plotter(num_lines)

    endflag = False
    def msgloop():
        while True:
            msg = conn.recv()

            # messages should be tuples,
            # msg[0] should be string

            if msg[0] == 'pushys':
                p.pushys(msg[1])
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
    def __init__(self,num_lines=1):
        self.pc, cc = Pipe()
        self.p = Process(target = remote_plotter, args=(cc,num_lines), daemon=True)
        self.p.start()

    def pushys(self,ys):
        self.pc.send(('pushys', ys))

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
