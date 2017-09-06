import numpy as np
import time

def remote_wavegraph_callback(conn):
    class wavegraph:
        def __init__(self,params):
            dims,name,colors = params

            self.dims = dims
            self.lastshow = time.time()
            self.lastq = np.zeros((dims,))

            self.name = name
            self.imgw = 200
            self.imgh = max(300, dims*12+60 + 20)

            self.im = np.zeros((self.imgh,self.imgw,3),dtype='float32')
            self.cursor = 0

            self.colors = colors
            self.que = []

        def paintloop(self):
            import cv2
            while True:
                time.sleep(0.032) #30 fps

                im = self.im
                imgw,imgh = self.imgw,self.imgh

                while len(self.que) > 0:
                    # print('winfrey paint...')
                    # self.quelock.acquire()
                    q = self.que.pop(0)
                    # self.quelock.release()
                    cursor = self.cursor

                    for i,k in enumerate(q):
                        q[i] = q[i] % imgh
                        while q[i]< -imgh/2: q[i]+=imgh
                        while q[i]> imgh/2 +10: q[i]-=imgh

                    # im[:,0:imgw-1] = im[:,1:imgw]
                    im[:,cursor:cursor+1] = 0.

                    dq = np.floor(-q+imgh/2).astype('int32')
                    dlq = np.floor(-self.lastq+imgh/2).astype('int32')

                    for i,k in enumerate(dq):
                        mi = min(dq[i], dlq[i])
                        ma = max(dq[i], dlq[i])

                        if mi==ma: ma+=1

                        im[mi:ma,cursor] += (1.5/((ma-mi)))**0.4 * self.colors[i]

                    self.lastq = q

                    self.cursor+=1
                    self.cursor%=self.imgw

                if (time.time()-self.lastshow)>0.125:
                    self.lastshow=time.time()

                    im2=im.copy()
                    cursor = self.cursor
                    im2[:,cursor:cursor+1] += .5

                    cv2.namedWindow(self.name)
                    cv2.imshow(self.name,im2)
                    cv2.waitKey(1)
                    cv2.waitKey(1)

        # self.painter = td.Thread(target=_one,daemon=True) # dead on exit
        # self.painter.start()
        def one(self,q):
            self.que.append(np.array(q))

    w = None
    endflag = False

    # wait for init parameters
    while 1:
        msg = conn.recv()
        if w is None:
            if msg[0] == 'init':
                w = wavegraph(msg[1])
                break

    def receive_loop():
        while 1:
            msg = conn.recv()
            if msg[0] == 'one':
                w.one(msg[1])
            else:
                return

    import threading as th
    th.Thread(target = receive_loop, daemon = True).start()
    w.paintloop()

from ipc import ipc
class wavegraph(ipc):
    def __init__(self,dims,name,colors):
        super().__init__(remote_wavegraph_callback)
        self.pretty('wavegraph initializing...')
        self.send(('init',(dims,name,colors)))
    def one(self,q):
        q = [float(i) for i in q]
        self.send(('one',q))
