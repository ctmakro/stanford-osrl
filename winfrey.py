import numpy as np
import time
import threading as td

class wavegraph(object):
    def __init__(self,dims,name,colors):
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

        def _one():
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
                        if dq[i]+1>dlq[i]: dq[i],dlq[i] = dlq[i],dq[i]
                        if dq[i]==dlq[i]: dlq[i]+=1

                        im[dq[i]:dlq[i],cursor] += (1.5/(dlq[i]-dq[i]))**0.4 * self.colors[i]

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

        self.painter = td.Thread(target=_one,daemon=True) # dead on exit
        self.painter.start()

    def one(self,q):
        self.que.append(q)
