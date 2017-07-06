import cv2
import numpy as np
import time
import threading as td
imgw = 200
imgh = 300

# cv2.startWindowThread()

class wavegraph(object):
    def __init__(self,dims,name,colors):
        self.dims = dims
        self.lastshow = time.time()
        self.lastq = np.zeros((dims,))

        self.name = name
        self.imgw = 200
        self.imgh = 300

        self.im = np.zeros((self.imgh,self.imgw,3),dtype='float32')

        self.colors = colors
        self.que = []
        # self.quelock = td.Lock()
        # deque is thread safe.

        self.painter = td.Thread(target=self._one)
        self.painter.start()

    def one(self,q):
        # self.quelock.acquire()
        self.que.append(q)
        # self.quelock.release()

        if not self.painter.is_alive():
            # print('dead, new one')
            self.painter = td.Thread(target=self._one)
            self.painter.start()

        if time.time()-self.lastshow>0.125:
            if hasattr(self,'im2'):
                self.lastshow=time.time()
                cv2.namedWindow(self.name)
                cv2.imshow(self.name,self.im2)
                cv2.waitKey(1)

    def _one(self):
        loop = 60 # loop 60 times before quit
        while loop>0:
            loop-=1
            time.sleep(0.032) #30 fps
            while len(self.que) > 0:
                # print('winfrey paint...')
                # self.quelock.acquire()
                q = self.que.pop(0)
                # self.quelock.release()

                imgw,imgh = self.imgw,self.imgh
                im = self.im
                for i,k in enumerate(q):
                    q[i] = q[i] % imgh
                    while q[i]< -imgh/2: q[i]+=imgh
                    while q[i]> imgh/2 +10: q[i]-=imgh

                im[:,0:imgw-1] = im[:,1:imgw]
                im[:,imgw-1:imgw] = 0.

                dq = np.floor(-q+imgh/2).astype('int32')
                dlq = np.floor(-self.lastq+imgh/2).astype('int32')

                for i,k in enumerate(dq):
                    if dq[i]+1>dlq[i]: dq[i],dlq[i] = dlq[i],dq[i]
                    if dq[i]==dlq[i]: dlq[i]+=1

                    im[dq[i]:dlq[i],imgw-1] += (1.5/(dlq[i]-dq[i]))**0.4 * self.colors[i]

                self.lastq = q

            im2=im.copy()

            for i,k in enumerate(dq):
                im2[dq[i],:] += self.colors[i]/2.

            self.im2 = im2

            # if time.time()-self.lastshow>0.1:
            #     if hasattr(self,'im2'):
            #         self.lastshow=time.time()
            #         cv2.namedWindow(self.name)
            #         cv2.imshow(self.name,self.im2)
            #         cv2.waitKey(1)

        # print('loop ended')

lastshow = time.time()
lastq = 0
def showwave(q):
    global imgw
    global lastq

    while q < - imgh/2:
        q += imgh

    while q > imgh/2+10:
        q -= imgh

    if hasattr(showwave,'im'):
        showwave.im[:,0:imgw-1] = showwave.im[:,1:imgw]
        showwave.im[:,imgw-1:imgw] = 0.
    else:
        showwave.im = np.zeros((imgh,imgw,3),dtype='float32')

    im = showwave.im
    dq = int(-q+imgh/2)
    dlq = int(-lastq+imgh/2)

    if dq+1>dlq:
        dq,dlq = dlq,dq

    if dq == dlq : dlq+=1

    im[dq:dlq,imgw-1] = (1.3/(dlq-dq))**.4

    lastq = q

    global lastshow
    if time.time()-lastshow>0.2:
        lastshow=time.time()

        im2=im.copy()
        im2[dq,:]+=np.array([.3,.5,.7],dtype='float32')
        cv2.imshow('q(s,a)',im2)
        cv2.waitKey(1)

def showbar(actions,idx):
    l = len(actions)
    actions*=.5
    global imgw
    imgw = max(imgw,l)

    while np.mean(actions) < - imgh/2:
        actions += imgh

    while np.mean(actions) > imgh/2+10:
        actions -= imgh

    if hasattr(showbar,'im'):
        showbar.im *= [0.93,0.99,0.995]
    else:
        showbar.im = np.zeros((imgh,imgw,3),dtype='float32')

    im = showbar.im

    segw = int(imgw/l)

    for i in range(l):
        hei = actions[i]
        hei = int(-hei + imgh/2)

        if i==idx:
            c = np.array([1.,1.,1.])
            im[hei-2:hei+2,(i)*segw:(i+1)*segw] = 1.
        else:
            c = np.array([.4,.5,.8])
            pass


        im[hei:hei+1,i*segw:(i+1)*segw,:] = c
        # im[hei:hei+2,i*segw:i*segw+segw-3,:] += 0.2 * 0.8

    im = np.clip(im,a_max=1.0,a_min=0.0)

    global lastshow
    if time.time()-lastshow>0.2:
        lastshow=time.time()
        cv2.imshow('stat',im)
        cv2.waitKey(1)

def test():
    # showbar([0,10,20,-10,-20])
    showwave(10)
    showwave(30)
