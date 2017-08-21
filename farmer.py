# farmer.py

# connector to the farms
from pyro_helper import pyro_connect

farmport = 20099

from farmlist import farmlist

def addressify(farmaddr,port):
    return farmaddr+':'+str(port)

addresses = [addressify(farm[0],farmport) for farm in farmlist]
capacities = [farm[1] for farm in farmlist]

failures = [0 for i in range(len(capacities))]

total_capacity = sum(capacities)

class remoteEnv:
    def __init__(self,fp,id): # fp = farm proxy
        self.fp = fp
        self.id = id

    def reset(self):
        return self.fp.reset(self.id)

    def step(self,actions):
        return self.fp.step(self.id, actions)

    def rel(self):
        while True: # releasing is important, so
            try:
                self.fp.rel(self.id)
                break
            except:
                pass
        self.fp._pyroRelease()

class farmer:
    def __init__(self):
        for idx,address in enumerate(addresses):
            fp = pyro_connect(address,'farm')
            print('(farmer) forced renewing... '+address)
            try:
                fp.forcerenew(capacities[idx])
                print('(farmer) fp.forcerenew() success on '+address)
            except Exception as e:
                print('(farmer) fp.forcerenew() failed on '+address)
                print(e)
                fp._pyroRelease()
                continue
            fp._pyroRelease()

    # find non-occupied instances from all available farms
    def acq_env(self):
        import random # randomly sample to achieve load averaging
        # l = list(enumerate(addresses))
        l = list(range(len(addresses)))
        random.shuffle(l)

        for idx in l:
            address = addresses[idx]
            capacity = capacities[idx]

            if failures[idx]>0:
                # wait for a few more rounds upon failure,
                # to minimize overhead on querying busy instances
                failures[idx] -= 1
                continue
            else:
                fp = pyro_connect(address,'farm')
                try:
                    result = fp.acq(capacity)
                except Exception as e:
                    print('(farmer) fp.acq() failed on '+address)
                    print(e)
                    fp._pyroRelease()
                    failures[idx] += 4
                    continue

                if result == False: # no free ei
                    fp._pyroRelease() # destroy proxy
                    failures[idx] += 4
                    continue
                else: # result is an id
                    id = result
                    re = remoteEnv(fp,id) # build remoteEnv around the proxy
                    print('(farmer) got one on '+address+' '+str(id))
                    return re

        # if none of the farms has free ei:
        return False

    def renew(self):
        for idx,address in enumerate(addresses):
            fp = pyro_connect(address,'farm')
            try:
                fp.renew(capacities[idx])
            except Exception as e:
                print('(farmer) fp.renew() failed on '+address)
                print(e)
                fp._pyroRelease()
                continue
            print('(farmer) '+address+' renewed.')
            fp._pyroRelease()
