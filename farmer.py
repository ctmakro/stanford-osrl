# farmer.py

# connector to the farms
from pyro_helper import pyro_connect

farmport = 20099

farmlist = [
'localhost',
]

def addressify(farmaddr,port):
    return farmaddr+':'+str(port)

def addresses():
    return [addressify(farm,farmport) for farm in farmlist]

class remoteEnv:
    def __init__(self,fp,id): # fp = farm proxy
        self.fp = fp
        self.id = id

    def reset(self):
        return self.fp.reset(self.id)

    def step(self,actions):
        return self.fp.step(self.id, actions)

    def rel(self):
        self.fp.rel(self.id)
        self.fp._pyroRelease()

class farmer:
    def __init__(self):
        for address in addresses:
            fp = pyro_connect(address,'farm')
            print('(farmer) forced renewing... '+address)
            fp.forcerenew()
            fp._pyroRelease()

    # find non-occupied instances from all available farms
    def acq_env(self):
        for address in addresses:
            fp = pyro_connect(address,'farm')
            result = fp.acq()
            if result == False: # no free ei
                fp._pyroRelease() # destroy proxy
                continue
            else: # result is an id
                id = result
                re = remoteEnv(fp,id) # build remoteEnv around the proxy
                print('(farmer) got one on '+address+' '+str(id))
                return re

        # if none of the farms has free ei:
        return False

    def renew(self):
        for address in addresses:
            fp = pyro_connect(address,'farm')
            fp.renew()
            print('(farmer) '+address+' renewed.')
