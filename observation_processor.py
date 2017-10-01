def get_observation(self):
    bodies = ['head', 'pelvis', 'torso', 'toes_l', 'toes_r', 'talus_l', 'talus_r']

    pelvis_pos = [self.pelvis.getCoordinate(i).getValue(self.osim_model.state) for i in range(3)]
    pelvis_vel = [self.pelvis.getCoordinate(i).getSpeedValue(self.osim_model.state) for i in range(3)]

    jnts = ['hip_r','knee_r','ankle_r','hip_l','knee_l','ankle_l']
    joint_angles = [self.osim_model.get_joint(jnts[i]).getCoordinate().getValue(self.osim_model.state) for i in range(6)]
    joint_vel = [self.osim_model.get_joint(jnts[i]).getCoordinate().getSpeedValue(self.osim_model.state) for i in range(6)]

    mass_pos = [self.osim_model.model.calcMassCenterPosition(self.osim_model.state)[i] for i in range(2)]
    mass_vel = [self.osim_model.model.calcMassCenterVelocity(self.osim_model.state)[i] for i in range(2)]

    body_transforms = [[self.osim_model.get_body(body).getTransformInGround(self.osim_model.state).p()[i] for i in range(2)] for body in bodies]

    muscles = [ self.env_desc['muscles'][self.MUSCLES_PSOAS_L], self.env_desc['muscles'][self.MUSCLES_PSOAS_R] ]

    # see the next obstacle
    obstacle = self.next_obstacle()

#        feet = [opensim.HuntCrossleyForce.safeDownCast(self.osim_model.forceSet.get(j)) for j in range(20,22)]
    self.current_state = pelvis_pos + pelvis_vel + joint_angles + joint_vel + mass_pos + mass_vel + list(flatten(body_transforms)) + muscles + obstacle
    return self.current_state

'''
above was copied from 'osim-rl/osim/env/run.py'.

observation:
0 pelvis r
1 x
2 y

3 pelvis vr
4 vx
5 vy

6-11 hip_r .. ankle_l [joint angles]

12-17 hip_r .. ankle_l [joint velocity]

18-19 mass_pos xy
20-21 mass_vel xy

22-(22+7x2-1=35) bodypart_positions(x,y)

36-37 muscles psoas

38-40 obstacles
38 x dist
39 y height
40 radius

radius of heel and toe ball: 0.05

'''

import numpy as np

class fifo:
    def __init__(self,size):
        self.size = size
        self.buf = [None for i in range(size)]
        self.head = 0
        self.tail = 0

    def push(self,obj):
        self.buf[self.tail] = obj
        self.tail+=1
        self.tail%= self.size

    def pop(self):
        item = self.buf[self.head]
        self.head+=1
        self.head%= self.size
        return item

    def fromhead(self,index):
        return self.buf[(self.head+index)%self.size]

    def fromtail(self,index):
        return self.buf[(self.tail-index-1)%self.size]

    def dump(self,reason):
        # dump the content into file
        with open('fifodump.txt','a') as f:
            string = 'fifodump reason: {}\n'.format(reason)
            for i in self.buf:
                string+=str(i)+'\n'
            string+='head:{} tail:{}\n'.format(self.head,self.tail)
            f.write(string)

# 41 dim to 48 dim
def process_observation(observation):
    o = list(observation) # an array

    pr = o[0]

    px = o[1]
    py = o[2]

    pvr = o[3]

    pvx = o[4]
    pvy = o[5]

    for i in range(6,18):
        o[i]/=4

    o = o + [o[22+i*2+1]-0.5 for i in range(7)] # a copy of original y, not relative y.

    # x and y relative to pelvis
    for i in range(7): # head pelvis torso, toes and taluses
        o[22+i*2+0] -= px
        o[22+i*2+1] -= py

    o[18] -= px # mass pos xy made relative
    o[19] -= py
    o[20] -= pvx # mass vel xy made relative
    o[21] -= pvy

    o[38]= min(4,o[38])/3 # ball info are included later in the stage
    # o[39]/=5
    # o[40]/=5

    o[0]/=4 # divide pr by 4
    o[1]=0 # abs value of pel x is not relevant
    o[2]-= 0.5 # minus py by 0.5

    o[3] /=4 # divide pvr by 4
    o[4] /=10 # divide pvx by 10
    o[5] /=10

    o[20]/=10
    o[21]/=10

    return o

_stepsize = 0.01
flatten = lambda l: [item for sublist in l for item in sublist]

# expand observation from 48 to 48*7 dims
processed_dims = 48 + 14*4 + 9*0 + 1*0 + 8
# processed_dims = 41*8
def generate_observation(new, old=None, step=None):

    global _stepsize
    if step is None:
        raise Exception('step should be a valid integer')

    # deal with old
    if old is None:
        if step!=0:
            raise Exception('step nonzero, old == None, how can you do such a thing?')

        old = {'dummy':None,'balls':[],'que':fifo(1200),'last':step-1}
        for i in range(6):
            old['que'].push(new)

    q = old['que']

    if old['last']+1 != step:
        raise Exception('step not monotonically increasing by one')
    else:
        old['last'] += 1

    if step > 1: # bug in osim-rl
        if q.fromtail(0)[36] != new[36]:
            # if last obs and this obs have different psoas value
            print('@step {} Damned'.format(step))
            q.push(['compare(que, new):', q.fromtail(0)[36], new[36]])
            q.dump(reason='obsmixed')
            raise Exception('Observation mixed up, potential bug in parallel code.')

    # q.pop() # remove head
    q.push(new) # add to tail

    # process new
    def lp(n):return list(process_observation(n))
    new_processed = lp(new)

    def bodypart_velocities(at):
        return [(q.fromtail(0+at)[i]-q.fromtail(1+at)[i])/_stepsize for i in range(22,36)]

    def relative_bodypart_velocities(at):
        # velocities, but relative to pelvis.
        bv = bodypart_velocities(at)
        pv1,pv2 = bv[2],bv[3]
        for i in range(len(bv)):
            if i%2==0:
                bv[i] -= pv1
            else:
                bv[i] -= pv2
        return bv

    vels = [bodypart_velocities(k) for k in [0,1]] #[[14][14]]
    relvels = [relative_bodypart_velocities(k) for k in [0,]] #[[14]]
    accs = [
        [
            (vels[t][idx] - vels[t+1][idx])/_stepsize
            for idx in range(len(vels[0]))]
        for t in [0,]]
    # [[14]]

    fv = [v/10 for v in flatten(vels)]
    frv = [rv/10 for rv in flatten(relvels)]
    fa = [a/10 for a in flatten(accs)]
    final_observation = new_processed + fv + frv + fa
    # 48+14*4

    # final_observation += flatten(
    #     [lp(q.fromtail(idx))[38:41] for idx in reversed([4,8,16,32,64])]
    # )
    # # 4 * 5
    # # 48*4

    balls = old['balls']

    def addball_if_new():
        current_pelvis = new[1]
        current_ball_relative = new[38]
        current_ball_height = new[39]
        current_ball_radius = new[40]

        absolute_ball_pos = current_ball_relative + current_pelvis

        if current_ball_radius == 0: # no balls ahead
            return

        compare_result = [abs(b[0] - absolute_ball_pos) < 1e-9 for b in balls]
        # [False, False, False, False] if is different ball

        got_new = sum([(1 if r==True else 0)for r in compare_result]) == 0

        if got_new:
            # for every ball there is
            for b in balls:
                # if this new ball is smaller in x than any ball there is
                if absolute_ball_pos < (b[0] - 1e-9):
                    print(absolute_ball_pos,balls)
                    print('(@ step )'+str(step)+')Damn! new ball closer than existing balls.')
                    q.dump(reason='ballcloser')
                    raise Exception('new ball closer than the old ones.')

            balls.append([
                absolute_ball_pos,
                current_ball_height,
                current_ball_radius,
            ])
            if len(balls)>3:
                print(balls)
                print('(@ step '+str(step)+')What the fuck you just did! Why num of balls became greater than 3!!!')
                q.dump(reason='ballgt3')
                raise Exception('ball number greater than 3.')
        else:
            pass # we already met this ball before.

    if step > 0:
        # initial observation is very wrong, due to implementation bug.
        addball_if_new()

    ball_vectors = []
    current_pelvis = new[1]

    # there should be at most 3 balls
    for i in range(3):
        if i<len(balls):
            rel = balls[i][0] - current_pelvis
            falloff = min(1,max(0,3-abs(rel))) # when ball is closer than 3 falloff become 1
            ball_vectors.append([
                min(4,max(-3, rel))/3, # ball pos relative to current pos
                balls[i][1] * 5 * falloff, # radius
                balls[i][2] * 5 * falloff, # height
            ])
        else:
            ball_vectors.append([
                0,
                0,
                0,
            ])

    # 9-d
    # final_observation += flatten(reversed(ball_vectors))

    # episode_end_indicator = max(0, (step/1000-0.6))/10 # lights up when near end-of-episode
    # final_observation[1] = episode_end_indicator
    #
    # final_observation += [episode_end_indicator]

    # flat_ahead_indicator = np.clip((current_pelvis - 5.0)/2, 0.0, 1.0)
    # # 0 at 5m, 1 at 7m
    #
    # final_observation += [flat_ahead_indicator]

    foot_touch_indicators = []
    for i in [29,31,33,35]: # y of toes and taluses
        # touch_ind = 1 if new[i] < 0.05 else 0
        touch_ind = np.clip(0.05 - new[i] * 10 + 0.5, 0., 1.)
        touch_ind2 = np.clip(0.1 - new[i] * 10 + 0.5, 0., 1.)
        # touch_ind2 = 1 if new[i] < 0.1 else 0
        foot_touch_indicators.append(touch_ind)
        foot_touch_indicators.append(touch_ind2)
    final_observation+=foot_touch_indicators # 8dim

    # for i,n in enumerate(new_processed):
    #     print(i,n)

    def final_processing(l):
        # normalize to prevent excessively large input
        for idx in range(len(l)):
            if l[idx] > 1: l[idx] = np.sqrt(l[idx])
            if l[idx] < -1: l[idx] = - np.sqrt(-l[idx])
    final_processing(final_observation)

    return final_observation, old

if __name__=='__main__':
    ff = fifo(4)
    ff.push(1)
    ff.push(3)
    ff.push(5)
    ff.pop()
    ff.pop()
    ff.push(6)
    ff.push(7)

    print(ff.fromhead(0))
    print(ff.fromhead(1))
    print(ff.fromtail(0))
    print(ff.fromtail(1))
