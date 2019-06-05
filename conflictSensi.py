import numpy as np
import random
import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import Point
import time
from joblib import Parallel, delayed
import multiprocessing

######################################### calculate lh lp ############################################################
class Bound:
    def __init__(self, cfg, vwind0, alphaw0):
        self.vuav = cfg['vuav']
        self.a = cfg['a']
        self.tt = cfg['tt']
        self.ori = cfg['ori']
        self.des = cfg['des']
        self.vwind0 = vwind0
        self.alphaw0 = alphaw0

    ############################## proposed error #######################################################################
    def winderror(self):
        #np.random.seed(100)
        self.evwind = 0 #np.random.normal(0, 0.05 * self.vwind0)
        #random.seed(100)
        ff = random.randint(1,2)
        if ff == 1 :
            self.evwind = -1 * self.evwind
       # np.random.seed(100)
        self.ealphaw = 0 #np.random.normal(0, 0.05 * self.alphaw0)
        #np.random.seed()

    def uaverror(self):
        np.random.seed()
        self.epGPS = 0 #np.random.uniform(-1.5, 1.5)
        self.evuav = 0 #np.random.normal(0, 0.05 * self.vuav)
        self.ea = 0 #np.random.normal(0, 0.05 * self.a)
        self.ett = 0 #np.random.normal(0, 0.05 * self.tt)

    ######################### calculate size (same in 3 case)#######################################################################
    def size(self):
        # np.random.seed()
        # self.uaverror()
        ### angle between uav and wind
        self.vecwind = ((self.vwind0 + self.evwind) * np.cos(self.alphaw0 + self.ealphaw),
                        (self.vwind0 + self.evwind) * np.sin(self.alphaw0 + self.ealphaw))
        self.uavdir = np.subtract(self.des, self.ori)
        self.uavvvec = self.vuav / np.linalg.norm(self.uavdir) * self.uavdir
        self.alphaw = np.arccos(
             np.dot(self.uavdir, self.vecwind) / (np.linalg.norm(self.uavdir) * np.linalg.norm(self.vecwind)))

        ### size
        self.valong = self.vuav + self.evuav + (self.vwind0 + self.evwind) * (np.cos(self.alphaw))
        self.vper = (self.vwind0 + self.evwind) * (np.sin(self.alphaw))

        self.lh = abs(self.valong) * (self.tt + self.ett) + (self.valong) ** 2 / 2 / (self.a + self.ea)
        self.lp = abs(self.vper) * (self.tt + self.ett) + (self.vper) ** 2 / 2 / (self.a + self.ea) + abs(self.epGPS)
        self.r = self.lh + self.lp


class Conflict:
    def __init__(self, cfg1, cfg2, vwind0, alphaw0):
        self.cfg1 = cfg1
        self.cfg2 = cfg2
        # type: (object, object, object, object) -> object
        ### uav1
        self.vuav1 = cfg1['vuav']
        self.a1 = cfg1['a']
        self.tt1 = cfg1['tt']
        self.ori1 = cfg1['ori']
        self.des1 = cfg1['des']
        ### uav2
        self.vuav2 = cfg2['vuav']
        self.a2 = cfg2['a']
        self.tt2 = cfg2['tt']
        self.ori2 = cfg2['ori']
        self.des2 = cfg2['des']
        ### wind
        self.vwind0 = vwind0
        self.alphaw0 = alphaw0

    ####################### find boundary ##############################################################
    def uavbound(self):
        ####################### stop time ##############################################################
        self.ref = Bound(self.cfg1, self.vwind0, self.alphaw0)
        self.ref.winderror()
        self.ref.uaverror()
        self.ref.size()
        self.tuav1 = np.linalg.norm(np.subtract(self.des1, self.ori1)) / self.ref.valong
        self.move = Bound(self.cfg2, self.vwind0, self.alphaw0)
        self.move.evwind = self.ref.evwind
        self.move.ealphaw = self.ref.ealphaw
        self.move.uaverror()
        self.move.size()
        self.tuav2 = np.linalg.norm(np.subtract(self.des2, self.ori2)) / self.move.valong
        self.systime = min(self.tuav1, self.tuav2)

        ####################### ref, move boundary ##############################################################
        #self.prolh = np.empty([1, 600])
        #self.prolp = np.empty([1, 600])
        #self.ref2 = Bound(self.cfg1, self.vwind0, self.alphaw0)

        #self.prolh2 = np.empty([1, 600])
        #self.prolp2 = np.empty([1, 600])
        #self.move2 = Bound(self.cfg2, self.vwind0, self.alphaw0)

        # for j in range(0, 600):
        #     self.ref2.winderror()
        #     self.ref2.uaverror()
        #     self.ref2.size()
        #     self.prolh[0, j] = self.ref2.lh
        #     self.prolp[0, j] = self.ref2.lp
        # 
        #     self.move2.winderror()
        #     self.move2.uaverror()
        #     self.move2.size()
        #     self.prolh2[0, j] = self.move2.lh
        #     self.prolp2[0, j] = self.move2.lp
        # 
        # self.prolh = np.sort(self.prolh)
        # self.prolp = np.sort(self.prolp)
        # 
        # self.ref.lp =  self.prolp[0, 539]
        # self.ref.lh = self.prolh[0, 539]
        # 
        # self.prolh2 = np.sort(self.prolh2)
        # self.prolp2 = np.sort(self.prolp2)
        # 
        # self.move.lp = self.prolp2[0, 539]
        # self.move.lh = self.prolh2[0, 539]

        self.refcir1 = self.ori1
        self.refr = self.ref.lp
        self.refcir2 = self.ori1 + self.ref.lh / (np.linalg.norm(self.ref.uavdir)) * np.array(self.ref.uavdir)
        self.refortho = (-self.ref.uavdir[1], self.ref.uavdir[0])
        self.refp1 = self.refcir1 + self.ref.lp / np.linalg.norm(self.refortho) * np.array(self.refortho)
        self.refp2 = self.refcir2 + self.ref.lp / np.linalg.norm(self.refortho) * np.array(self.refortho)
        self.refp3 = self.refcir2 - self.ref.lp / np.linalg.norm(self.refortho) * np.array(self.refortho)
        self.refp4 = self.refcir1 - self.ref.lp / np.linalg.norm(self.refortho) * np.array(self.refortho)
        self.refrect = [self.refp1, self.refp2, self.refp3, self.refp4]


        self.refvel = self.move.uavvvec + self.move.vecwind - self.ref.uavvvec - self.ref.vecwind
        self.movecir1 = self.ori2
        self.mover = self.move.lp
        self.movecir2 = self.movecir1 + np.array(self.refvel) * self.systime + self.move.lh / np.linalg.norm(self.refvel) * np.array(self.refvel)
        self.moveortho = (-self.refvel[1], self.refvel[0])
        self.movep1 = self.movecir1 + self.move.lp / np.linalg.norm(self.moveortho) * np.array(self.moveortho)
        self.movep2 = self.movecir2 + self.move.lp / np.linalg.norm(self.moveortho) * np.array(self.moveortho)
        self.movep3 = self.movecir2 - self.move.lp / np.linalg.norm(self.moveortho) * np.array(self.moveortho)
        self.movep4 = self.movecir1 - self.move.lp / np.linalg.norm(self.moveortho) * np.array(self.moveortho)
        self.moverect = [self.movep1, self.movep2, self.movep3, self.movep4]

    ##################### collision check ##############################################################
    def collision(self):
        self.uavbound()
        FLAG = 0
        geo = geometry()
        self.fff = 1
        if self.ref.valong < 0 or self.move.valong < 0:
            self.fff = 0
        if  self.fff == 1:
            if (
                    geo.Flag2rect(self.moverect, self.refrect) == 1 or
                    geo.Flagrectc(self.moverect, self.refcir1, self.refr) == 1 or
                    geo.Flagrectc(self.moverect, self.refcir2, self.refr) == 1 or
                    geo.Flag2cir(self.movecir1, self.mover, self.refcir1, self.refr) == 1 or
                    geo.Flag2cir(self.movecir1, self.mover, self.refcir2, self.refr) == 1 or
                    geo.Flagrectc(self.refrect, self.movecir1, self.mover) == 1 or
                    geo.Flag2cir(self.movecir2, self.mover, self.refcir1, self.refr) == 1 or
                    geo.Flag2cir(self.movecir2, self.mover, self.refcir2, self.refr) == 1 or
                    geo.Flagrectc(self.refrect, self.movecir2, self.mover) == 1
                ):

                    FLAG = 1
                #print 'collision'
        return FLAG



############################### geometry collision check ###########################################################
class geometry:
    ###################### check collision between rect and circle #################################################
    def dist(self, p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        px = x2 - x1
        py = y2 - y1

        something = px * px + py * py

        u = ((x3 - x1) * px + (y3 - y1) * py) / (float(something)+1e-8)
        #print ((x3 - x1) * px + (y3 - y1) * py), u, something, float(something)
        if u > 1:
            u = 1
        elif u < 0:
            u = 0

        x = x1 + u * px
        y = y1 + u * py

        dx = x - x3
        dy = y - y3

        dist = math.sqrt(dx * dx + dy * dy)
        return dist

    def Flagrectc(self, rect, c, r):
        rect = rect
        c = c
        r = r

        distances = [self.dist(rect[i], rect[j], c) for i, j in zip([0, 1, 2, 3], [1, 2, 3, 0])]
        point = Point(c)
        polygon = Polygon(rect)

        flag = 0
        if any(d < r for d in distances) == True:
            flag = 1
        if any(d < r for d in distances) == False and polygon.contains(point) == True: #np.abs(
                #distances[0] + distances[2] - np.linalg.norm(np.subtract(rect[3], rect[0]))) < 10 ** -10 and np.abs(
            #distances[1] + distances[3] - np.linalg.norm(np.subtract(rect[1], rect[0]))) < 10 ** -10:
            flag = 1  # type: int
        return flag

    ####################### check collision between 2 rect ########################################
    def Flag2rect(self, poly1, poly2):
        polygons = [Polygon(poly1), Polygon(poly2)]
        flag = 0
        if polygons[0].intersects(polygons[1]) == True and polygons[0].touches(polygons[1]) == False:
            flag = 1
        return flag

    ####################### check collision between 2 circle ########################################
    def Flag2cir(self, c1, r1, c2, r2):
        flag = 0
        if np.linalg.norm(np.subtract(c1, c2)) < r1 + r2:
            flag = 1
        return flag



################## Draw #################################################################################
class Draw:
    ###################### draw rect ################################################################
    def drawrect(self, ax, rect):
        coord = rect
        coord.append(coord[0]) #repeat the first point to create a 'closed loop'
        xs, ys = zip(*coord) #create lists of x and y values
        plt.plot(xs, ys)

    ################# draw circle ###################################################################
    def drawcir(self, ax, cir, r):
        circle1 = plt.Circle(cir, r)
        ax.add_artist(circle1)



################# ori des ###############################################################################
class orides:
    def __init__(self, xx, yy):
        self.xx = xx
        self.yy = yy

    ############# a=(x=0, x=xx, y=0, y=yy)########################################################################
    def point(self):
        aa = [(0, np.random.uniform(0, self.yy)), (self.xx, np.random.uniform(0, self.yy)), (np.random.uniform(0, self.xx), 0), (np.random.uniform(0, self.xx), self.yy)]
        a = range(0, 3)
        o = random.choice(a)
        self.ori = aa[o]
        a.remove(o)
        d = random.choice(a)
        self.des = aa[d]
        return self.ori
        return self.des


############################################################################################################
if __name__ == '__main__':

    N = 2000
    Cpro = 0
    Cdyn = 0
    Csta = 0
    xx = 800
    yy = 800

    inputs = range(N)
    # uav1pos = []
    # uav2pos = []
    #
    # for i in range(N):
    #     ccc1 = orides(xx, yy)
    #     ccc1.point()
    #     ccc2 = orides(xx, yy)
    #     ccc2.point()
    #     uav1pos.append([ccc1.ori, ccc1.des])
    #     uav2pos.append([ccc2.ori, ccc2.des])
    #
    # from tempfile import TemporaryFile
    # outfile = TemporaryFile()
    # np.save('outfile1', uav1pos)
    # np.save('outfile2', uav2pos)

    uav1pos = np.load('outfile1.npy')
    uav2pos = np.load('outfile2.npy')


    t1=time.time()
    a=0
    resuptime = []
    #resaa = []
    #reswindalpha = []
    #resuavv = []
    #reswind = []

    #def processInput(i):
    for j in range(200):
        uptime = 0.005 + 0.01*j #11 #200
        #aa = 3.2+0.0495*j #19 #72
        #windalpha = 0.0314*j #201
        #vuav = 14 + 0.2*j #61
        #wind = 0.0143 + 0.0286*j #200
        a=0
        for i in range(0, N):
            #print i
            ccc1 = orides(xx, yy)
            ccc1.point()
            ccc2 = orides(xx, yy)
            ccc2.point()

            cfg1 = {'vuav': 20, #np.random.lognormal(3,0.1), #np.random.lognormal(3,0.1),
                'a': 4.95, # aa, #np.random.uniform(3.2, 6.7), #np.random.uniform(3.2, 6.7),
                'tt': uptime, #1, #np.random.uniform(0,2), #np.random.uniform(0,2),
                'ori': uav1pos[i][0],
                'des': uav1pos[i][1]} #ccc1.des}

            cfg2 = {'vuav': 20, #np.random.lognormal(3,0.1), #np.random.lognormal(3,0.1),
                'a': 4.95, #aa, #np.random.uniform(3.2, 6.7), #np.random.uniform(3.2, 6.7),
                'tt':uptime, #1, #np.random.uniform(0,2), #np.random.uniform(0,2),
                'ori': uav2pos[i][0], #ccc2.ori,
                'des': uav2pos[i][1]} #ccc2.des}

            vwind0 = 2.86 #3.2 * np.random.weibull(2.2)
            alphaw0 = np.pi  #np.random.uniform(0, 2 * np.pi)

            cr1 = Conflict(cfg1, cfg2, vwind0, alphaw0)
        #Cpro = Cpro + cr1.collision()
            a = a + cr1.collision()

        resuptime.append([uptime, a])
        #resaa.append([aa, a])
        #reswindalpha.append([windalpha, a])
        #resuavv.append([vuav, a])
        #reswind.append([wind, a])

    t2 = time.time()

    resuptime = np.array(resuptime)
    #resaa = np.array(resaa)
    #reswindalpha = np.array(reswindalpha)
    #resuavv = np.array(resuavv)
    #reswind = np.array(reswind)


    with open('testuptime1.txt', 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(resuptime.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in resuptime:
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt='%-7.2f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')


    plt.figure(1)
    ax = plt.gca()
    plt.xlim(-100, 1000)
    plt.ylim(-100,1000)
    Draw().drawrect(ax, cr1.moverect)
    plt.hold('True')
    Draw().drawrect(ax, cr1.refrect)
    plt.hold('True')
    Draw().drawcir(ax, cr1.refcir1, cr1.refr)
    plt.hold('True')
    Draw().drawcir(ax, cr1.refcir2, cr1.refr)
    plt.hold('True')
    Draw().drawcir(ax, cr1.movecir1, cr1.mover)
    plt.hold('True')
    Draw().drawcir(ax, cr1.movecir2, cr1.mover)
    plt.show()