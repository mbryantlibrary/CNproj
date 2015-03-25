import math
import numpy as np
from NeuroRobot.Controller import Controller

class TrialType():
    #The type of trial to run
    LIGHTA = 1 #Only light A displayed
    LIGHTB = 2 #Only light B displayed
    BOTH_BLINKB = 3 #Both are displayed, light B blinks
    BOTH_BLINKA = 4 #Both are displayed, light A blinks

def dist(p1,p2):
    #distance between two points
    distv = p2 - p1
    return math.sqrt(distv[0]**2 + distv[1]**2)

class Trial():
    def __init__(self,netparams,trialtype = TrialType.LIGHTA,trialLength = 125):
        self.trialLength = trialLength
        self.trialtype = trialtype
        self.controller = Controller(netparams)
        
        #regardless of trial type, both lights are present. Trial type only affects
        #the sensory input the robot gets.
        r1 = np.random.rand()*2*np.pi #angle of light A from origin
        d1 = 100 + 50*np.random.rand() #distance
        self.lightA = Light(d1 * np.array([math.cos(r1),math.sin(r1)]))
        
        r2 = r1 + np.pi/2 + np.random.rand() * np.pi
        d2 = 100 + 50*np.random.rand()
        self.lightB = Light(d2 * np.array([math.cos(r2),math.sin(r2)]))
        
        self.robot = Robot(self.controller,[self.lightA,self.lightB],trialtype=trialtype)
        self.fp = 0 #proportion of time robot spends within distance of less than 4 times its radius
        self.ps = [] #stores amount of plasticity at each time step to average over time
        self.score = 0
    
    def run(self):
        
        disti = 0 #initial distance to light
        distf = 0 #final distance to light
        thresh = 16 #threshold for fp, i.e. 4 times robot radius
        fpCount = 0 #count of timesteps robot spends near light for fp
        self.ps = []
        
        #set initial light distance
        if(self.trialtype == TrialType.LIGHTA | self.trialtype == TrialType.BOTH_BLINKB):
            disti = self.getDists()[0] #target Light A
        else:
            disti = self.getDists()[1] #target Light B
        
        for t in np.arange(0,self.trialLength, self.robot.timestep):
            self.robot.step()
            if(self.trialtype == TrialType.LIGHTA | self.trialtype == TrialType.BOTH_BLINKB):
                if(self.getDists()[0] <= thresh):
                    fpCount += 1 #robot is near light
            else:
                if(self.getDists()[1] <= thresh):
                    fpCount += 1
            self.ps.append(self.robot.controller.nn.getPlasticity())
        
        self.fitP = fpCount/10 #proportion of time robot spends near light
        self.fitH = np.mean(self.ps) #average plasticity
        
        #get final distance to light
        if(self.trialtype == TrialType.LIGHTA | self.trialtype == TrialType.BOTH_BLINKB):
            distf = self.getDists()[0]
        else:
            distf = self.getDists()[1]
        
        self.fitD = 1 - (distf/disti) #fitness measure to minimise distance to light
        
        self.fitness = (self.fitD + self.fitP) * self.fitH * 1000
        return self.fitness
        
    def getDists(self):
        return dist(self.robot.position,self.lightA.position),dist(self.robot.position,self.lightB.position)

class Light():
    def __init__(self,position,intensity = 1):
        self.position = position
        self.intensity = intensity
    
    def getIntensityAtPoint(self,point,a=0.03,b=100):
        distv = self.position - point
        dist = math.sqrt(distv[0]**2 + distv[1]**2)
        angle = math.acos(distv[0]/dist)
        #if angle between sensor and light is above pi/2, nothing is sensed
        if(abs(angle) > math.pi/2):
            return 0
        else:
            return 0.5* (1+math.cos(angle))/(1 + math.exp(a*(dist-b)))
    def __str__(self):
        return "Position: [%f,%f], intensity = %f" % (self.position[0],self.position[1],self.intensity)

class Robot():
    def __init__(self,controller,lights,trialtype=TrialType.LIGHTA,radius=4,timestep=(1./60.)):
        self.position = [0,0]
        self.trialtype = trialtype
        self.heading = np.random.rand() * np.pi * 2
        self.radius = radius
        self.controller = controller
        self.timestep = timestep
        self.sensorOffsets = [-math.pi/3,math.pi/3]
        self.lights = lights
        self.lastInput = []
    
    def getMovement(self):
        #get current timestep change in x/y position, mainly for visualiser
        return self.dx,self.dy
    
    def step(self):
        #get sensory input and pass to controller to get wheel velocities
        self.lastInput = self.getSensorInput()
        (mL,mR) = self.controller.step(self.lastInput)
        v = (mL + mR) / 2 #standard differential drive model, get forward velocity
        a = (mL - mR) * 2 * self.radius #angular velocity given by multiplying by axle width
        dist = v * self.timestep #integrate
        ang = a * self.timestep
        self.heading += ang
        #calculate changes in position
        self.dx,self.dy = dist * math.cos(self.heading),dist*math.sin(self.heading)
        self.position = [self.position[0] + self.dx, self.position[1] + self.dy]
    
    def getSensorPosition(self,sensor):
        #get actual position of sensor from robot heading and radius
        #sensor = offset angle
        actualHeading = self.heading + sensor
        return self.position + self.radius * np.array([math.cos(actualHeading),math.sin(actualHeading)])
    
    def getSensorInput(self):
        inputs = []
        if(self.trialtype == TrialType.LIGHTA):
            for sensor in self.sensorOffsets:
                inputs.append(self.lights[0].getIntensityAtPoint(self.getSensorPosition(sensor)))
                inputs.append(0)
        elif(self.trialtype == TrialType.LIGHTB):
            for sensor in self.sensorOffsets:
                inputs.append(0)
                inputs.append(self.lights[1].getIntensityAtPoint(self.getSensorPosition(sensor)))
        elif(self.trialtype == TrialType.BOTH_BLINKA):
            blink = np.random.rand() < 0.15
            for sensor in self.sensorOffsets:
                if(blink):
                    inputs.append(self.lights[0].getIntensityAtPoint(self.getSensorPosition(sensor)))
                else:
                    inputs.append(0)
                inputs.append(self.lights[1].getIntensityAtPoint(self.getSensorPosition(sensor)))
        elif(self.trialtype == TrialType.BOTH_BLINKB):
            blink = np.random.rand() < 0.15
            for sensor in self.sensorOffsets:
                inputs.append(self.lights[0].getIntensityAtPoint(self.getSensorPosition(sensor)))
                if(blink):
                    inputs.append(self.lights[1].getIntensityAtPoint(self.getSensorPosition(sensor)))
                else:
                    inputs.append(0)
                        
        return inputs
            
