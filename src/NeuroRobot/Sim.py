import math
import numpy as np
from NeuroRobot.Controller import Controller

class TrialType():
    LIGHTA = 1
    LIGHTB = 2
    BOTH_BLINKB = 3
    BOTH_BLINKA = 4

def dist(p1,p2):
    distv = p2 - p1
    return math.sqrt(distv[0]**2 + distv[1]**2)

class Trial():
    def __init__(self,netparams,trialtype = TrialType.LIGHTA,trialLength = 125):
        self.trialLength = trialLength
        self.trialtype = trialtype
        self.controller = Controller(netparams)     
        r1 = np.random.rand()*2*np.pi
        d1 = 100 + 50*np.random.rand()                
        self.lightA = Light(d1 * np.array([math.cos(r1),math.sin(r1)]))
        
        r2 = r1 + np.pi/2 + np.random.rand() * np.pi
        d2 = 100 + 50*np.random.rand()
        self.lightB = Light(d2 * np.array([math.cos(r2),math.sin(r2)]))
        
        self.robot = Robot(self.controller,[self.lightA,self.lightB],trialtype=trialtype)
        self.fp = 0
        self.ps = []
        self.score = 0
    
    def run(self):
        
        disti = 0
        distf = 0
        thresh = 16
        fpCount = 0
        self.ps = []
         
        if(self.trialtype == TrialType.LIGHTA | self.trialtype == TrialType.BOTH_BLINKB):
            disti = self.getDists()[0]
        else:
            disti = self.getDists()[1]
             
        for t in range(0,100):#np.arange(0,self.trialLength, self.robot.timestep):
            self.robot.step()
            if(self.trialtype == TrialType.LIGHTA | self.trialtype == TrialType.BOTH_BLINKB):
                if(self.getDists()[0] <= thresh):
                    fpCount += 1
            else:
                if(self.getDists()[1] <= thresh):
                    fpCount += 1
            self.ps.append(self.robot.controller.nn.getPlasticity())
             
            
        self.fitP = fpCount/10
        self.fitH = np.mean(self.ps)
        
        if(self.trialtype == TrialType.LIGHTA | self.trialtype == TrialType.BOTH_BLINKB):
            distf = self.getDists()[0]
        else:
            distf = self.getDists()[1]
        
        self.fitD = 1 - (distf/disti)
        
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
            return self.dx,self.dy
    
    def step(self):
        self.lastInput = self.getSensorInput()
        (mL,mR) = self.controller.step(self.lastInput)
        v = (mL + mR) / 2
        a = (mL - mR) * 2 * self.radius
        dist = v * self.timestep
        ang = a * self.timestep
        self.heading += ang
        self.dx,self.dy = dist * math.cos(self.heading),dist*math.sin(self.heading)
        self.position = [self.position[0] + self.dx, self.position[1] + self.dy]
    
    def getSensorPosition(self,sensor):
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
            
