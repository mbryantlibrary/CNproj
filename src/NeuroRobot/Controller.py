import numpy as np
from math import sin
import cmath

def wmF(k):
    '''
    Maps a weight change output from the plasticity function to a sinusoidal function
    '''
    a = np.sin(0.5*k)
    a[a<0]=0
    return a

def wmFScalar(ks):
    a = np.sin(0.5*ks)
    if(a < 0):
        return 0
    else:
        return a

def p(xarr,H1=(0.2*np.pi),H2=(0.8*np.pi)):
    '''
    Determines amount of plasticity.
    '''
    rarr = []
    c = H1/(H2-H1)
    for x in xarr:
        ax = np.abs(x)
        if(ax<H1):
            rarr.append(0)
        elif(ax>H2):
            rarr.append(1)
        else:
            rarr.append(ax/(H2-H1) - c)
    return np.array(rarr)

class Coupling():
    def __init__(self,nodeA,nodeB,weight=np.random.uniform(),maxOscCoupling=2,plasticityRate = 0.3,prefPhase = 0.2*np.pi):
        self.nodeA = nodeA
        self.nodeB = nodeB
        self.weight = weight
        self.plasticityRate = plasticityRate
        self.prefPhase = prefPhase
        self.updateWeight()
        
    def getPlasticity(self):
        return p([self.nodeA.phi() - self.prefPhase])
    
    def updateWeight(self):
        ddk = self.plasticityRate * self.getPlasticity() * sin(self.nodeA.theta - self.nodeB.theta) - self.prefPhase
        self.weight += np.asscalar(ddk)
        self.strength = wmFScalar(self.weight)

class Oscillator():
    def __init__(self,name,natFreq=np.random.uniform(0,6)):
        self.theta = 0
        self.natFreq = natFreq
        self.couplings = []
        self.name = name
        
    def phi(self):
        phaseSum = complex(0,0)
        for coupling in self.couplings:
            phaseSum += complex(coupling.strength,0) * np.exp(1j*(coupling.nodeB.theta - self.theta))
        return np.angle(phaseSum)
    
    def __str__(self):
        s = "Oscillator {}: \n\tTheta={}\n\tOmega={}\n\tCouplings:".format(self.name,self.theta,self.omega)
        for coupling in self.couplings:
            s += "\n\t\tNode " + str(coupling.nodeB.name) + ", strength=" + str(coupling.strength)
        return s


class KNetwork():
    def __init__(self,netparams,timestep=0.1):
        self.netp = netparams
        n = netparams.n
        self.nodes = []
        self.timestep = timestep
        self.reporter = Reporter()
        self.biasL = netparams.biasL
        self.biasR = netparams.biasR
        
        for i in range(0,n):
            self.nodes.append(Oscillator(i,netparams.natFreqs[i]))
        for i,node in enumerate(self.nodes):
            if(i == len(self.nodes)-1):
                nodeJ = self.nodes[0]
            else:
                nodeJ = self.nodes[i+1]
            node.couplings.append(Coupling(node,nodeJ,weight=netparams.weights[i],maxOscCoupling=netparams.maxOscCoupling,plasticityRate=netparams.plasticityRates[i],prefPhase=netparams.prefPhases[i]))
    
    def getPlasticity(self):
        sumP = 0
        for node in self.nodes:
            for coupling in node.couplings:
                sumP += coupling.getPlasticity()
        return sumP/len(self.nodes)
    
    def step(self):
        thetas = []
        phases = []
        ps = []
        w = []
        for node in self.nodes:
            phaseSum = 0
            for coupling in node.couplings:
                phaseSum += coupling.strength + sin(coupling.nodeB.theta - node.theta)
                ps.append(p([coupling.nodeA.phi() - coupling.prefPhase]))
                coupling.updateWeight()
                w.append(coupling.strength)
            node.theta = (node.theta + self.timestep*(node.natFreq + phaseSum)) % (2 * np.pi)
        for node in self.nodes:
            thetas.append(node.theta)
            phases.append(node.phi())
        self.reporter.update(thetas,phases,self.getMotorOutput(),ps)
        self.netp.weights = w
       
    def getMotorOutput(self):
        nodes = self.nodes
        Ml = 2 * sin(nodes[2].phi() - self.biasL)
        Mr = 2 * sin(nodes[2].phi() - self.biasR)
        return (Ml,Mr)
    
    def __str__(self):
        s = ""
        for i,nodes in enumerate(self.nodes):
            s += "----time=" + str(i) + "----\n"
            for node in nodes:
                s += str(node) + "\n"
        return s

class Controller():
    def __init__(self,netparams):
        self.nn = KNetwork(netparams)
    
    def step(self,sensors):
        self.nn.step()
        return self.nn.getMotorOutput()
        

class Reporter():
    def __init__(self):
        self.thetas = []
        self.phases = []
        self.motorOutputs = []
        self.ps = []
    
    def update(self,thetas,phases,motorOutputs,ps):
        self.thetas.append(thetas)
        self.phases.append(phases)
        self.motorOutputs.append(motorOutputs)
        self.ps.append(ps)