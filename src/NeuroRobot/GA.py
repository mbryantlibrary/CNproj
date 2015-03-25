import numpy as np
from Sim import Trial

'''
Implements Harvey's Microbial GA (http://www.sussex.ac.uk/Users/inmanh/Microbial.pdf) with 2 player tournament selection, mutation and recombination.
'''
from NeuroRobot.Sim import TrialType


def mapRange(x,low,high):
    # map a number from a [-1,1] range to a [low,high] range.
    return ((high-low) * (x+1) / 2) + low

class Genotype():
    def __init__(self,nosc=3):
        #initialise a uniformly random genome with values in [-1,1]
        #nosc is the number of oscillators to use
        
        self.genes = np.random.uniform(-1,1,size=(nosc*4 + 3))
        self.n = nosc
    
    def mutate(self,creep=0.4):
        #add random Gaussian creep to each gene
        #if creep would take gene out of [-1,1] range, it is reflected
        #eg adding 0.5 to a gene of 0.8 would take it to 0.7
        
        for i,gene in enumerate(self.genes):
            cr = np.random.normal(scale=creep)
            newGene = gene + cr
            
            #check not out of range
            if(newGene > 1):
                newGene = 2 - newGene
            elif(newGene < -1):
                newGene = -2 - newGene
            self.genes[i] = newGene
    
    def crossOver(self,otherGeno,pCross=0.05):
        #copy genes from another genotype with probability pCross
        #helps proliferate 'good' genes in the population
        for i,gene in enumerate(self.genes):
            if(np.random.rand() < pCross):
                gene = otherGeno.genes[i]
    
    def __str__(self):
        return str(self.genes)
    
class NetParams():    
    def __init__(self,genotype=Genotype(),weights=np.random.uniform(size=3)):
        genes = genotype.genes
        n = self.n=genotype.n
        self.natFreqs = []
        self.sensorGains = []
        self.prefPhases = []
        self.plasticityRates = []
        self.weights = weights
        for i in range(0,n):
            self.natFreqs.append(mapRange(genes[i],0,5))
            self.sensorGains.append(mapRange(genes[i+n],-8,8))
            self.prefPhases.append(mapRange(genes[i+2*n],-np.pi/2,np.pi/2))
            self.plasticityRates.append(mapRange(genes[i+2*n],0,0.9))
            
        self.maxOscCoupling = mapRange(genes[n*4],0,5)
        self.biasL = mapRange(genes[n*4 + 1],0,2*np.pi)
        self.biasR = mapRange(genes[n*4 + 2],0,2*np.pi)
    
    def reset_weights(self):
        self.weights=np.random.uniform(size=3)
    
    def __str__(self):
        s = "Parameters:\n"
        s += "\tMaxOscCoupling=" + str(self.maxOscCoupling) + "\n"
        s += "\tBiasL=" + str(self.biasL) + "\n"
        s += "\tBiasR=" + str(self.biasR) + "\n"
        s += "Oscillators:\n"
        for i,freq in enumerate(self.natFreqs):
            s+="\tOsc " + str(i) + ":\n"
            s+="\t\tNatFreq=" + str(freq) + "\n"
            s+="\t\tPrefPhase=" + str(self.prefPhases[i]) + "\n"
            s+="\t\tPlastRate=" + str(self.plasticityRates[i]) + "\n"
        return s
    
class GA():
    def __init__(self,npop=100,gen=200,mut=0.4,cross=0.05,deme=5):
        print("Initialising GA...")
        self.npop = npop
        self.gen = gen
        self.mut = mut
        self.cross = cross
        self.deme = deme
        
        self.pop = []
        self.fitnesses = []
        self.stats = []
        for i in range(0,npop):
            print("Calculating %d of %d fitnesses..." % (i+1,npop))
            self.pop.append(Genotype())
            self.fitnesses.append(self.get_fitness(i))
            
        print("GA initialised.")
    
    def run(self,print_output=False):
        print("Starting GA run...")
        for i in range(0,self.gen):
            (a,b) = self.choose_indices()
            fitA = self.fitnesses[a]
            fitB = self.fitnesses[b]
            if(fitA > fitB):
                self.reproduce(a,b)
            else:
                self.reproduce(b,a)
            self.stats.append(GAStats(i,self.fitnesses))
            if(print_output):
                print(self.stats[-1])
        print("Finished")
                
    def choose_indices(self):
        a = np.random.randint(len(self.pop))
        b = 0
        while(b==0):
            b = np.random.randint(2 * self.deme) - self.deme
        b = a + b
        if(b < 0):
            b = len(self.pop) + b
        elif(b >= len(self.pop)):
            b -= len(self.pop)
        return (a,b)           
    
    def reproduce(self,winner,loser):
        self.pop[loser].mutate()
        self.pop[loser].crossOver(self.pop[winner])
        self.fitnesses[loser] = self.get_fitness(loser)
    
    def get_fitness(self,index):
        return Fitness(self.pop[index]).calculate()
    
    def __str__(self):
        return str(self.pop)
    
class GAStats():
    def __init__(self,n,fitnesses):
        self.n = n
        self.maxFit = np.max(fitnesses)
        self.avgFit = np.mean(fitnesses)
        self.minFit = np.min(fitnesses)
        self.variance = np.var(fitnesses)
            
    def __str__(self):
        return "N = %3i | Max = %f | min = %f | avg = %f | var = %f" % (self.n,self.maxFit, self.minFit, self.avgFit,self.variance)

class Fitness():
    def __init__(self,genotype):
        self.netp = NetParams(genotype)
        self.trials = []
    
    def calculate(self):
        sumFit = 0
        sumFit += self.doRun(TrialType.LIGHTA)
        sumFit += self.doRun(TrialType.LIGHTB)
        sumFit += self.doRun(TrialType.BOTH_BLINKB)
        sumFit += self.doRun(TrialType.BOTH_BLINKA)
        return sumFit/4
    
    def doRun(self,trialtype):
        self.netp.reset_weights()
        for i in range(0,5):
            Trial(self.netp,trialtype).run()
        sumFit = 0
        for i in range(0,3):
            t = Trial(self.netp,trialtype)
            sumFit = t.run()
        return sumFit/3