import numpy as np
from NeuroRobot.Sim import Trial

'''
Implements Harvey's Microbial GA (http://www.sussex.ac.uk/Users/inmanh/Microbial.pdf) with 2 player tournament selection, mutation and recombination.
'''
from NeuroRobot.Sim import TrialType


def mapRange(x,low,high):
    # map a number from a [-1,1] range to a [low,high] range.
    return ((high-low) * (x+1) / 2) + low

class Genotype():
    def __init__(self,genes=np.random.uniform(-1,1,size=(21))):
        #initialise a uniformly random genome with values in [-1,1]
        
        self.genes = genes
        self.n = 3
    
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
        #helps keep 'good' genes in the population
        for i,gene in enumerate(self.genes):
            if(np.random.rand() < pCross):
                gene = otherGeno.genes[i]
    
    def __str__(self):
        return str(self.genes)
    
class NetParams():    
    def __init__(self,genotype=Genotype(),weights=np.random.uniform(size=6)):
        genes = genotype.genes
        self.natFreqs = []
        self.prefPhases = []
        self.maxOscCouplings = []
        self.sensorGains = []
        self.plasticityRates = []
        self.weights = weights #weights are randomly initialised by default
        
        self.maxOscCoupling= mapRange(genes[0],0,5)
        for i in range(1,6,2):
            #genes 0-8 encode oscillator parameters; 0-2 is osc 1, 4-6 osc 2 etc.
            self.natFreqs.append(mapRange(genes[i],0,5))
            self.prefPhases.append(mapRange(genes[i + 1],-np.pi/2,np.pi/2))
        for i in range(7,10):
            #genes 9-11 encode a rate of plasticity change for each coupling
            self.plasticityRates.append(mapRange(genes[i],0,0.9))
        for i in range(10,14):
            #genes 12-15 encode gains for each sensor: AL, AR, BL, BR
            self.sensorGains.append(mapRange(genes[i],-8,8))
        self.biasL = mapRange(genes[14],0,2*np.pi)
        self.biasR = mapRange(genes[15],0,2*np.pi)
        self.H1 = mapRange(genes[16],0,0.2)
        self.H2 = mapRange(genes[17],0.2,0.25)
    
    def reset_weights(self):
        #called at the start of each fitness run to randomise weights
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
    def __init__(self,npop=30,gen=100,mut=0.4,cross=0.05,deme=5):
        print("Initialising GA...")
        self.npop = npop #number of individuals
        self.gen = gen #number of generations to run
        self.mut = mut #mutation rate, or stdev of Gaussian noise to add
        self.cross = cross #probability of each gene being copied, between 0 and 1
        self.deme = deme #deme size for geographic selection, indices selected with no more than this difference
        self.pop = []
        self.fitnesses = []
        self.stats = []
        #calculate all fitnesses
        for i in range(0,npop):
            print("Calculating %d of %d fitnesses..." % (i+1,npop))
            self.pop.append(Genotype())
            self.fitnesses.append(self.get_fitness(i))
        print("GA initialised.")
    
    def run(self,print_output=False):
        #performs an entire evolutionary run
        print("Starting GA run...")
        for i in range(0,self.gen):
            #select two individuals
            (a,b) = self.choose_indices()
            fitA = self.fitnesses[a]
            fitB = self.fitnesses[b]
            #which is fitter?
            if(fitA > fitB):
                self.reproduce(a,b)
            else:
                self.reproduce(b,a)
            self.stats.append(GAStats(i,self.fitnesses))
            if(print_output):
                print(self.stats[-1])
        print("Finished")
                
    def choose_indices(self):
        #selects two different individuals from a population
        #with geographic selection - imagine the indices lying on a 1D ring.
        #individuals are selected from within the deme size, e.g if deme=5
        #pairs could be (1,5), (95,99), (70,69) but not (30,40).
        #values are wrapped so (99,3) is a valid pair
        a = np.random.randint(len(self.pop))
        b = 0
        while(b==0): #can't select same individual twice
            b = np.random.randint(2 * self.deme) - self.deme
        b = a + b
        #check index not out of bounds and adjust if so
        if(b < 0):
            b = len(self.pop) + b
        elif(b >= len(self.pop)):
            b -= len(self.pop)
        return (a,b)           
    
    def reproduce(self,winner,loser):
        #do reproduction steps
        self.pop[loser].mutate()
        self.pop[loser].crossOver(self.pop[winner])
        #genotype is modified so only now do we measure fitness
        self.fitnesses[loser] = self.get_fitness(loser)
    
    def get_fitness(self,index):
        return Fitness(self.pop[index]).calculate()
    
    def save(self,filename):
        dataArray = np.array([])
        for i in range(0,self.npop):
            dataRow = np.hstack(((self.fitnesses[i],self.pop[i].genes)))
            dataArray = np.vstack((dataArray,dataRow))
        np.savetxt(fname=filename,X=dataArray,delimiter=',')
            
    def __str__(self):
        return str(self.pop)
    
class GAStats():
    #to keep track of per-generation stats
    def __init__(self,n,fitnesses):
        self.n = n
        self.maxFit = np.max(fitnesses)
        self.avgFit = np.mean(fitnesses)
        self.minFit = np.min(fitnesses)
        self.variance = np.var(fitnesses)
            
    def __str__(self):
        return "N = %3i | Max = %f | min = %f | avg = %f | var = %f" % (self.n,self.maxFit, self.minFit, self.avgFit,self.variance)

class Fitness():
    #calculates fitness of an individual
    def __init__(self,genotype):
        self.netp = NetParams(genotype)
        self.trials = []
    
    def calculate(self):
        #get average of a run of each different trial type
        sumFit = 0
        sumFit += self.doRun(TrialType.LIGHTA)
        sumFit += self.doRun(TrialType.LIGHTB)
        sumFit += self.doRun(TrialType.BOTH_BLINKB)
        sumFit += self.doRun(TrialType.BOTH_BLINKA)
        return sumFit/4
    
    def doRun(self,trialtype):
        self.netp.reset_weights() 
        #weights are initialised to zero at the start of the run
        #but are persistent throughout the rest of the run to allow plasticity
        for i in range(0,5):
            #these trials are not evaluated, just for plasticity
            Trial(self.netp,trialtype).run()
        sumFit = 0
        for i in range(0,3):
            #evaluate three trials for fitness
            t = Trial(self.netp,trialtype)
            sumFit = t.run()
        return sumFit/3 #average fitness
    