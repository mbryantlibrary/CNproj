from NeuroRobot.GA import GA,Genotype

def testGAChoosesTwoIndicesWithinDeme():
    ga = GA(npop=20,deme=3)
    (a,b) = ga.choose_indices()
    print("a = {}, b = {}".format(a,b))
    assert abs(a-b) <= 3
    
def testGenotypeMutatesGenome():
    gen = Genotype()
    lenA = len(gen.genes)
    gen.mutate()
    lenB = len(gen.genes)
    print("lenBefore = {}, lenAfter = {}".format(lenA,lenB))
    assert lenA == lenB
    
    