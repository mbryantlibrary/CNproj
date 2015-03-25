from NeuroRobot import Visualiser
from NeuroRobot.Sim import TrialType, Trial
from NeuroRobot.GA import NetParams

#v = Visualiser.Visualiser(trialtype=TrialType.BOTH_BLINKA)

t = Trial(NetParams(), TrialType.BOTH_BLINKA)
t.run()
print(t.getDists())
