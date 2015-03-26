from tkinter import Tk,Canvas,Frame
from GA import Genotype,NetParams
from Sim import Trial,TrialType


class RobotCanvas(Frame):
    
    def __init__(self,parent,genotype,trialtype):
        Frame.__init__(self,parent)
        
        netp = NetParams(genotype)
        trial = Trial(netp,trialtype=trialtype)
        
        self.canvas = Canvas(parent, width=400,height=400)
        self.canvas.pack()
        self.robotID = self.createOval(0,0,4)
        if(trial.trialtype != TrialType.LIGHTB):
            self.lightAID = self.createOval(trial.lightA.position[0], trial.lightA.position[1], 5,colour="red")
        if(trial.trialtype != TrialType.LIGHTA):
            self.lightBID = self.createOval(trial.lightB.position[0], trial.lightB.position[1], 5,colour="green")
        self.trial=trial
    
    def drawFrame(self):
        self.canvas.create_rectangle
        dx,dy = self.trial.robot.getMovement()
        dx *= 1000
        dy *= 1000
        self.canvas.move(self.robotID,dx,dy)
    
    def createOval(self,x,y,r,canvasW=400,scale=1,colour="blue"):
        r *= scale
        rX,rY = x + canvasW/2, (-y + canvasW/2)
        x1,y1 = rX - r, rY - r
        x2,y2 = rX + r, rY + r
        return self.canvas.create_oval(x1,y1,x2,y2, fill=colour)

    def animate(self):
        self.trial.robot.step()
        self.drawFrame()
        self.after(100,self.animate)

class Visualiser():    
    def __init__(self,genotype=Genotype(),trialtype = TrialType.BOTH_BLINKA):
        self.win = Tk()
        self.rc = RobotCanvas(self.win,genotype,trialtype)
        
        self.rc.animate()
        self.win.mainloop()