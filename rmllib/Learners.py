import pandas

class Learner():
    def __init__(self):
        return

class RNB():
    def __init__(self):
        super()
    
    def fit(self, data):
        labeledX = data.X.loc[data.Mask.Mask, :]
        labeledY = data.Y.loc[data.Mask.Mask, :]
        labeledE = data.E.loc[data.Mask.Mask, data.Mask.Mask]
        print(labeledX, labeledY)