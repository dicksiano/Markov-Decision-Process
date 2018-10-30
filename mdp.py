import numpy as np

class GridMDP:

    """ Follow this codification is very important!
        It makes that if the robot slips to orthogonal position 
        (wich is the +1), it goes to the correct next state
    """
    possibleActions = [
        (-1, 0),  # up      --- up    + 1 = right
        ( 0, 1),  # right   --- right + 1 = down
        ( 1, 0),  # down    --- down  + 1 = left
        ( 0, -1), # left    --- left  + 1 = up
    ]
    numActions = len(possibleActions)

    def __init__(self, rewardTable, terminalPos, actionProbabilities):
        self.rewardTable = rewardTable
        self.terminalPos = terminalPos
        self.shape = rewardTable.shape
        self.M = rewardTable.shape[0]
        self.N = rewardTable.shape[1]
        self.size = rewardTable.size
        self.transitionMatrix = self.initTransitionMatrix(actionProbabilities)

    def initTransitionMatrix(self, actionProbabilities):
        T = np.zeros((self.M, self.N, self.numActions, self.M, self.N))     # T(s, a, s')
        x, y = np.unravel_index(np.arange(self.size), self.shape)           # Separate coordinates for each cell

        for action in range(self.numActions):
            for act in actionProbabilities:
                realMove = (action + act["move"]) % self.numActions        # up -> right -> down -> left -> up -> ...

                dx, dy = self.possibleActions[realMove]                    # dx, dy
                X = np.clip(x + dx, 0, self.M - 1)                         # x' = x + dx
                Y = np.clip(y + dy, 0, self.N - 1)                         # y' = y + dy

                T[x, y, action, X, Y] += act["prob"]                       # (x,y) --a--> (x',y')

        terminalPos = np.where(self.terminalPos.flatten())[0]
        T[x[terminalPos], y[terminalPos], :, :, :] = 0                     # Monster, gold: restart the iteration

        return T
    
    def valueIteration(self, discount, iterations):
        utilityTable = np.zeros_like(self.rewardTable)                      # Matrix M x N auxiliar
        
        for i in range(iterations):
            utilityTable = self.updateUtilityTable(utilityTable, discount)  # Bellman's update
        return self.bestPolicy(utilityTable), utilityTable

    def updateUtilityTable(self, utilityTable, discount):
        table = np.zeros_like(utilityTable)

        for i in range(self.M):                                             # Update each cell (i,j)
            for j in range(self.N):
                table[i, j] = self.calculateUtility(i, j, discount, utilityTable)
        return table

    def calculateUtility(self, i, j, discount, utilityTable):
        if self.terminalPos[ (i,j) ]:
            return self.rewardTable[ (i,j) ]

        # Vi+1(s) <- R(s) + γ max Σ P(s' | a, s) Vi(s')
        return self.rewardTable[ (i,j) ] + discount * np.max( self.nextStates(utilityTable, i, j) )

    def nextStates(self, utilityTable, i, j):
        s = self.transitionMatrix[i, j, :, :, :] * utilityTable
        for s1 in s:
            if s1[i][j] != 0: s1[i][j] = s1[i][j] - 10                      # If robot collides with wall
        return  np.sum( np.sum(s, axis=-1), axis=-1)                        # Keeps on same spot, punishment of -10

    def bestPolicy(self, utilityTable):                                     # Find action which maximizes P(s | a, s') V(s')
        return np.argmax((utilityTable * self.transitionMatrix).sum(axis=-1).sum(axis=-1), axis=2)  # Axis = 2 chooses action