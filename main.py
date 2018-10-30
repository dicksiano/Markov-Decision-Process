import numpy as np

from mdp import GridMDP
import view 

if __name__ == '__main__':
  # Grid shape
  shape = (4, 8)

  # Gold
  goal1, goal2 = (1, 1), (2, 5)

  # Monster
  monster1, monster2 = (1, 0), (2, 4)

  # Pit
  pits = [ (0,1), (0,6), (1,2), (1,6), (3,2), (3,6) ]

  # Rewards 
  defaultReward = -0.1
  goldReward = 100
  monsterPunishment = -100
  pitPunishment = -50

  # Reward table
  rewardTable = np.zeros(shape) + defaultReward
  rewardTable[goal1] = goldReward
  rewardTable[goal2] = goldReward
  rewardTable[monster1] = monsterPunishment
  rewardTable[monster2] = monsterPunishment
  for pit in pits:
    rewardTable[pit] = pitPunishment
    
  # Grid positions that ends the game
  terminalPos = np.zeros_like(rewardTable, dtype=np.bool)
  goldPos = np.zeros_like(rewardTable, dtype=np.bool)
  monsterPos = np.zeros_like(rewardTable, dtype=np.bool)
  terminalPos[goal1] = goldPos[goal1] = True
  terminalPos[goal2] = goldPos[goal2] = True
  terminalPos[monster1] = monsterPos[monster1] = True
  terminalPos[monster2] = monsterPos[monster2] = True

  # Action probabilities
  actionProbabilities=[
                          {"move": 0, "prob": 0.6}, # Sucess
                          {"move": 1, "prob": 0.4}, # Robot slips to Right!
                      ]

  grid = GridMDP(rewardTable, terminalPos, actionProbabilities)
  policy, utilityTable = grid.valueIteration(0.9, 1000)

  # Visualization
  view.plotHeatmap(utilityTable)
  view.plotPolicy(policy, goldPos, monsterPos)

  # Print results
  print('Result of Value Iteration:')
  print(utilityTable)
  print('\nBest policy: ')
  print(policy)