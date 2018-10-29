import matplotlib.pyplot as plt
import numpy as np
import cv2

def plotHeatmap(utilityGrid):
    utilityNormalized = (utilityGrid - utilityGrid.min()) / (utilityGrid.max() - utilityGrid.min())
    utilityNormalized = (255*utilityNormalized).astype(np.uint8)

    rgbTable = cv2.applyColorMap(utilityNormalized, cv2.COLORMAP_JET)

    plt.imshow(rgbTable[:, :, ::1])    
    plt.show()

def plotPolicy(policy, goldPos, monsterPos):
    markers = "^>v<"
    size = 150 // np.max(policy.shape)
    boxWidth = size // 10
      
    for i, marker in enumerate(markers):
        y, x = np.where((policy == i) & np.logical_not(goldPos) & np.logical_not(monsterPos))
        plt.plot(x, policy.shape[0] - y, marker, ms=size)

    y, x = np.where(goldPos)
    plt.plot(x, policy.shape[0] - y, 'o', ms=size, mew=boxWidth)
    
    y, x = np.where(monsterPos)
    plt.plot(x, policy.shape[0] - y, 'x', ms=size, mew=boxWidth)

    plt.show()