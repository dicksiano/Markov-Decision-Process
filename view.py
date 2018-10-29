import matplotlib.pyplot as plt
import numpy as np
import cv2

def plotHeatmap(utilityGrid, grid):
    utilityNormalized = (utilityGrid - utilityGrid.min()) / (utilityGrid.max() - utilityGrid.min())
    utilityNormalized = (255*utilityNormalized).astype(np.uint8)

    rgbTable = cv2.applyColorMap(utilityNormalized, cv2.COLORMAP_JET)

    plt.imshow(rgbTable[:, :, ::1])    
    plt.show()

