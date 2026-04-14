from matplotlib import pyplot as plt
from matplotlib import colors

cmap = colors.ListedColormap(['Red','Green','Grey'])
plt.figure(figsize=(10,10))
data=[[1,1,1,1,1,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,2,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,0],
      [1,0,0,2,2,0,0,0,0,0],
      [1,0,0,0,0,0,0,0,0,0],
      [1,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,0]]
plt.pcolor(data[::-1],cmap=cmap,edgecolor='k',linewidth=3)
plt.show()
