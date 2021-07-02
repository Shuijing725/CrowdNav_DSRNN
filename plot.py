
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np


legends = ['DS-RNN FoV=360', '']

# change the folder directories here!
# for holonomic weight
logs1 = pd.read_csv("data/example_model/progress.csv")
# for the unicycle weight
# logs1 = pd.read_csv("data/example_model_unicycle/progress.csv")

#3:logs3, 4:logs4, 5:logs5, 6:logs6, 7:logs7, ,9:logs9
logDicts={1:logs1} # 1:logs1, 2:logs2, 3:logs3, 4:logs4, 5:logs5}
graphDicts={0:'eprewmean', 1:'loss/value_loss'}

legendList=[]
# summarize history for accuracy

# for each metric
for i in range(len(graphDicts)):
	plt.figure(i)
	plt.title(graphDicts[i])
	j = 0
	for key in logDicts:
		if graphDicts[i] not in logDicts[key]:
			continue
		else:
			plt.plot(logDicts[key]['misc/total_timesteps'],logDicts[key][graphDicts[i]])

			legendList.append(legends[j])
			print('avg', str(key), graphDicts[i], np.average(logDicts[key][graphDicts[i]]))
		j = j + 1
	print('------------------------')

	plt.xlabel('total_timesteps')
	plt.legend(legendList, loc='lower right')
	legendList=[]



plt.show()


