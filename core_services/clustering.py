import pandas as pd
import numpy as np
from sklearn import cluster
import numpy as np
from sklearn.cluster import KMeans



def kmeansclustering(fileName, prefix, encoding):
	df = pd.read_csv(filepath_or_buffer='core_encodedFiles/' + encoding + '_' + fileName + '_' + str(prefix) +  '.csv', header=0)
	estimator = KMeans(n_clusters=5)	
	estimator.fit(df)
	print {i: np.where(estimator.labels_ == i)[0] for i in range(estimator.n_clusters)}



