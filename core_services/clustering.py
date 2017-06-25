import pandas as pd
import numpy as np
from sklearn import cluster



def kmeansclustering(fileName, prefix, encoding):
	df = pd.read_csv(filepath_or_buffer='core_encodedFiles/' + encoding + '_' + fileName + '_' + str(prefix) +  '.csv', header=0)


