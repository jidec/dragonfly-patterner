import pandas as pd
import numpy as np

df = pd.DataFrame(np.zeros([4,4]))
print(df)
outputs = [0,1,1,1]
labels = [0,1,0,1]

for i in range(0,3):
    print(df[outputs[i]][labels[i]])