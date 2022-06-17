import pandas as pd
import numpy as np
import torch
import os; os.system('')

print(torch.cuda.is_available())

df = pd.DataFrame(np.zeros([4,4]))
print(df)
outputs = [0,1,1,1]
labels = [0,1,0,1]

# for every class value in tensor
lst = []
for i in range(0,3):
    lst.append(df[outputs[i]][labels[i]])

print(sum(lst) / len(lst))