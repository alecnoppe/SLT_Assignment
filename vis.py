import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sns

df_1a = pd.read_csv("data/1a_output.csv", index_col=0)
df_1b = pd.read_csv("data/1b_output.csv", index_col=0)
df_1c = pd.read_csv("data/1c_output.csv", index_col=0)

dfm_1a = df_1a.melt('k', var_name='train_test', value_name='error')

#1a
sns.lineplot(data=dfm_1a, x='k', y='error', hue='train_test')
plt.xticks(np.arange(1, 21, step=1))
#plt.axhline(0.93, color='r', linestyle='--')
#plt.axhline(0.957333, color='r', linestyle='--')
plt.axvline(6, color='r', linestyle='--')
plt.title("KNN Train/Test error for K between 1, 20")
plt.show()

#1b
sns.lineplot(data=df_1b, x='k', y='loocv_err')
plt.xticks(np.arange(1, 21, step=1))
plt.title("KNN LOOCV error for K between 1, 20")
#plt.axhline(0.928, color='r', linestyle='--')
plt.axvline(4, color='r', linestyle='--')
plt.show()


print(df_1c)
#1c

colors = {}
for x in range(1,21):
    colors[x] = "lightgray"

colors[8] = "red"
colors[2] = "green"

df_1c_fast = df_1c.loc[df_1c['p']==2]

# print(df_1c_fast.loc[df_1c_fast['loocv_err'].idxmax()] )
# print(df_1c.loc[df_1c['loocv_err'].idxmax()] )

sns.lineplot(data=df_1c, x='k', y='loocv_err', hue='p', palette=colors)
plt.xticks(np.arange(1, 21, step=1))
plt.title("KNN LOOCV error using Minkowski distance for K between 1, 20 and p between 1, 15")
plt.axvline(3, color='r', linestyle='--')
plt.axvline(4, color='g', linestyle='--')

plt.show()


sns.lineplot(data=df_1c, x='k', y='loocv_err')
plt.xticks(np.arange(1, 21, step=1))
plt.title("KNN LOOCV error using Minkowski distance for K between 1, 20 averaged over p between 1,15")
plt.axvline(3, color='r', linestyle='--')
plt.axvline(4, color='g', linestyle='--')

plt.show()


sns.lineplot(data=df_1c, x='p', y='loocv_err')
plt.xticks(np.arange(1, 21, step=1))
plt.title("KNN LOOCV error using Minkowski distance for p between 1, 15 averaged over K between 1, 20")
plt.axvline(8, color='r', linestyle='--')
plt.axvline(2, color='g', linestyle='--')

plt.show()