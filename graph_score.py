import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

a = pd.read_pickle('score.bin')
a["k"] = range(2, 12)
for col in ['km_scores', 'km_silhouette', 'db_score', 'vmeasure_score']:
    plt.figure()
    a.plot(x='k', y=col, label=col)
    plt.show()
