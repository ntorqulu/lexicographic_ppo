import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'Seed': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'Return': [200, 195, 210, 220, 225, 215, 205, 200, 210],
    'Agent': ['0', '0', '0', '1', '1', '1', '0', '0', '0']
}

df = pd.DataFrame(data)

palette = {'0': 'yellow', '1': 'red'}

plt.figure(figsize=(10, 6))

sns.boxplot(x='Return', y='Agent', data=df, hue='Agent', palette=palette, legend=False)

plt.title('Box Plot of Average Returns by Agent')
plt.xlabel('Average Return')
plt.ylabel('Agent')

plt.show()
