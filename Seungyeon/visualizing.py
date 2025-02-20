# import
import pickle
import matplotlib.pyplot as plt

# define def
def visualizing_result_data(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)

    data.reset_index(drop=True, inplace=True)
    data.columns = ['label', 'point', 'win', 'draw', 'lose']

    labels = data['label'].unique()

    fig, axs = plt.subplots(5, 4, figsize=(20, 25), squeeze=False)

    for x in range(len(labels)):
        for y in range(4):
            temp = data.loc[data.label == labels[x]]
            axs[x, y].plot(temp[data.columns[y+1]])
            axs[x, y].set_title(f"{labels[x]}, {data.columns[y+1]}")

    plt.show()