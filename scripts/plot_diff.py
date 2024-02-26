import pickle
import numpy as np
import plotly.express as px

with open("../results/skip_8_22_linear_U_recompute_2_error_5.pkl", 'rb') as f:
    loaded_dict = pickle.load(f)

with open("../results/no_skip.pkl", 'rb') as f:
    base_dict = pickle.load(f)

batch_id = 0
batch_data = loaded_dict[batch_id]
b_data = base_dict[batch_id]

max_seq_len = 512

pick = 600
layers = 32
start = 128 + 40
end = 0 + pick

data = []
for x in range(start, end):
    data.append(batch_data[x])

base_data = []
for x in range(start, end):
    base_data.append(b_data[x])

plot_data = np.zeros((pick, layers, 3))
for i, (x, y) in enumerate(zip(data, base_data)):
    for j, token in enumerate(x):
        if x[j] == 99999999 or y[j] == 99999999:
            plot_data[i][j] = [255,0,0]
            continue
        if x[j] != y[j]:
            plot_data[i][j] = [0,0,0]
            continue
        if x[j] == y[j]:
            plot_data[i][j] = [255,255,255]
            continue

plot_data = np.swapaxes(plot_data, 0, 1)
fig = px.imshow(plot_data)
fig.show()