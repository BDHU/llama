import pickle
import numpy as np
import plotly.express as px

with open("../results/skip_8_22_linear_U_recompute_2_error_3_check_3.pkl", 'rb') as f:
    loaded_dict = pickle.load(f)

batch_id = 0

batch_data = loaded_dict[batch_id]
print(batch_data.keys())

pick = 2600
layers = 32
start = 128
end = 0 + pick

data  = []
for x in range(start, end):
    data.append(batch_data[x])

plot_data = np.zeros((pick, layers, 3))
for i, x in enumerate(data):
    for j, token in enumerate(x):
        if j == 0:
            # black
            plot_data[i][j] = [0,0,0]
            continue
        if x[j] == 99999999:
            plot_data[i][j] = [255,0,0]
            continue
        if x[j] == x[j-1]:
            plot_data[i][j] = [255,255,255]
            continue
        if x[j] != x[j-1]:
            plot_data[i][j] = [0,0,0]
            continue

plot_data = np.swapaxes(plot_data, 0, 1)
fig = px.imshow(plot_data)
fig.show()