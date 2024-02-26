import matplotlib.pyplot as plt
import numpy as np
import torch
import math

def plot(file_name, legend):
    nlls = []
    total_nll = 0
    with open(file_name) as fp:
        Lines = fp.readlines()
        for line in Lines:
            nnl = torch.tensor(float(line))
            nlls.append(nnl)
            total_nll += 1


    # interval = 128'
    interval = 128

    avg_ppls = []
    labels = []
    for i in range(total_nll):
        if i % interval == 0 and i > 0:
            # avg_ppl = torch.exp(torch.stack(nlls[i-interval:i]).mean())
            avg_ppl = torch.exp(torch.stack(nlls[:i]).mean())
            print(avg_ppl)
            avg_ppls.append(math.log(avg_ppl))
            labels.append(str(i))

    x_axis = [x for x in range(len(avg_ppls))]
    plt.xticks(x_axis, labels, rotation='vertical')
    plt.plot(x_axis, avg_ppls, marker="*", label=legend)


plot("../results/skip_8_22_linear_U_recompute_2_error_1_check_3.txt", "1")
plot("../results/skip_8_22_linear_U_recompute_2_error_3_check_3.txt", "3")
plot("../results/skip_8_22_linear_U_recompute_2_error_8_check_3.txt", "8")
plot("../results/skip_8_22_linear_U_recompute_2_error_15_check_3.txt", "15")
plot("../results/skip_8_22_linear_U_recompute_2_error_20_check_3.txt", "20")
plot("../results/no_skip.txt", "none")


plt.ylabel("ppl")
# plt.ylabel("tokens")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()