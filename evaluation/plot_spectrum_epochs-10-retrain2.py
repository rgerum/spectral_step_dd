import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from rgerum_utils.format_glob import format_glob_pd

from helpers import draw_spectrum, iter_pair, get_step, plot_spectrum

#data = format_glob_pd("../cedar/logs_2023-04-21/noise-*/model-resnet_noise-{noise:f}_k-{k:d}_run-{run:d}/all.txt")
#data = format_glob_pd("../cedar/logs_2023-04-24/noise-0.2/model-resnet_noise-{noise:f}_k-{k:d}_run-{run:d}/ckpt_{epoch:d}.pth")
#data["classes"] = 10
#data["model"] = "resnet"
data2 = format_glob_pd("../logs_2023-06-02/{model}/classes-{classes:d}/noise-0.2/model-*_noise-{noise:f}_k-{k:d}_run-{run:d}/ckpt_{epoch:d}.pth")
data = data2#pd.concat((data, data2))
print(data)

k_unique = list(sorted(data.k.unique()))
data = data.sort_values(by=["k", "epoch"])
data = data[(data.k == 3) | (data.k == 12) | (data.k == 64)]
N = 10
model = "resnet"
data = data[data.model == model]
data = data[data.classes == N]
print(data)

k_unique = list(sorted(data.k.unique()))
noise_unique = list(sorted(data.epoch.unique()))
cols = len(data.epoch.unique())
rows = len(data.k.unique())
print("cols", cols, rows)
fig, subplots = plt.subplots(rows, cols, sharex=True, sharey=True)
ks = []
steps = np.zeros((rows, cols))*np.nan

for (epoch_id, k_id), ddd in iter_pair(data, "epoch", "k"):
    print(ddd.filename[:-4] + "_spectrum.npz")
    spect_d = np.load(ddd.filename[:-4] + "_spectrum.npz", allow_pickle=True)["output"][()]

    if 0:
        get_step(spect_d, N)
        steps[k_id, epoch_id] = spect_d["step_diff"]

        plt.sca(subplots[k_id, epoch_id])
        plt.title(f"{ddd.k} {ddd.epoch}")
        plot_spectrum(spect_d, N)
    plt.gca().spines[["top", "right"]].set_visible(False)
plt.suptitle(f"{N} {model}")
#plt.savefig(__file__[:-3]+"__1.png")

fig, subplots = plt.subplots(2, 1, sharex=True)
plt.sca(subplots[0])
for k_id, (noise, d) in enumerate(data.groupby("k")):
    index = ~np.isnan(steps[k_id])
    plt.semilogx(np.array(noise_unique)[index]+1, steps[k_id, index], '-o')
plt.gca().set(xlabel="k", ylabel="step")
plt.ylim(0, plt.gca().get_ylim()[1])

plt.sca(subplots[1])
for epoch_id, (noise, d) in enumerate(data.groupby("k")):
    acc = np.loadtxt(Path(d.iloc[0].filename).parent / "all.txt")
    plt.plot(acc[:, 2]+1, 1 - np.array(acc)[:, 0] / 100, '-o')
plt.gca().set(xlabel="k", ylabel="val error")
plt.ylim(0, 1)
plt.suptitle(f"{N} {model}")
#plt.savefig(__file__[:-3]+"__2.png")

fig, subplots = plt.subplots(1, 1, sharex=True)
plt.sca(subplots)

for epoch_id, (noise, d) in enumerate(data.groupby("k")):
    acc = np.loadtxt(Path(d.iloc[0].filename).parent / "all.txt")

    accuracy = []
    for i in noise_unique:
        accuracy.append(acc[i, 0])

    from helpers import plot_color_grad
    plot_color_grad(steps[epoch_id, :], 1 - np.array(accuracy)[:] / 100, marker="o", ms=2)
    #plt.plot(steps[epoch_id, :], 1 - np.array(accuracy)[:] / 100, '-o')
    plt.plot(steps[epoch_id, :][0], 1 - np.array(accuracy)[0] / 100, '+k', ms=5)
plt.gca().set(xlabel="step", ylabel="val error")
plt.ylim(0, 1)
plt.xlim(0, plt.gca().get_xlim()[1])
plt.suptitle(f"{N} {model}")
#plt.savefig(__file__[:-3]+"__3.png")

""" xxx """
fig, subplots = plt.subplots(2, 2, sharex=True, sharey=True)
if 0:
    plt.sca(subplots[0])
    for k_id, (noise, d) in enumerate(data.groupby("k")):
        index = ~np.isnan(steps[k_id])
        plt.semilogx(np.array(noise_unique)[index]+1, steps[k_id, index], '-o')
    plt.gca().set(xlabel="k", ylabel="step")
    plt.ylim(0, plt.gca().get_ylim()[1])

    plt.sca(subplots[1])

for epoch_id, (noise, d) in enumerate(data.groupby("k")):
    acc = np.loadtxt(Path(d.iloc[0].filename).parent / "all.txt")
    plt.sca(subplots[epoch_id%2, epoch_id//2])
    plt.semilogx(acc[:, 2]+1, 1 - np.array(acc)[:, 0] / 100, '-o', color=f"C{epoch_id}")
plt.gca().set(xlabel="k", ylabel="val error")
plt.ylim(0, 1)

plt.sca(subplots[0, 1])
data2 = format_glob_pd("../logs_2023-05-19/resnet-50epochs/classes-10/noise-0.2/epoch-*/model-resnet_noise-0.2_k-64_run-0/all.txt")
data2b = format_glob_pd("../logs_2023-05-25/resnet-50epochs_control/classes-10/noise-0.2/epoch-*/model-resnet_noise-0.2_k-64_run-0/all.txt")
data2bb = format_glob_pd("../logs_2023-06-01/resnet-50epochs_control2/classes-10/noise-0.2/epoch-*/model-resnet_noise-0.2_k-64_run-0/all.txt")
retrain_error = []
retrain_error2 = []
retrain_error2b = []
retrain_epochs = np.array([0, 2, 9, 29, 99, 299, 999, 2999])
for i, file in data2.iterrows():
    d = np.loadtxt(file.filename)
    retrain_error.append(d[-1, 0])
    retrain_error2.append(np.loadtxt(data2b.iloc[i].filename)[-1, 0])
    retrain_error2b.append(np.loadtxt(data2bb.iloc[i].filename)[-1, 0])
print(d.shape, d[-1, 0])
plt.plot(retrain_epochs+1, 1-np.array(retrain_error)/100, "--oC2", label="retrain", mec="w")
plt.plot(retrain_epochs+1, 1-np.array(retrain_error2)/100, ":oC2", label="control", mec="w")
plt.plot(retrain_epochs+1, 1-np.array(retrain_error2b)/100, "-oC2", label="control2", mec="w")
retrain_error64 = retrain_error

plt.sca(subplots[1, 0])
data2 = format_glob_pd("../logs_2023-05-19/resnet-50epochs/classes-10/noise-0.2/epoch-*/model-resnet_noise-0.2_k-12_run-0/all.txt")
data2b = format_glob_pd("../logs_2023-05-25/resnet-50epochs_control/classes-10/noise-0.2/epoch-*/model-resnet_noise-0.2_k-12_run-0/all.txt")
data2bb = format_glob_pd("../logs_2023-06-01/resnet-50epochs_control2/classes-10/noise-0.2/epoch-*/model-resnet_noise-0.2_k-12_run-0/all.txt")
retrain_error = []
retrain_error2 = []
retrain_error2b = []
retrain_epochs = np.array([0, 2, 9, 29, 99, 299, 999, 2999])
for i, file in data2.iterrows():
    d = np.loadtxt(file.filename)
    retrain_error.append(d[-1, 0])
    retrain_error2.append(np.loadtxt(data2b.iloc[i].filename)[-1, 0])
    retrain_error2b.append(np.loadtxt(data2bb.iloc[i].filename)[-1, 0])
print(d.shape, d[-1, 0])
plt.plot(retrain_epochs[:len(retrain_error)]+1, 1-np.array(retrain_error)/100, "--oC1", label="retrain", mec="w")
plt.plot(retrain_epochs+1, 1-np.array(retrain_error2)/100, ":oC1", label="control", mec="w")
plt.plot(retrain_epochs+1, 1-np.array(retrain_error2b)/100, "-oC1", label="control2", mec="w")
retrain_error12 = retrain_error

plt.sca(subplots[0, 0])
data2 = format_glob_pd("../logs_2023-05-19/resnet-50epochs/classes-10/noise-0.2/epoch-*/model-resnet_noise-0.2_k-3_run-0/all.txt")
data2b = format_glob_pd("../logs_2023-05-25/resnet-50epochs_control/classes-10/noise-0.2/epoch-*/model-resnet_noise-0.2_k-3_run-0/all.txt")
data2bb = format_glob_pd("../logs_2023-06-01/resnet-50epochs_control2/classes-10/noise-0.2/epoch-*/model-resnet_noise-0.2_k-3_run-0/all.txt")
retrain_error = []
retrain_error2 = []
retrain_error2b = []
retrain_epochs = np.array([0, 2, 9, 29, 99, 299, 999, 2999])
for i, file in data2.iterrows():
    d = np.loadtxt(file.filename)
    retrain_error.append(d[-1, 0])
    retrain_error2.append(np.loadtxt(data2b.iloc[i].filename)[-1, 0])
    retrain_error2b.append(np.loadtxt(data2bb.iloc[i].filename)[-1, 0])
print(d.shape, d[-1, 0])
plt.plot(retrain_epochs[:len(retrain_error)]+1, 1-np.array(retrain_error)/100, "--oC0", label="retrain", mec="w")
plt.plot(retrain_epochs+1, 1-np.array(retrain_error2)/100, ":oC0", label="control", mec="w")
plt.plot(retrain_epochs+1, 1-np.array(retrain_error2b)/100, "-oC0", label="control", mec="w")
retrain_error3 = retrain_error
plt.legend()



fig, subplots = plt.subplots(1, 1, sharex=True)
plt.sca(subplots)

for epoch_id, (noise, d) in enumerate(data.groupby("k")):
    if d.iloc[0].k == 3:
        retrain_error = retrain_error3
    if d.iloc[0].k == 12:
        retrain_error = retrain_error12
    if d.iloc[0].k == 64:
        retrain_error = retrain_error64
    acc = np.loadtxt(Path(d.iloc[0].filename).parent / "all.txt")

    accuracy = []
    for i in noise_unique:
        accuracy.append(acc[i, 0])

    from helpers import plot_color_grad
    plot_color_grad(steps[epoch_id, :], 1 - np.array(retrain_error)[:] / 100, marker="o", color1=f"C{epoch_id}", ms=2)
    #plt.plot(steps[epoch_id, :], 1 - np.array(accuracy)[:] / 100, '-o')
    plt.plot(steps[epoch_id, :][0], 1 - np.array(retrain_error)[0] / 100, '+k', ms=5)
plt.gca().set(xlabel="step", ylabel="retrain val error")
plt.ylim(0, 1)
plt.xlim(0, plt.gca().get_xlim()[1])
plt.suptitle(f"{N} {model}")
plt.show()

