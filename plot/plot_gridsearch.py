import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Plot results of gridsearch over clip / lr")
parser.add_argument('-f', '--file')
parser.add_argument('-T', '--epoch', default=10)
parser.add_argument('--max-epoch', default=None)
args = parser.parse_args()


epoch = int(args.epoch)
max_epoch = args.max_epoch
max_epoch = epoch if (max_epoch is None) else int(max_epoch)
filename = args.file


df = pd.read_csv(filename)
df = df[df['epoch'] == epoch]
df = df[df['total_epochs'] == epoch]
df = df.sort_values("base_lr")

for c in sorted(df['max_grad_norm'].unique()):
    dd = df[df['max_grad_norm'] == c]
    l, = plt.plot(dd['base_lr'], dd['train_acc'], '-o', label=f"C={c:0.2f}")

plt.xlabel("Learning rate")
plt.ylabel("Final train accuracy")
plt.title(f"LR scans by value of clipping C ({epoch} epochs)")
plt.yticks([ i / 100 for i in range(10, 70, 5) ], minor=True)
plt.xscale('log')
plt.legend()
plt.grid(which='major', alpha=0.8)
plt.grid(which='minor', alpha=0.2)
plt.show()
