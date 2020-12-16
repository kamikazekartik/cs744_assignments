import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
sns.set_context('talk')

files = [
	'output/T4_fullmodel/results_resnet50_Cifar10_batchsize128_AMP',
	'output/T4_fullmodel/results_resnet50_Cifar10_batchsize128_Full',
	'output/T4_fullmodel/results_resnet50_Cifar10_batchsize128_Half',
	'output/v100_fullmodel/results_resnet50_Cifar10_batchsize128_AMP',
	'output/v100_fullmodel/results_resnet50_Cifar10_batchsize128_Full',
	'output/v100_fullmodel/results_resnet50_Cifar10_batchsize128_Half',
]

data = []
data.append(['gpu', 'precision', 'avg_epoch_train_time'])
for file in files:
  gpu = 'T4'
  if 'v100' in file:
    gpu = 'v100'
  tmp = file.replace('output/v100_fullmodel/results_' ,'').replace('output/T4_fullmodel/results_' ,'')
  tmp = tmp.split('_')
  model = tmp[0]
  dataset = tmp[1]
  batch_size = int(tmp[2].replace('batchsize',''))
  precision = tmp[3]
  df=pd.read_csv(file, sep=',',header=0)
  df = df.iloc[1:] # delete the first (re-training) row (epoch 0)
  avg_train_time = df['epoch_train_time'].mean()
  data.append([gpu, precision, avg_train_time])

df = pd.DataFrame(data[1:],columns=data[0])
cat_precision_order = CategoricalDtype(['Full', 'AMP', 'Half'], ordered=True)
df['precision'] = df['precision'].astype(cat_precision_order)
df = df.sort_values('precision')
df_resnet_T4 = df[(df['gpu']=="T4")]
print(df_resnet_T4.head())
df_resnet_v100 = df[(df['gpu']=="v100")]
print(df_resnet_v100.head())
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(9,8))
colors = ['red', 'green', 'blue']
df_resnet_T4.plot(ax=axes[0], x='precision', y='avg_epoch_train_time', kind='bar', color=colors); axes[0].set_title('T4')
df_resnet_v100.plot(ax=axes[1], x='precision', y='avg_epoch_train_time', kind='bar', color=colors); axes[1].set_title('v100')
axes[0].set_xlabel("Precision")
axes[0].get_legend().remove()
axes[1].set_xlabel("Precision")
axes[1].get_legend().remove()
axes[0].set_ylabel("Per-Epoch Training Time (s)")
plt.show()

#df['settng'] = df.model + "\n" + df.dataset + "\n" + df.precision
#df['tmp'] = df.batch_size + df.precision
#df = df[['settng', 'tmp', 'batch_size', 'avg_epoch_train_time']]

#fig, ax = plt.subplots(figsize=(12, 8))
#x = np.arange(len(df.settng.unique()))
#bar_width = 0.25
#b1 = ax.bar(x, df.loc[df['batch_size']=='64', 'avg_epoch_train_time'], width=bar_width, label='batch size = 64')
#b2 = ax.bar(x + (1*bar_width), df.loc[df['tmp']=='128Half', 'avg_epoch_train_time'], width=bar_width, label='batch size = 128')
#b3 = ax.bar(x + (2*bar_width), df.loc[df['tmp']=='128AMP', 'avg_epoch_train_time'], width=bar_width, label='batch size = 256')
#b4 = ax.bar(x + (3*bar_width), df.loc[df['batch_size']=='512', 'avg_epoch_train_time'], width=bar_width, label='batch size = 512')
#b5 = ax.bar(x + (4*bar_width), df.loc[df['batch_size']=='1024', 'avg_epoch_train_time'], width=bar_width, label='batch size = 1024')
#ax.set_xticks(x + bar_width / 2)
#ax.set_xticklabels(df.settng.unique(), rotation=90)
#ax.legend()
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#ax.spines['left'].set_visible(False)
#ax.spines['bottom'].set_color('#DDDDDD')
#ax.tick_params(bottom=False, left=False)
#ax.set_axisbelow(True)
#ax.yaxis.grid(True, color='#EEEEEE')
#ax.xaxis.grid(False)
#ax.set_xlabel('Setting & Batch Size', labelpad=15)
#ax.set_ylabel('Avg. Train Time (s)', labelpad=15)
#ax.set_title('Average Training Time per Epoch by Setting and Batch Size', pad=15)
#fig.tight_layout()
#plt.show()
