import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_context('talk')

files = [
#	'output/v100_fullmodel/results_lenet_EMNIST_batchsize1024_AMP',
#	'output/v100_fullmodel/results_lenet_EMNIST_batchsize1024_Full',
#	'output/v100_fullmodel/results_lenet_EMNIST_batchsize1024_Half',
	'output/v100_fullmodel/results_lenet_EMNIST_batchsize128_AMP',
	'output/v100_fullmodel/results_lenet_EMNIST_batchsize128_Full',
	'output/v100_fullmodel/results_lenet_EMNIST_batchsize128_Half',
	'output/v100_fullmodel/results_lenet_EMNIST_batchsize256_AMP',
	'output/v100_fullmodel/results_lenet_EMNIST_batchsize256_Full',
	'output/v100_fullmodel/results_lenet_EMNIST_batchsize256_Half',
	'output/v100_fullmodel/results_lenet_EMNIST_batchsize512_AMP',
	'output/v100_fullmodel/results_lenet_EMNIST_batchsize512_Full',
	'output/v100_fullmodel/results_lenet_EMNIST_batchsize512_Half',
	'output/v100_fullmodel/results_lenet_EMNIST_batchsize64_AMP',
	'output/v100_fullmodel/results_lenet_EMNIST_batchsize64_Full',
	'output/v100_fullmodel/results_lenet_EMNIST_batchsize64_Half',
	'output/v100_fullmodel/results_resnet50_Cifar10_batchsize128_AMP',
	'output/v100_fullmodel/results_resnet50_Cifar10_batchsize128_Full',
	'output/v100_fullmodel/results_resnet50_Cifar10_batchsize128_Half',
	'output/v100_fullmodel/results_resnet50_Cifar10_batchsize256_AMP',
	'output/v100_fullmodel/results_resnet50_Cifar10_batchsize256_Full',
	'output/v100_fullmodel/results_resnet50_Cifar10_batchsize256_Half',
	'output/v100_fullmodel/results_resnet50_Cifar10_batchsize512_AMP',
	'output/v100_fullmodel/results_resnet50_Cifar10_batchsize512_Half',
	'output/v100_fullmodel/results_resnet50_Cifar10_batchsize64_AMP',
	'output/v100_fullmodel/results_resnet50_Cifar10_batchsize64_Full',
	'output/v100_fullmodel/results_resnet50_Cifar10_batchsize64_Half',
#	'output/v100_fullmodel/results_vgg16_Cifar10_batchsize1024_AMP',
#	'output/v100_fullmodel/results_vgg16_Cifar10_batchsize1024_Full',
#	'output/v100_fullmodel/results_vgg16_Cifar10_batchsize1024_Half',
	'output/v100_fullmodel/results_vgg16_Cifar10_batchsize128_AMP',
	'output/v100_fullmodel/results_vgg16_Cifar10_batchsize128_Full',
	'output/v100_fullmodel/results_vgg16_Cifar10_batchsize128_Half',
	'output/v100_fullmodel/results_vgg16_Cifar10_batchsize256_AMP',
	'output/v100_fullmodel/results_vgg16_Cifar10_batchsize256_Full',
	'output/v100_fullmodel/results_vgg16_Cifar10_batchsize256_Half',
	'output/v100_fullmodel/results_vgg16_Cifar10_batchsize512_AMP',
	'output/v100_fullmodel/results_vgg16_Cifar10_batchsize512_Full',
	'output/v100_fullmodel/results_vgg16_Cifar10_batchsize512_Half',
	'output/v100_fullmodel/results_vgg16_Cifar10_batchsize64_AMP',
	'output/v100_fullmodel/results_vgg16_Cifar10_batchsize64_Full',
	'output/v100_fullmodel/results_vgg16_Cifar10_batchsize64_Half'
]

data = []
data.append(['model', 'dataset', 'batch_size', 'precision', 'avg_epoch_train_time'])
for file in files:
  tmp = file.replace('output/v100_fullmodel/results_' ,'')
  tmp = tmp.split('_')
  model = tmp[0]
  dataset = tmp[1]
  batch_size = int(tmp[2].replace('batchsize',''))
  precision = tmp[3]
  df=pd.read_csv(file, sep=',',header=0)
  df = df.iloc[1:] # delete the first (re-training) row (epoch 0)
  avg_train_time = df['epoch_train_time'].mean()
  data.append([model, dataset, batch_size, precision, avg_train_time])

df = pd.DataFrame(data[1:],columns=data[0])
#print(df)
orders = ['Full', 'AMP', 'Half']
df_lenet = df[(df['model']=="lenet") & (df['dataset']=="EMNIST")]
df_lenet = df_lenet.pivot(index='batch_size', columns='precision', values='avg_epoch_train_time')
df_lenet = df_lenet[orders]
df_vgg16 = df[(df['model']=="vgg16") & (df['dataset']=="Cifar10")]
df_vgg16 = df_vgg16.pivot(index='batch_size', columns='precision', values='avg_epoch_train_time')
df_vgg16 = df_vgg16[orders]
df_resnet = df[(df['model']=="resnet50") & (df['dataset']=="Cifar10")]
df_resnet = df_resnet.pivot(index='batch_size', columns='precision', values='avg_epoch_train_time')
df_resnet = df_resnet[orders]
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(16,7))
styles = ['-',':','--']
colors = ['red', 'green', 'blue']
df_lenet.plot(ax=axes[0], style=styles, color=colors, grid=True); axes[0].set_title('LeNet & EMNIST')
df_vgg16.plot(ax=axes[1], style=styles, color=colors, grid=True); axes[1].set_title('VGG-16 & Cifar10')
df_resnet.plot(ax=axes[2], style=styles, color=colors, grid=True); axes[2].set_title('ResNet50 & Cifar10')
plt.xticks([64,128,256,512])
axes[0].set_xlabel("Batch Size")
axes[1].set_xlabel("Batch Size")
axes[2].set_xlabel("Batch Size")
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
