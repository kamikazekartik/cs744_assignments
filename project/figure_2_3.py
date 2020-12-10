import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_context('talk')

files = [
	'output/v100_fullmodel_2/results_vgg11_Cifar10_batchsize128_Full',
	'output/v100_fullmodel_2/results_vgg11_Cifar10_batchsize128_AMP',
	'output/v100_fullmodel_2/results_vgg11_Cifar10_batchsize128_Half',
	'output/v100_fullmodel_2/results_vgg16_Cifar10_batchsize128_Full',
	'output/v100_fullmodel_2/results_vgg16_Cifar10_batchsize128_AMP',
	'output/v100_fullmodel_2/results_vgg16_Cifar10_batchsize128_Half'
]

data = []
data.append(['model', 'dataset', 'precision', 'epoch', 'training_loss', 'test_acc', 'epoch_train_time', 'total_train_time'])
for file in files:
  tmp = file.replace('output/v100_fullmodel_2/results_' ,'')
  tmp = tmp.split('_')
  model = tmp[0]
  dataset = tmp[1]
  batch_size = int(tmp[2].replace('batchsize',''))
  precision = tmp[3]
  df=pd.read_csv(file, sep=',',header=0)
  df = df.iloc[1:] # delete the first (re-training) row (epoch 0)
  for i, row in df.iterrows():
    epoch = int(row[1])
    training_loss = float(row[2])
    test_acc = float(row[3])
    epoch_train_time = float(row[4])
    total_train_time = float(row[5])
    data.append([model, dataset, precision, epoch, training_loss, test_acc, epoch_train_time, total_train_time])

df = pd.DataFrame(data[1:],columns=data[0])
df_vgg11 = df[(df['model']=="vgg11") & (df['dataset']=="Cifar10") & (df['test_acc']!=-1)]
df_vgg11 = df_vgg11.pivot(index='epoch', columns='precision', values='test_acc')
df_vgg16 = df[(df['model']=="vgg16") & (df['dataset']=="Cifar10") & (df['test_acc']!=-1)]
df_vgg16 = df_vgg16.pivot(index='epoch', columns='precision', values='test_acc')
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(14,7))
df_vgg11.plot(ax=axes[0]); axes[0].set_title('VGG-11 & Cifar10')
df_vgg16.plot(ax=axes[1]); axes[1].set_title('VGG-16 & Cifar10')
axes[0].set_xlabel("Epoch")
axes[1].set_xlabel("Epoch")
axes[0].set_ylabel("Test Accuracy (%)")
plt.show()

plt.clf()
df_vgg11 = df[(df['model']=="vgg11") & (df['dataset']=="Cifar10") & (df['test_acc']!=-1)]
#df_vgg11 = df_vgg11.pivot(index='epoch_train_time', columns='precision', values='test_acc')
df_vgg16 = df[(df['model']=="vgg16") & (df['dataset']=="Cifar10") & (df['test_acc']!=-1)]
#df_vgg16 = df_vgg16.pivot(index='epoch_train_time', columns='precision', values='test_acc')
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(14,7))
for key, grp in df_vgg11.groupby(['precision']):
  ax = grp.plot(ax=axes[0], kind='line', x='total_train_time', y='test_acc', label=key)
for key, grp in df_vgg16.groupby(['precision']):
  ax = grp.plot(ax=axes[1], kind='line', x='total_train_time', y='test_acc', label=key)
axes[0].set_title('VGG-11 & Cifar10')
axes[1].set_title('VGG-16 & Cifar10')
axes[0].set_xlabel("Training Time (s)")
axes[1].set_xlabel("Training Time (s)")
axes[0].set_ylabel("Test Accuracy (%)")
plt.show()
