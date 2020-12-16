import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_context('talk')

files = [
	'output/v100_fullmodel_2/results_vgg11_Cifar10_batchsize128_Full.csv',
	'output/v100_fullmodel_2/results_vgg11_Cifar10_batchsize128_AMP.csv',
	'output/v100_fullmodel_2/results_vgg11_Cifar10_batchsize128_Half.csv',
	'output/v100_fullmodel_2/results_vgg16_Cifar10_batchsize128_Full.csv',
	'output/v100_fullmodel_2/results_vgg16_Cifar10_batchsize128_AMP.csv',
	'output/v100_fullmodel_2/results_vgg16_Cifar10_batchsize128_Half_fixed.csv',
	'output/v100_convergence/results_resnet50_Cifar10_batchsize128_Full.csv',
	'output/v100_convergence/results_resnet50_Cifar10_batchsize128_AMP.csv',
	'output/v100_convergence/results_resnet50_Cifar10_batchsize128_Half.csv'
]

data = []
data.append(['model', 'dataset', 'precision', 'epoch', 'training_loss', 'test_acc', 'epoch_train_time', 'total_train_time'])
for file in files:
  tmp = file.replace('output/v100_fullmodel_2/results_' ,'').replace('output/v100_convergence/results_' ,'').replace('_fixed.csv','').replace('.csv','')
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
orders = ['Full', 'AMP', 'Half']
styles = ['-',':','--']
colors = ['red', 'green', 'blue']
df_vgg11 = df[(df['model']=="vgg11") & (df['dataset']=="Cifar10") & (df['test_acc']!=-1)]
df_vgg11 = df_vgg11.pivot(index='epoch', columns='precision', values='test_acc')
df_vgg11 = df_vgg11[orders]
df_vgg16 = df[(df['model']=="vgg16") & (df['dataset']=="Cifar10") & (df['test_acc']!=-1)]
df_vgg16 = df_vgg16.pivot(index='epoch', columns='precision', values='test_acc')
df_vgg16 = df_vgg16[orders]
df_resnet50 = df[(df['model']=="resnet50") & (df['dataset']=="Cifar10") & (df['test_acc']!=-1)]
df_resnet50 = df_resnet50.pivot(index='epoch', columns='precision', values='test_acc')
df_resnet50 = df_resnet50[orders]
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(20,7))
df_vgg11.plot(ax=axes[0], style=styles, color=colors, grid=True); axes[0].set_title('VGG-11 & Cifar10')
df_vgg16.plot(ax=axes[1], style=styles, color=colors, grid=True); axes[1].set_title('VGG-16 & Cifar10')
df_resnet50.plot(ax=axes[2], style=styles, color=colors, grid=True); axes[2].set_title('ResNet50 & Cifar10')
axes[0].set_xlabel("Epoch")
axes[1].set_xlabel("Epoch")
axes[2].set_xlabel("Epoch")
axes[0].set_ylabel("Test Accuracy (%)")
plt.show()



plt.clf()
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(20,7))

df_vgg11 = df[(df['model']=="vgg11") & (df['dataset']=="Cifar10") & (df['test_acc']!=-1)]
labels = ['Full', 'AMP', 'Half']
for key in labels:
  if key=='Full':
    color = 'red'
    style = '-'
  elif key=='AMP':
    color = 'green'
    style = ':'
  else:
    color = 'blue'
    style = '--'
  ax = df_vgg11[(df['precision']==key)].plot(ax=axes[0], kind='line', x='total_train_time', y='test_acc', label=key, style=style, color=color, grid=True)
axes[0].set_title('VGG-11 & Cifar10')
axes[0].set_xlabel("Training Time (s)")
axes[0].set_ylabel("Test Accuracy (%)")

df_vgg16 = df[(df['model']=="vgg16") & (df['dataset']=="Cifar10") & (df['test_acc']!=-1)]
for key in labels:
  if key=='Full':
    color = 'red'
    style = '-'
  elif key=='AMP':
    color = 'green'
    style = ':'
  else:
    color = 'blue'
    style = '--'
  ax = df_vgg16[(df['precision']==key)].plot(ax=axes[1], kind='line', x='total_train_time', y='test_acc', label=key, style=style, color=color, grid=True)
axes[1].set_title('VGG-16 & Cifar10')
axes[1].set_xlabel("Training Time (s)")

df_resnet50 = df[(df['model']=="resnet50") & (df['dataset']=="Cifar10") & (df['test_acc']!=-1)]
for key in labels:
  if key=='Full':
    color = 'red'
    style = '-'
  elif key=='AMP':
    color = 'green'
    style = ':'
  else:
    color = 'blue'
    style = '--'
  ax = df_resnet50[(df['precision']==key)].plot(ax=axes[2], kind='line', x='total_train_time', y='test_acc', label=key, style=style, color=color, grid=True)
axes[2].set_title('ResNet50 & Cifar10')
axes[2].set_xlabel("Training Time (s)")

plt.show()
