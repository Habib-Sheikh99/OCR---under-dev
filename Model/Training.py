import numpy as np
import pandas as pd
from google.colab import drive

import torch
from torch import nn, optim
from torch import functional as F

from matplotlib import pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision.transforms import transforms
from torchmetrics.classification import Accuracy


training_data=pd.read_csv('/content/drive/MyDrive/Data/emnist-balanced-train.csv', header=None)
training_data.columns=['label']+[f'pixel_{i}' for i in range(784)]

data_point=training_data.iloc[3,1:].values.reshape(28,28)
plt.imshow(data_point, cmap='gray')


### MODEL CLASS ###


class Neural_Network(nn.Module):
    def __init__(self, lr:float=0.00001, epochs:int=1000, visualize=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.visualize=visualize
        self.learning_rate=lr
        self.epochs=epochs
        self.losses=[]
        self.history=[]

        self.cnn_layer1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.cnn_layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.cnn_layer3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.cnn_layer4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1=nn.Linear(in_features=64*4*4, out_features=128)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(in_features=128, out_features=47) # 47 is the number of classes


        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.batch_size=64
        self.num_classes=47
        self.total_step=None

        self.metric = Accuracy(task='multiclass', num_classes=47).to(self.device)
        self.to(self.device)
  
    def forward(self, x):
        output = self.cnn_layer1(x)
        output=self.cnn_layer2(output)

        output=self.pool1(output)

        output=self.cnn_layer3(output)
        output=self.cnn_layer4(output)

        output=self.pool2(output)

        output = output.reshape(output.size(0), -1)
        # print("X-shape : ", output.shape) ### Un-Comment to confirm the dimension of image (if there is an error, Obviously!)
        output = self.fc1(output)
        output = self.relu(output)
        output=self.fc2(output)

        return output

    def _plot_loss(self):
                    sns.lineplot(self.losses)
                    plt.show()
          
    def fit(self, train_loader):

        self.train_loader = train_loader
        self.total_steps = len(train_loader)
        self.epoch_accuracies=[]
    
        for epoch in range(self.epochs):
    
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.view(-1, 1 , 28, 28).to(self.device)
                labels = labels.to(images.device)

                outputs = self.forward(images)
                self.loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                _ , predicted = torch.max(outputs.data, 1)
                self.metric.update(predicted, labels)
                self.losses.append(self.loss.item())

            self.accuracy = self.metric.compute()
            self.metric.reset()
            print(f"Epoch {epoch+1}/{self.epochs}    Loss ~ {self.loss.item():.6f}  Accuracy ~ {self.accuracy:.6f}")

        if self.visualize:
            self._plot_loss()




### Training ###

x_train = training_data.drop('label', axis=1)
y_train = training_data['label']
# y_train.value_counts()

xtrain = x_train.to_numpy().astype(np.float32)
ytrain = y_train.to_numpy().astype(np.float32)


print(len(ytrain))

training_sets = TensorDataset(
    torch.from_numpy(xtrain),
    torch.from_numpy(ytrain).long()
    )

train_loader = DataLoader(
                            training_sets,
                            batch_size=64,
                            shuffle=True
                        )




model = Neural_Network(visualize=True, lr=0.0001, epochs=10)
model.fit(train_loader)

### RESULT :
# Epoch 1/10    Loss ~ 0.552683  Accuracy ~ 0.681534
# Epoch 2/10    Loss ~ 0.405365  Accuracy ~ 0.820966
# Epoch 3/10    Loss ~ 0.349056  Accuracy ~ 0.846543
# Epoch 4/10    Loss ~ 0.288678  Accuracy ~ 0.859406
# Epoch 5/10    Loss ~ 0.237350  Accuracy ~ 0.868847        OverALL : 90% after just 10 epochs    
# Epoch 6/10    Loss ~ 0.361403  Accuracy ~ 0.877163        
# Epoch 7/10    Loss ~ 0.630927  Accuracy ~ 0.884743        Note! This is Training result.
# Epoch 8/10    Loss ~ 0.389284  Accuracy ~ 0.888626
# Epoch 9/10    Loss ~ 0.472142  Accuracy ~ 0.893218
# Epoch 10/10   Loss ~ 0.259637  Accuracy ~ 0.898768


