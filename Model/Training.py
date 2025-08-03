import numpy as np
import pandas as pd
from google.colab import drive

import torch
from torch import nn, optim
# from torch import functional as F 

from matplotlib import pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import transform
from torchmetrics.classification import Accuracy


training_data=pd.read_csv('/content/drive/MyDrive/Data/emnist-balanced-train.csv', header=None)
training_data.columns=['label']+[f'pixel_{i}' for i in range(784)]

data_point=training_data.iloc[3,1:].values.reshape(28,28)
plt.imshow(data_point, cmap='gray')


### MODEL CLASS ###
                    # NOTE : This isn't that Comlex NN. The Complex NN is going to come soon.
class Neural_Network(nn.Module):
    def __init__(self, lr: float = 0.00001, epochs: int = 1000, visualize=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.visualize = visualize
        self.learning_rate = lr
        self.epochs = epochs
        self.losses = []
        self.epoch_accuracies = []

        # CNN Layers
        self.cnn_layer1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.cnn_layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.cnn_layer3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.cnn_layer4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=47)  # 47 classes

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.metric = Accuracy(task='multiclass', num_classes=47).to(self.device)

        self.to(self.device)

    def forward(self, x):
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = self.pool1(x)
        
        x = self.cnn_layer3(x)
        x = self.cnn_layer4(x)            ### Model Structure (Not so Complex)
        x = self.pool2(x)
        
        x = x.reshape(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

    def fit(self, train_loader):
        
        self.train_loader = train_loader
        self.total_steps = len(train_loader)

        for epoch in range(self.epochs):
            self.metric.reset()

            for i, (images, labels) in enumerate(self.train_loader):            ### Training Process ###
                images = images.view(-1, 1, 28, 28).to(self.device)
                labels = labels.to(self.device).long()

                outputs = self.forward(images)            
                self.loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.to(self.device).long()

                # Update metric
                self.metric.update(predicted, labels)
                self.losses.append(self.loss.item())

            # Compute accuracy
            acc = self.metric.compute()
            if acc is not None:
                acc_val = acc.detach().cpu().item()
                self.epoch_accuracies.append(acc_val)
            else:
                print(f"[WARNING] Epoch {epoch+1}: Accuracy returned None.")
                acc_val = 0.0
                self.epoch_accuracies.append(acc_val)

            print(f"Epoch {epoch + 1}/{self.epochs}    Loss ~ {self.loss.item():.6f}  Accuracy ~ {acc_val:.6f}")
            # print(f"[DEBUG] Metric Updates Complete — Accuracy: {acc_val:.4f}, Batches: {len(train_loader)}")
            # print(f"[DEBUG] Accuracies Collected: {self.epoch_accuracies}")
            

        if self.visualize:
            self._plot_loss()

    def _plot_loss(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.lineplot(data=self.losses, label='Training Loss')
        plt.title("Training Loss")

        if self.epoch_accuracies:
            plt.subplot(1,2,2)
            sns.lineplot(data=self.epoch_accuracies, marker='o', label='Training Accuracy')
        
        plt.tight_layout()
        plt.legend()
        plt.show()


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
# Epoch 1/15    Loss ~ 0.516033  Accuracy ~ 0.678688
# Epoch 2/15    Loss ~ 0.381906  Accuracy ~ 0.823360
# Epoch 3/15    Loss ~ 0.342590  Accuracy ~ 0.847553
# Epoch 4/15    Loss ~ 0.362492  Accuracy ~ 0.860124
# Epoch 5/15    Loss ~ 0.285667  Accuracy ~ 0.868936
# Epoch 6/15    Loss ~ 0.422216  Accuracy ~ 0.877429
# Epoch 7/15    Loss ~ 0.324237  Accuracy ~ 0.883910        ### OverALL : 92% in just 15 Epochs
# Epoch 8/15    Loss ~ 0.260086  Accuracy ~ 0.889512        
# Epoch 9/15    Loss ~ 0.157135  Accuracy ~ 0.894681
# Epoch 10/15   Loss ~ 0.302242  Accuracy ~ 0.900186        ### Note! This is Training Result...
# Epoch 11/15   Loss ~ 0.297627  Accuracy ~ 0.904043        ### Information : The Resulting Plots are available in "Training_Results.pdf"
# Epoch 12/15   Loss ~ 0.381384  Accuracy ~ 0.909051
# Epoch 13/15   Loss ~ 0.384912  Accuracy ~ 0.913280
# Epoch 14/15   Loss ~ 0.225305  Accuracy ~ 0.917101
# Epoch 15/15   Loss ~ 0.146838  Accuracy ~ 0.922207

