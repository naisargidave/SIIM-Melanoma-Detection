# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:28:37 2020

@author: naisa
"""

import pandas as pd
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from efficientnet_pytorch import EfficientNet

class SIIMDataset(Dataset):
    def __init__(self, base_dir, train_test):
        self.base_dir = base_dir
        self.train_test = train_test
        
        csv_file = train_test + '.csv'
        csv_path = os.path.join(base_dir, csv_file)
#        print('csv_path = ',csv_path)
        self.csv_data = pd.read_csv(csv_path)
#        print(csv_data.head())
        if train_test == 'train':
            self.transform = self.train_transform_image()
            self.train_preprocess()
        else:
            self.transform = self.test_transform_image()
            
        self.length = len(self.csv_data)
        
    def preprocess(self):
        
        self.preprocessed_data = self.csv_data.copy()    
    
    def train_preprocess(self):
        
        data_subset_1 = self.csv_data[self.csv_data['target'] == 1]

        data_subset_0 = self.csv_data[self.csv_data['target'] == 0].sample(len(data_subset_1)) 

        self.csv_data = pd.concat([data_subset_1,data_subset_0])
    
    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        img_name =  self.csv_data.iloc[index, 0] + '.jpg'
        label = self.csv_data.iloc[index, -1]
        img_path = os.path.join(self.base_dir, self.train_test, 'jpeg', img_name)
#        print(img_path)
        image = Image.open(img_path).convert('RGB')
#        image.show(image)
        return {'image' : self.transform(image), 'target': label, 'name': self.csv_data.iloc[index, 0]}
    
    def train_transform_image(self):
        transform_list = []
        transform_list.append(transforms.Resize((256, 256)))
        transform_list.append(transforms.ToTensor())
        return transforms.Compose(transform_list)
    
    def test_transform_image(self):
        transform_list = []
        transform_list.append(transforms.Resize((256, 256)))
        transform_list.append(transforms.ToTensor())
        return transforms.Compose(transform_list)
    
class MelanomaDetectionModel(nn.Module):
    def __init__(self):
        super(MelanomaDetectionModel, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        num_features = self.model._fc.in_features
        self.model._fc = nn.Linear(num_features, 1)
        
    def forward(self, data):
        return self.model(data)
        
class TrainModel:
    
    def __init__(self):
        self.train_data = SIIMDataset(base_dir = 'C:\\Users\\naisa\\Desktop\\SIIM-ISIC Melanoma', train_test = 'train')
        self.test_data = SIIMDataset(base_dir = 'C:\\Users\\naisa\\Desktop\\SIIM-ISIC Melanoma', train_test = 'test')
        self.batch_size = 16
        self.test_batch_size = 1

        self.train_dl = DataLoader(self.train_data, self.batch_size, shuffle=True)
        self.test_dl = DataLoader(self.test_data, self.test_batch_size, shuffle = False)
        
        self.device = torch.device('cuda')
        
        self.network = MelanomaDetectionModel().to(self.device)
#        print(self.network)
        
        self.learning_rate = 0.001
        self.optimizer = optim.Adam(self.network.parameters(), lr = self.learning_rate)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = 0.5, verbose = True, 
                                                              cooldown = 2, min_lr = 1e-5)
        
        self.loss_criterion = nn.BCEWithLogitsLoss()
        
        self.counter = 0 

        self.save_network(0)
        
    def train(self, epoch):
        start_epoch = (epoch // 5) * 5
        self.load_network(start_epoch)
        start_epoch += 1
        self.writer = SummaryWriter(comment = '-resnet-' + str(self.learning_rate))

        for epoch in range(start_epoch, 41):
            accuracy = []
            for i, data in enumerate(self.train_dl):
                model_input = data['image'].to(self.device)
                label = data['target'].to(self.device).view(-1,1).float()
                
                network_output = self.network(model_input)
                loss = self.loss_criterion(network_output, label)
                self.counter += 1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print("Epoch : %d, Iteration : %d, Loss : %f"
                    % (epoch, i, loss.item()))
                
                self.writer.add_scalar('loss', loss.item(), self.counter)
                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], self.counter) 
                
                prediction = torch.round(torch.sigmoid(network_output))
                correct_preds = (prediction==label).sum().float()
                accuracy_per_batch = correct_preds/self.batch_size
                accuracy.append(accuracy_per_batch.item())
            if epoch % 5 == 0:
                self.save_network(epoch)
                
            accuracy = torch.Tensor(accuracy)    
            avg_accuracy = torch.mean(accuracy)
            self.scheduler.step(avg_accuracy)
    def save_network(self, epoch):
        torch.save(self.network.state_dict(), str(epoch) + '-network.pth')

    def load_network(self, epoch):
        self.network.load_state_dict(torch.load(str(epoch) + '-network.pth'))
    
    def test(self, epoch):
        self.load_network(epoch)
        self.network.eval()

        names = []
        predictions = []
        
        for i, data in enumerate(self.test_dl):
            
            model_input = data['image'].to(self.device)
            file_name = data['name'][0]
            
            network_output = self.network(model_input)
            prediction = torch.round(torch.sigmoid(network_output))
            prediction = int(prediction.item())

            names.append(file_name)
            predictions.append(prediction)
            
            
            print("index : %d, Prediction : %d"
                  % (i, prediction))
            
        df = pd.DataFrame(list(zip(names, predictions)), 
               columns =['image_name', 'target']) 
        df.to_csv('submission.csv')
    
        
        
t = TrainModel()
t.train(1)

#t.save_network()

t.test(40)
        