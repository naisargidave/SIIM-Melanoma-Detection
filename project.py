# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:28:37 2020

@author: naisa
"""

import pandas as pd
from PIL import Image
import os
import cv2
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
    
    def __init__(self, base_dir, train_test, data):
        self.csv_data = data
        self.base_dir = base_dir
        self.train_test = train_test
        self.col_name = [x for x in self.csv_data.columns if 'sex' in x or 'anatom' in x or 'approx' in x]

        
        if train_test == 'train':
            self.transform = self.train_transform_image()
        else:
            self.transform = self.test_transform_image()
            
        self.length = len(self.csv_data)
        
    
    def __len__(self):
        return self.length
#        return 1
        
    def __getitem__(self, index):
        img_name =  self.csv_data.iloc[index]['image_name'] + '.jpg'
        if self.train_test == 'train':
            label = self.csv_data.iloc[index]['target']
        else:
            label = []
        img_path = os.path.join(self.base_dir, self.train_test, 'jpeg', img_name)
#        image = Image.open(img_path).convert('RGB')
        image = cv2.imread(img_path)
        

        return {'image' : self.transform(image), 'target': label, 
                'name': self.csv_data.iloc[index]['image_name'],
                'metadata': torch.Tensor(np.array(self.csv_data.iloc[index][self.col_name], dtype = np.int8))}
    
    def train_transform_image(self):
        transform_list = [CenterCrop(0.5),
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])]
        return transforms.Compose(transform_list)
    
    def test_transform_image(self):
        transform_list = []
        transform_list.append(CenterCrop(0.5))
        transform_list.append(transforms.Resize((256, 256)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225]))
        return transforms.Compose(transform_list)
    
class MelanomaDetectionModel(nn.Module):
    def __init__(self, n_features):
        super(MelanomaDetectionModel, self).__init__()
        cnn_out_features = 600
        metadata_out_features = 150
        
        self.model = EfficientNet.from_pretrained('efficientnet-b2')
        num_features = self.model._fc.in_features
        self.model._fc = nn.Linear(num_features, cnn_out_features)
        
        self.metadata_features = nn.Sequential(
                    nn.Linear(n_features, 400),
                    nn.ReLU(),
                    nn.Linear(400, metadata_out_features),
                    nn.ReLU()
                )
        self.final_layer = nn.Linear(cnn_out_features + metadata_out_features, 1)
        
    def forward(self, images, metadata):
        images_out = self.model(images)
        metadata_out = self.metadata_features(metadata)
        intermediate = torch.cat([images_out, metadata_out], dim = 1)
        return self.final_layer(intermediate)

class CenterCrop:
    
    def __init__(self, p):
        self.p = p
        
    def __call__(self,image):
        if np.random.random() < self.p :
            circle = cv2.circle((np.ones(image.shape) * 255).astype(np.uint8), 
                                (image.shape[0]//2, image.shape[1]//2), 
                                np.random.randint(image.shape[0]//2 - 3 , image.shape[0]//2 + 15),
                                (0,0,0), -1 )
            
            mask = circle - 255
            
            image = np.multiply(image, mask)
            

#            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#            #thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            #            cv2.THRESH_BINARY,21,8)
#            ret, thresh = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY)
#            #            
#            # Create mask
#            height,width, _ = image.shape
#            mask = np.zeros((height,width), np.uint8)
#            
#            edges = cv2.Canny(thresh, 50, 200)
#            
#            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 500, param1 = 90, param2 = 60, minRadius = 0, maxRadius = 0)
#            if circles is not None:
#                for i in circles[0,:]:
#                    i[2]=i[2]+4
#                    # Draw on mask
#                    cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)
#            else:
#                mask = np.ones((height,width), np.uint8) * 255
#            
#                
#            # Copy that image using that mask
#            masked_data = cv2.bitwise_and(image, image, mask=mask)
#            
#            # Apply Threshold
#            _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
#            
#            # Find Contour
#            contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#            x,y,w,h = cv2.boundingRect(contours[0])
#            
#            # Crop masked_data
#            image = masked_data[y:y+h,x:x+w]
#            
            
        return Image.fromarray(image)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'

class TrainModel:
    
    def __init__(self, train_df, test_df):
        self.train_data = SIIMDataset(base_dir = os.getcwd(), 
                                      train_test = 'train', data = train_df)
        self.test_data = SIIMDataset(base_dir = os.getcwd(),
                                     train_test = 'test', data = test_df)
        self.batch_size = 4
        self.test_batch_size = 1

        self.train_dl = DataLoader(self.train_data, self.batch_size, shuffle=True)
        self.test_dl = DataLoader(self.test_data, self.test_batch_size, shuffle = False)
        
        self.device = torch.device('cuda')
        
        self.network = MelanomaDetectionModel(9).to(self.device)
#        print(self.network)
        
        self.learning_rate = 0.01
        self.optimizer = optim.Adam(self.network.parameters(), lr = self.learning_rate)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = 0.2, verbose = True, 
                                                              patience = 1, min_lr = 1e-5)
        
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
                metadata = data['metadata'].to(self.device)
                label = data['target'].to(self.device).view(-1,1).float()
                
                network_output = self.network(model_input, metadata)
#                print(network_output)
#                print(label)
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
            
            metadata = data['metadata'].to(self.device)
            model_input = data['image'].to(self.device)
            file_name = data['name'][0]
            
            network_output = self.network(model_input, metadata)
            prediction = torch.round(torch.sigmoid(network_output))
            prediction = int(prediction.item())

            names.append(file_name)
            predictions.append(prediction)
            
            
            print("index : %d, Prediction : %d"
                  % (i, prediction))
            
        df = pd.DataFrame(list(zip(names, predictions)), 
               columns =['image_name', 'target']) 
        df.to_csv(os.path.join(os.getcwd() , 'submission.csv'), index = False)
    
        
        
def get_train_df():
    
    
    data = pd.read_csv(os.path.join(os.getcwd() , 'final_train.csv'))
    
    data_subset_1 = data[data['target'] == 1]
    data_subset_0 = data[data['target'] == 0].sample(len(data_subset_1)) 
    data_subset = pd.concat([data_subset_1,data_subset_0])
    
    return data_subset  


def get_test_df():
    
    test_data = pd.read_csv(os.path.join(os.getcwd() , 'final_test.csv'))
   
    return test_data

train_df = get_train_df()
test_df = get_test_df()

t = TrainModel(train_df, test_df)
#t.train(1)

t.test(40)

