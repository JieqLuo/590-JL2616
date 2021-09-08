#!/usr/bin/python3

import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
import os

class Data():
    "class is used to do the whole MLworkflow for homework 1 part3"
    def __init__(self,path):
        "necessary global variables defined"
        self.path=path
        self.df=pd.DataFrame()
        self.x_train=[]
        self.y_train=[]
        self.x_test=[]
        self.y_test=[]
        self.x=[]
        self.y=[]
        self.model_type=''
        self.nfit=2
        self.popt=[]
        self.x_mu=0
        self.x_sigma=0
        self.y_mu=0
        self.y_sigma=0
        
        
        self.iterations=[]; self.loss_train=[];  self.loss_val=[]
        self.iteration=0
     
    #Read json file and select appropriate dataset
    def read_method(self):
        self.df=pd.read_json(self.path)
        self.df=self.df[['x','y','is_adult']]
        self.df.rename(columns={'x':'age','y':'weight'},inplace= True)
        self.df['is_adult']=self.df['is_adult'].astype(int)
      
    #set model_type or algorihtm type for the whole mlworkflow(train and test)
    def model_type_method(self,algorithm):
        self.model_type=algorithm   
        if algorithm=='logistic':
            print('logistic successful')
            self.nfit=4           
        elif algorithm=='linear':
            print('linear successful')
            self.df=self.df.loc[self.df.age<18]
    
    #split data to train and test datapart and split features and target variables
    def get_x_y_method(self,x_label,y_label):
        if self.model_type=='logistic':
            print('sucessful normalization')
#             self.x,self.x_mu,self.x_sigma=self.normalization_function(self.df[x_label].to_numpy())
#             self.y,self.y_mu,self.y_sigma=self.normalization_function(self.df[y_label].to_numpy())
            mu,sigma=self.normalization_method(self.df)
            self.x_mu=mu[x_label]
            self.x_sigma=sigma[x_label]
            self.y_mu=mu[y_label]
            self.y_sigma=sigma[y_label]  

        self.x=self.df[x_label].to_numpy()
        self.y=self.df[y_label].to_numpy()
        
        train,test=self.data_split_method(0.8)
        self.x_train=train[x_label].to_numpy()
        self.y_train=train[y_label].to_numpy()
        self.x_test=test[x_label].to_numpy()
        self.y_test=test[y_label].to_numpy()
        
    #define data split method to get training dataset part and test dataset part
    def data_split_method(self,fraction):
        
        index_split=np.random.rand(len(self.df))<fraction
        
        train=self.df[index_split]
        test=self.df[~index_split]
        print(len(self.df), len(train),len(test))
        return train,test
    
    #define MSE as loss function to return traing loss and calcuate test loss
    def loss_method(self,p):
        # mean squared error
        diff_train=self.y_train-self.model_method(self.x_train,p)
        training_loss=(diff_train**2).mean()
        
        diff_test=self.y_test-self.model_method(self.x_test,p)
        test_loss=(diff_test**2).mean()
        self.loss_train.append(training_loss)
        self.loss_val.append(test_loss)
        self.iterations.append(self.iteration)

        self.iteration+=1
        
        return training_loss
    
    #normalize the whole dataset
    def normalization_method(self,data):
        
        mu=self.df.mean()
        sigma=self.df.std()
        self.df=(self.df-self.df.mean())/self.df.std()
        return mu,sigma

    #define model function for specific algorithms
    def model_method(self,x,p):
        if self.model_type=='logistic':
#             return self.sigmoid(x,p)
                
            return p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))
        else:
            return p[0]+p[1]*x

    #Use scipy optimize minimize function to train the training dataset to optimize parameters
    def train_method(self):
        np.random.seed(3)
        po=np.random.uniform(0.5,1.0,size=self.nfit)
        #train model using scipy optimizer
        res =minimize(self.loss_method,po, method='Nelder-Mead',tol=1e-15)
        self.popt=res.x
        print('OPTIMAL PARAM:', self.popt)
    
    
    #define test mehtod to get predicted test target variables
    def test_method(self):

        if self.model_type=='logistic':
            y_pred=self.popt[0]+self.popt[1]*(1.0/(1.0+np.exp(-(self.x_test-self.popt[2])/(self.popt[3]+0.00001))))
#             y_pred=self.sigmoid(self.x_test,self.popt)
        else: 
            y_pred=self.popt[0]+self.popt[1]*self.x_test                              
        return y_pred

    
    #visualization for loss comparsion, linear regression, logistic regression models
    def visualization_method(self,x_label,y_label):
        
        # loss plot
        fig, ax = plt.subplots()       
        ax.plot(self.iterations  , self.loss_train  , 'bo', label="Training loss") 
        ax.plot(self.iterations  , self.loss_val  , 'ro', label="Test loss")        
        ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel('optimizer iterations', fontsize=FS)
        plt.ylabel('loss', fontsize=FS)
        plt.title('loss plot')
        plt.show()         
        
        
        
        # visualize linear regression, logistic regression models
        fig, ax = plt.subplots()
    
        #justify whether logistic regression
        if self.model_type=='logistic':
            ax.plot(self.x_train*self.x_sigma+self.x_mu  , self.y_train*self.y_sigma+self.y_mu  , 'bo', label="Training set")
            ax.plot(self.x_test*self.x_sigma+self.x_mu, self.y_test*self.y_sigma+self.y_mu, 'yx', label="Test set")       
            
            #when target varibale is a two-category variable, transfer the probability results to 0,1 label
            if y_label=='is_adult':
                y_prob=self.model_method(self.x, self.popt)*self.y_sigma+self.y_mu
                y_categ=(y_prob>=0.5).astype(int)
                denormalize_x=self.x*self.x_sigma+self.x_mu
                sorted_x,sorted_y=zip(*sorted(zip(denormalize_x.tolist(), y_categ.tolist())))
                ax.plot(sorted_x, sorted_y , 'r-', label="Model")
            else:
                ax.plot(self.x*self.x_sigma+self.x_mu, self.model_method(self.x,self.popt)*self.y_sigma+self.y_mu , 'r-', label="Model")                
                
                
        else:
            #It is not necessary to do normalization for linear regression
            ax.plot(self.x_train, self.y_train, 'bo', label="Training set")
            ax.plot(self.x_test, self.y_test, 'yx', label="Test set")
            ax.plot(self.x, self.model_method(self.x, self.popt), 'r-', label="Model")                               
        
        ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel(x_label, fontsize=FS)
        plt.ylabel(y_label, fontsize=FS)
        plt.title(self.model_type)
        plt.show()      


# def main():
#     wl=Data(path)
#     wl.read_function()
#     wl.get_x_y_function('age','weight')
#     wl.train_function()
#     y_pred=wl.test_function()
#     wl.visualization_function('age','weight')

if __name__=='__main__':    
#    path = sys.argv[1]
    current_path=os.getcwd()
    path=current_path+'/'+'weight.json'
    
    #run linear algorithm for age and weight variables
    wl=Data(path)
    wl.read_method()
    wl.model_type_method('linear')    
    wl.get_x_y_method('age','weight')
    wl.train_method()
    y_pred=wl.test_method()
    wl.visualization_method('age','weight')
    print('first algorithm done!')

    #run logistic algorithm for age and weight variables    
    wl1=Data(path)
    wl1.read_method()
    wl1.model_type_method('logistic')
    wl1.get_x_y_method('age','weight')
    wl1.train_method()
    y_pred1=wl1.test_method()
    wl1.visualization_method('age','weight') 
    print('second algorithm done!')

    #run logistic algorithm for weight and is_adult variables    
    wl2=Data(path)
    wl2.read_method()
    wl2.model_type_method('logistic')
    wl2.get_x_y_method('weight','is_adult')
    wl2.train_method()
    y_pred2=wl2.test_method()
    wl2.visualization_method('weight','is_adult') 
    print('Third algorithm done!')
