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
        self.method=''

        self.nfit=4
        self.popt=[]
        self.x_mu=0+
        self.x_sigma=0
        self.y_mu=0
        self.y_sigma=0
        
        
        self.iterations=[]; self.loss_train=[];  self.loss_val=[]
        self.iteration=0
     
    #Read json file and select appropriate dataset
    def read_method(self):
        self.df=pd.read_json(self.path)
        self.df=self.df[['x','y']]
        self.df.rename(columns={'x':'age','y':'weight'},inplace= True)
      
   
    #split data to train and test datapart and split features and target variables
    def get_x_y_method(self,x_label,y_label):

        mu,sigma=self.normalization_method(self.df)
        self.x_mu=mu[x_label]
        self.x_sigma=sigma[x_label]
        self.y_mu=mu[y_label]
        self.y_sigma=sigma[y_label]  

        self.x=self.df[x_label].to_numpy()
        self.y=self.df[y_label].to_numpy()
        
        train,test=self.data_split_method(0.8)
        train=train.sort_values(by=['age'])
        self.x_train=train[x_label].reset_index(drop=True)
        self.y_train=train[y_label].reset_index(drop=True)
        self.x_test=test[x_label].reset_index(drop=True)
        self.y_test=test[y_label].reset_index(drop=True)
        
    #define data split method to get training dataset part and test dataset part
    def data_split_method(self,fraction):
        
        index_split=np.random.rand(len(self.df))<fraction
        
        #only use index to represent train dataset and test dataset
        train=self.df[index_split]
        test=self.df[~index_split]
        print(len(self.df), len(train),len(test))
        return train,test
    
    #define MSE as loss function to return traing loss and calcuate test loss
    def loss_method(self,x,y,p):
        diff_train=y-self.model_method(x,p)
        training_loss=np.mean((diff_train**2))          
        
        return training_loss
    
    #normalize the whole dataset
    def normalization_method(self,data):
        
        mu=self.df.mean()
        sigma=self.df.std()
        self.df=(self.df-self.df.mean())/self.df.std()
        return mu,sigma

    #define model function for specific algorithms
    def model_method(self,x,p):                
        return p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))



    #Use scipy optimize minimize function to train the training dataset to optimize parameters
    def optimizer(self,algo="GD",method="mini-batch",niter=10000):
        po=np.random.uniform(0.1,1,size=self.nfit)
        self.method=method
        self.model_type=algo
        #train model using scipy optimizer
        if method=="stocastic":
            LR=0.006
        else:
            LR=0.01
        res =self.minimizer(po=po,algo=algo,niter=niter,LR=LR,method=method)
        self.popt=res
        print('OPTIMAL PARAM:', self.popt)
        
    
    def minimizer(self,po,algo="GD",niter=1000,LR=0.003,method="batch"):
        #PARAM
        dx=0.001   #STEP SIZE FOR FINITE DIFFERENCE
        t=0          #INITIAL ITERATION COUNTER
        tmax=niter    #MAX NUMBER OF ITERATION
        tol=10**-30    #EXIT AFTER CHANGE IN F IS LESS THAN THIS 
        xi=po

        CLIP=True
        index_use=0
        epoch=0
        momentum=0.2
        change_pre=np.zeros(self.nfit)      
        print("INITAL GUESS: ",xi)

        while(t<=tmax):
            t=t+1
            #NUMERICALLY COMPUTE GRADIENT 
            if method=="batch":
                if t==1: index_use=self.x_train.index.values
                if t>1: epoch+=1
            elif method=="minibatch":
                # if t==1: mini_batches=self.create_mini_batches(X=self.x_train.to_numpy(),y=self.y_train.to_numpy(),batch_size=int(len(self.x_train)*0.5))
                if t==1: 
                    batch_size=int(self.x_train.index.shape[0]/2)
                    
                    index1=np.random.choice(self.x_train.index,batch_size,replace=False)
                    index_use=index1

                    index2=[i for i in self.x_train.index if i not in index1]

                    index2=np.array(index2)
                else:
                    if t%2==0:
                        index_use=index1
                    else:
                        index_use=index2
                        epoch+=1

            elif method=="stocastic":
                if t==1: 
                    count=0
                if count==self.x_train.index.shape[0]-1:
                    count=0
                    epoch+=1
                else:
                    count+=1
                index_use=count




            df_dx=np.zeros(self.nfit);
            for i in range(0,self.nfit):
                dX=np.zeros(self.nfit);
                dX[i]=dx; 
                xm1=xi-dX; 
#                     print(xi,xm1,dX,dX.shape,xi.shape)
                df_dx[i]=(self.loss_method(self.x_train[index_use],self.y_train[index_use],xi)-self.loss_method(self.x_train[index_use],self.y_train[index_use],xm1))/dx          
                if(CLIP):
                    max_grad=100
                    if df_dx[i]>max_grad: df_dx[i]=max_grad
                    if df_dx[i]<-max_grad: df_dx[i]=-max_grad

            if algo=="GD":
                change=LR*df_dx
                xip1=xi-change
            elif algo=="GD+momentum":
                change=LR*df_dx+momentum*change_pre
                xip1=xi-change
                change_pre=change
            ximl=xi
            xi=xip1

                            
            if(t%100==0):
                df=np.mean(np.absolute(self.loss_method(self.x_train,self.y_train,xip1)-self.loss_method(self.x_train,self.y_train,ximl)))
                print(t,"	",xi,"	","	",self.loss_method(self.x_train,self.y_train,xi)) #,df) 


                diff_train=self.y_train.to_numpy()-self.model_method(self.x_train.to_numpy(),xi)
                training_loss=np.mean((diff_train**2))          
                
                diff_test=self.y_test.to_numpy()-self.model_method(self.x_test.to_numpy(),xi)
                test_loss=np.mean((diff_test**2)) 
                self.loss_train.append(training_loss)
                self.loss_val.append(test_loss)
                self.iterations.append(t)


                if(df<tol):
                    print("STOPPING CRITERION MET (STOPPING TRAINING)")
                    break
        self.popt=xi
        return self.popt


  
    #visualization for loss comparsion, linear regression, logistic regression models
    def visualization_method(self):
        x_label="age"
        y_label="weight"
        
        # loss plot
        fig, ax = plt.subplots()       
        ax.plot(self.iterations  , self.loss_train  , 'bo', label="Training loss") 
        ax.plot(self.iterations  , self.loss_val  , 'ro', label="Test loss")        
        ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel('optimizer iterations', fontsize=FS)
        plt.ylabel('loss', fontsize=FS)
        plt.title(f"Loss plot for {self.model_type} + {self.method}.")
        plt.show()         
        
        
        self.x_train=self.x_train.to_numpy()
        self.y_train=self.y_train.to_numpy()
        self.x_test=self.x_test.to_numpy()
        self.y_test=self.y_test.to_numpy()

        # visualize logistic regression models
        fig, ax = plt.subplots()
    
        #justify whether logistic regression
        ax.plot(self.x_train*self.x_sigma+self.x_mu, self.y_train*self.y_sigma+self.y_mu , 'bo', label="Training set")
        ax.plot(self.x_test*self.x_sigma+self.x_mu, self.y_test*self.y_sigma+self.y_mu, 'yx', label="Test set") 
        denormalized_x=self.x*self.x_sigma+self.x_mu
        denormalized_y=self.model_method(self.x,self.popt)*self.y_sigma+self.y_mu 
        sorted_x,sorted_y=zip(*sorted(zip(denormalized_x.tolist(), denormalized_y.tolist())))
        ax.plot(sorted_x, sorted_y, 'r-', label="Model")  
        ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel(x_label, fontsize=FS)
        plt.ylabel(y_label, fontsize=FS)
        plt.title(f"fitting plot for {self.model_type} + {self.method}.")
        plt.show()   


        
        #It is not necessary to do normalization for linear regression
        fig, ax = plt.subplots()  
        ax.plot(self.model_method(self.x_train,self.popt), self.y_train, 'bo', label="Train set")   
        ax.plot(self.model_method(self.x_test,self.popt), self.y_test, 'yo', label="Test set")   
        plt.title(f"y value plot for {self.model_type} + {self.method}.")
        ax.legend()
        plt.show()   

if __name__=='__main__':    
#    path = sys.argv[1]
    current_path=os.getcwd()
    path=current_path+'/'+'weight.json'
    
    #run logistic algorithm for age and weight variables    
    wl1=Data(path)
    wl1.read_method()
    wl1.get_x_y_method('age','weight')
    wl1.optimizer(algo="GD",method="batch",niter=5000)
    wl1.visualization_method()
    print('GD+batch done!')


    wl2=Data(path)
    wl2.read_method()
    wl2.get_x_y_method('age','weight')
    wl2.optimizer(algo="GD",method="minibatch",niter=5000)
    wl2.visualization_method()
    print('GD+minibatch done!')


    wl3=Data(path)
    wl3.read_method()
    wl3.get_x_y_method('age','weight')
    wl3.optimizer(algo="GD",method="stocastic",niter=15000)
    wl3.visualization_method()
    print('GD+stocastic done!')


    wl4=Data(path)
    wl4.read_method()
    wl4.get_x_y_method('age','weight')
    wl4.optimizer(algo="GD+momentum",method="batch",niter=5000)
    wl4.visualization_method()
    print('GD+momentum batch done!')    

    wl5=Data(path)
    wl5.read_method()
    wl5.get_x_y_method('age','weight')
    wl5.optimizer(algo="GD+momentum",method="minibatch",niter=5000)
    wl5.visualization_method()
    print('GD+momentum minibatch done!')    



    wl6=Data(path)
    wl6.read_method()
    wl6.get_x_y_method('age','weight')
    wl6.optimizer(algo="GD+momentum",method="stocastic",niter=15000)
    wl6.visualization_method()
    print('GD+momentum stocastic done!')    
