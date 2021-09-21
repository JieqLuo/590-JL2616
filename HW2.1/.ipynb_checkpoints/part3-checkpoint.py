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
        self.nfit=4
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
    def loss_method(self,x,y,p):
        diff_train=y-self.model_method(x,p)
        training_loss=np.mean((diff_train**2))          
        
        diff_test=self.y_test-self.model_method(self.x_test,p)
        test_loss=np.mean((diff_test**2)) 
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
        return p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))



    
    def create_mini_batches(self,X,y,batch_size):
        mini_batches = []
        data = np.hstack((X,y)).reshape(-1,len(self.df.columns))
        np.random.seed(1)
        np.random.shuffle(data)
        n_minibatches =data.shape[0]// batch_size
        
        i=0
        
        for i in range(n_minibatches + 1):

            mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
            x_mini = mini_batch[:, :-1]
            y_mini = mini_batch[:, -1].reshape((-1,1))
            mini_batches.append((x_mini,y_mini))
            if mini_batch.shape[0]==0:
                print(i,batch_size)
            print(mini_batch.shape,x_mini.shape,y_mini.shape)
        if data.shape[0] % batch_size !=0:
            mini_batch =data[i* batch_size:data.shape[0]]
            x_mini = mini_batch[:, :-1]
            y_mini = mini_batch[:, -1].reshape((-1,1))
            mini_batches.append((x_mini,y_mini))
        return mini_batches

    #Use scipy optimize minimize function to train the training dataset to optimize parameters
    def optimizer(self,algo="GD",method="mini-batch"):
        po=np.random.uniform(0.1,1,size=self.nfit)
        #train model using scipy optimizer
        res =self.minimizer(po=po,algo=algo,niter=100000,LR=0.1,method=method)
        self.popt=res
        print('OPTIMAL PARAM:', self.popt)
        
    
    def minimizer(self,po,algo="GD",niter=1000,LR=0.003,method="batch"):
        if algo=="GD":
            #PARAM
            dx=0.001   #STEP SIZE FOR FINITE DIFFERENCE
            t=0          #INITIAL ITERATION COUNTER
            tmax=niter    #MAX NUMBER OF ITERATION
            tol=10**-30    #EXIT AFTER CHANGE IN F IS LESS THAN THIS 
            xi=po
            print("INITAL GUESS: ",xi)

            while(t<=tmax):
                t=t+1

                #NUMERICALLY COMPUTE GRADIENT 
                df_dx=np.zeros(self.nfit)
                if method=="batch":
                    for i in range(0,self.nfit):
                        dX=np.zeros(self.nfit);
                        dX[i]=dx; 
                        xm1=xi-dX; 
    #                     print(xi,xm1,dX,dX.shape,xi.shape)
                        df_dx[i]=(self.loss_method(self.x_train,self.y_train,xi)-self.loss_method(self.x_train,self.y_train,xm1))/dx
                    xip1=xi-LR*df_dx #STEP 
                    xi=xip1
                
                
                elif method=="minibatch":
                    mini_batches=self.create_mini_batches(X=self.x_train,y=self.y_train,batch_size=int(len(self.df)*0.4))
                    for batch in mini_batches:
                        x_mini,y_mini=batch
                        for i in range(0,self.nfit):
                            dX=np.zeros(self.nfit);
                            dX[i]=dx; 
                            xm1=xi-dX; 
        #                     print(xi,xm1,dX,dX.shape,xi.shape)
                            df_dx[i]=(self.loss_method(x_mini,y_mini,xi)-self.loss_method(x_mini,y_mini,xm1))/dx 
                        xip1=xi-LR*df_dx
                        xi=xip1

#                 elif method==" stochastic": 
#                     for batch in mini_batches:
#                         self.x_train,y_mini=batch
#                         for i in range(0,self.nfit):
#                             dX=np.zeros(self.nfit);
#                             dX[i]=dx; 
#                             xm1=xi-dX; 
#         #                     print(xi,xm1,dX,dX.shape,xi.shape)
#                             df_dx[i]=(self.loss_method(x_mini,y_mini,xi)-self.loss_method(x_mini,y_mini,xm1))/dx 
#                         xip1=xi-LR*df_dx
#                         xi=xip1                    
                if(t%10==0):
                    df=np.mean(np.absolute(self.loss_method(self.x_train,self.y_train,xip1)-self.loss_method(self.x_train,self.y_train,xi)))
                    print(t,"	",xi,"	","	",self.loss_method(self.x_train,self.y_train,xi)) #,df) 

                    if(df<tol):
                        print("STOPPING CRITERION MET (STOPPING TRAINING)")
                        break
        

        elif algo=="GD+momentum":
            #PARAM
            dx=0.001                            #STEP SIZE FOR FINITE DIFFERENCE
            t=0                                 #INITIAL ITERATION COUNTER
            tmax=niter                          #MAX NUMBER OF ITERATION
            tol=10**-10                         #EXIT AFTER CHANGE IN F IS LESS THAN THIS 
            xi=po
            print("INITAL GUESS: ",xi)
            
            change_pre=np.zeros(self.nfit)
            momentum=0.2
            while(t<=tmax):
                t=t+1

                #NUMERICALLY COMPUTE GRADIENT 
                df_dx=np.zeros(self.nfit)
                for i in range(0,self.nfit):
                    dX=np.zeros(self.nfit);
                    dX[i]=dx; 
                    xm1=xi-dX; 
#                     print(xi,xm1,dX,dX.shape,xi.shape)
                    df_dx[i]=(self.loss_method(self.x_train,self.y_train,xi)-self.loss_method(self.x_train,self.y_train,xm1))/dx
                #print(xi.shape,df_dx.shape)
 

                change=LR*df_dx+momentum*change_pre
                xip1=xi-change #STEP 
                change_pre=change
               
                if(t%10==0):
                    df=np.mean(np.absolute(self.loss_method(self.x_train,self.y_train,xip1)-self.loss_method(self.x_train,self.y_train,xi)))
                    print(t,"	",xi,"	","	",self.loss_method(self.x_train,self.y_train,xi)) #,df) 

                    if(df<tol):
                        print("STOPPING CRITERION MET (STOPPING TRAINING)")
                        break
                xi=xip1
        self.popt=xip1
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
        plt.title('loss plot')
        plt.show()         
        
        
        
        # visualize logistic regression models
        fig, ax = plt.subplots()
    
        #justify whether logistic regression
        ax.plot(self.x_train*self.x_sigma+self.x_mu  , self.y_train*self.y_sigma+self.y_mu , 'bo', label="Training set")
        ax.plot(self.x_test*self.x_sigma+self.x_mu, self.y_test*self.y_sigma+self.y_mu, 'yx', label="Test set") 
        denormalized_x=self.x*self.x_sigma+self.x_mu
        denormalized_y=self.model_method(self.x,self.popt)*self.y_sigma+self.y_mu 
        sorted_x,sorted_y=zip(*sorted(zip(denormalized_x.tolist(), denormalized_y.tolist())))
        ax.plot(sorted_x, sorted_y, 'r-', label="Model")  
        ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel(x_label, fontsize=FS)
        plt.ylabel(y_label, fontsize=FS)
        plt.title(self.model_type)
        plt.show()   


        
        #It is not necessary to do normalization for linear regression
        fig, ax = plt.subplots()  
        ax.plot(self.model_method(self.x_train,self.popt), self.y_train, 'bo', label="Train set")   
        ax.plot(self.model_method(self.x_test,self.popt), self.y_test, 'yo', label="Test set")   
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
    wl1.optimizer(algo="GD",method="batch")
    wl1.visualization_method()