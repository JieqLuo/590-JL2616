#------------------------
#DATA CLASS
#------------------------
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize

IPLOT=True
class DataClass:

    #INITIALIZE
	def __init__(self,FILE_NAME):



		#READ FILE
		with open(FILE_NAME) as f:
			self.input = json.load(f)  #read into dictionary

		#CONVERT DICTIONARY INPUT AND OUTPUT MATRICES #SIMILAR TO PANDAS DF   
		X=[]; Y=[]
		for key in self.input.keys():
			if(key in X_KEYS): X.append(self.input[key])
			if(key in Y_KEYS): Y.append(self.input[key])

		#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
		self.X=np.transpose(np.array(X))
		self.Y=np.transpose(np.array(Y))

		#EXTRACT AGE<18
		if(model_type=="linear"):
			self.Y=self.Y[self.X[:]<18]; 
			self.X=self.X[self.X[:]<18]; 

		#TAKE MEAN AND STD DOWN COLUMNS (I.E DOWN SAMPLE DIMENSION)
		self.XMEAN=np.mean(self.X,axis=0); self.XSTD=np.std(self.X,axis=0) 
		self.YMEAN=np.mean(self.Y,axis=0); self.YSTD=np.std(self.Y,axis=0) 

		self.nfit=4
		self.method=''
		self.model_type=''

	def report(self):
		print("--------DATA REPORT--------")
		print("X shape:", self.X.shape)
		print("X means:",np.mean(self.X,axis=0))
		print("X stds:",np.std(self.X,axis=0))
		print("X examples")
		#PRINT FIRST 5 SAMPLES
		for i in range(0,self.X.shape[1]):
			print("X column ",i,": ",self.X[0:5,i])

	def partition(self,f_train=0.825, f_val=0.15, f_test=0.025):
		#f_train=fraction of data to use for training

		#TRAINING: 	 DATA THE OPTIMIZER "SEES"
		#VALIDATION: NOT TRAINED ON BUT MONITORED DURING TRAINING
		#TEST:		 NOT MONITORED DURING TRAINING (ONLY USED AT VERY END)
		if(f_train+f_val+f_test != 1.0):
			raise ValueError("f_train+f_val+f_test MUST EQUAL 1");

		#PARTITION DATA
		rand_indices = np.random.permutation(self.X.shape[0]) #randomize indices
		CUT1=int(f_train*self.X.shape[0]); 
		CUT2=int((f_train+f_val)*self.X.shape[0]); 
		self.train_idx, self.val_idx, self.test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]

	def normalize(self):
		self.X=(self.X-self.XMEAN)/self.XSTD 
		self.Y=(self.Y-self.YMEAN)/self.YSTD  

	def model(self,x,p):
		if(model_type=="linear"):   return  p[0]*x+p[1]  
		if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.0001))))

	def predict(self,p):
		self.YPRED_T=self.model(self.X[self.train_idx],p)
		self.YPRED_V=self.model(self.X[self.val_idx],p)
		self.YPRED_TEST=self.model(self.X[self.test_idx],p)

	def un_normalize(self):
		self.X=self.XSTD*self.X+self.XMEAN 
		self.Y=self.YSTD*self.Y+self.YMEAN 
		self.YPRED_T=self.YSTD*self.YPRED_T+self.YMEAN 
		self.YPRED_V=self.YSTD*self.YPRED_V+self.YMEAN 
		self.YPRED_TEST=self.YSTD*self.YPRED_TEST+self.YMEAN 

	#------------------------
	#DEFINE LOSS FUNCTION
	#------------------------
	def loss(self,p):
		global iteration,iterations,loss_train,loss_val

		#MAKE PREDICTIONS FOR GIVEN PARAM
		self.predict(p)

		#LOSS (MSE)
		training_loss=(np.mean((self.YPRED_T-self.Y[self.train_idx])**2.0))  #MSE
		validation_loss=(np.mean((self.YPRED_V-self.Y[self.val_idx])**2.0))  #MSE

		loss_train.append(training_loss); loss_val.append(validation_loss)
		iterations.append(iteration)

		iteration+=1

		return training_loss
	#------------------------
	#DEFINE optimizer function
	#------------------------

	def optimizer(self,algo="GD",method="mini-batch",niter=15000):

	    po=np.random.uniform(-1,1,size=self.nfit)
	    self.method=method
	    self.model_type=algo
	    LR=0.01
	    res =self.minimizer(po=po,algo=algo,niter=niter,LR=LR,method=method)
	    self.popt=res
	    print('OPTIMAL PARAM:', self.popt)
        
	#------------------------
	#DEFINE minimizer function
	#------------------------    
	def minimizer(self,po,algo="GD",niter=1000,LR=0.01,method="batch"):

	    dx=0.001   #STEP SIZE FOR FINITE DIFFERENCE
	    t=0          #INITIAL ITERATION COUNTER
	    tmax=niter    #MAX NUMBER OF ITERATION
	    tol=10**-20    #EXIT AFTER CHANGE IN F IS LESS THAN THIS 
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
	            index_use=self.train_idx

	        df_dx=np.zeros(self.nfit);
	        for i in range(0,self.nfit):
	            dX=np.zeros(self.nfit);
	            dX[i]=dx; 
	            xm1=xi-dX; 
	#                     print(xi,xm1,dX,dX.shape,xi.shape)
	            df_dx[i]=(self.loss(xi)-self.loss(xm1))/dx          
	            if(CLIP):
	                max_grad=100
	                if df_dx[i]>max_grad: df_dx[i]=max_grad
	                if df_dx[i]<-max_grad: df_dx[i]=-max_grad

	        if algo=="GD":
	            xip1=xi-LR*df_dx
	        elif algo=="GD+momentum":
	            xip1=xi-LR*df_dx+momentum*change_pre
	            change_pre=LR*df_dx+momentum*change_pre
	        ximl=xi
	        xi=xip1

	                        
	        if(t%100==0):
	            df=np.mean(np.absolute(self.loss(xip1)-self.loss(ximl)))
	            print(t,"	",xi,"	","	",self.loss(xi)) #,df) 

	            if(df<tol):
	                print("STOPPING CRITERION MET (STOPPING TRAINING)")
	                break
	    self.popt=xi
	    return self.popt
	#FUNCTION PLOTS
	def plot_1(self,xla='x',yla='y'):
		if(IPLOT):
			fig, ax = plt.subplots()
			ax.plot(self.X[self.train_idx]    , self.Y[self.train_idx],'o', label='Training') 
			ax.plot(self.X[self.val_idx]      , self.Y[self.val_idx],'x', label='Validation') 
			ax.plot(self.X[self.test_idx]     , self.Y[self.test_idx],'*', label='Test') 
			ax.plot(self.X[self.train_idx]    , self.YPRED_T,'.', label='Model') 
			plt.xlabel(xla, fontsize=18);	plt.ylabel(yla, fontsize=18); 	plt.legend()
			plt.show()

	#PARITY PLOT
	def plot_2(self,xla='y_data',yla='y_predict'):
		if(IPLOT):
			fig, ax = plt.subplots()
			ax.plot(self.Y[self.train_idx]  , self.YPRED_T,'*', label='Training') 
			ax.plot(self.Y[self.val_idx]    , self.YPRED_V,'*', label='Validation') 
			ax.plot(self.Y[self.test_idx]    , self.YPRED_TEST,'*', label='Test') 
			plt.xlabel(xla, fontsize=18);	plt.ylabel(yla, fontsize=18); 	plt.legend()
			plt.show()
#------------------------
#MAIN 
#------------------------
model_type="logistic";   NFIT=2; X_KEYS=['x']; Y_KEYS=['y']
DATA_KEYS=['x','y']
iteration=0
iterations=[]
loss_train=[]
loss_val=[]
#INITIALIZE DATA OBJECT 
Data=DataClass('weight.json')

#BASIC DATA PRESCREENING
Data.report()
Data.partition()
Data.normalize()
Data.report()
Data.optimizer(algo="GD",method="batch",niter=5000)
Data.plot_1()					#PLOT DATA
Data.plot_2()	