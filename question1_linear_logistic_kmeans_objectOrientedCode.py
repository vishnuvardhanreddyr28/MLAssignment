# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:51:38 2020

@author: ADMIN
"""
import numpy as np
import scipy
import random
from scipy.stats import norm


class climb_steps():
    np.random.seed(654)
    def __init__(self,step,throws,probdis=[]):
        self.step=step
        self.throw=throws
        self.probdis=probdis
        
    def probability_calculator(self):
      j=1
      for number in range(self.throw):
        if(len(self.probdis)==0):
            value=random.randint(1,7)
        else:
            value=random.choice(self.probdis)    
        if(value<=2 and j!=1):
          j=j-1
        elif(value==6):
          j=j+random.randint(1,7)
        else:
          j=j+1
      if(j>=self.step):
        return 1
      else:
        return 0
    
    def probability_value(self):
        x=0
        for i in range(1000):
          x=x+self.probability_calculator()
        print(x/1000)


class logistic_regression():
    def __init__(self,x,y):
        self.lr=x
        self.epochs=y

    def generate_data(self,p,coefficents):
        self.x=[]
        self.p=p+1
        self.bias=coefficents[0]
        random.seed(1)
        for j in range(p):
            tempx=scipy.stats.norm.rvs(0,1,100)
            self.x.append(tempx)
        
        self.y=coefficents[1]*self.x[0]+self.bias
        for j in range(len(coefficents)-2):
            self.y+=coefficents[j+2]*self.x[j+1]
            error=scipy.stats.norm.rvs(0,0.25,100)
            self.y+=error
            self.y=np.array(self.y)
            self.x=np.array(self.x)
        label=self.y
        self.y=[]
        label=(label>(sum(label)/len(label)))
        for j in label:
            if(j):
                self.y.append(1.0)
            else:
                self.y.append(0.0)
        self.y=np.array(self.y)
        
    def sigmoid(self):
        self.ypred=1/(1+np.exp(-self.ypred))

    def loss_function(self):
        self.error=-1*(self.y*np.log(self.ypred)+(1-self.y)*np.log(1-self.ypred)).mean()
    

    def initialize_parameters(self):
        self.param=[]
        for j in range(self.p):
            self.param.append(0.0)
        self.param=np.array(self.param)

    def applyl1regularization(self,lambda1):
        self.n=len(self.y)
        self.initialize_parameters()
        for j in range(self.epochs):
            l1=lambda1*self.param*self.param
            self.ypred=np.dot(self.param[1:],self.x)+self.param[0]
            self.sigmoid()
            self.loss_function()
            print("iteration ",j," error= ",self.error)
            self.param[1:]=self.param[1:]+(2*self.lr/self.n)*((np.dot(self.x,self.y-self.ypred))-l1[1:])
            self.param[:1]=self.param[:1]+(2*self.lr/self.n)*(sum(self.y-self.ypred)-l1[:1])
        print("parameter values ",self.param)
        
    def applyl2regularization(self,lambda2):
        self.n=len(self.y)
        self.initialize_parameters()
        for j in range(self.epochs):
            l2=abs(lambda2*self.param)
            self.ypred=np.dot(self.param[1:],self.x)+self.param[0]
            self.sigmoid()
            self.loss_function()
            print("iteration ",j," error= ",self.error)
            self.param[1:]=self.param[1:]+(2*self.lr/self.n)*((np.dot(self.x,self.y-self.ypred))-l2[1:])
            self.param[:1]=self.param[:1]+(2*self.lr/self.n)*(sum(self.y-self.ypred)-l2[:1])
        print("parameter values ",self.param)

    def gradient_descent(self):
        self.n=len(self.y)
        self.initialize_parameters()
        for j in range(self.epochs):
            self.ypred=np.dot(self.param[1:],self.x)+self.param[0]
            self.sigmoid()
            self.loss_function()
            print("iteration ",j," error= ",self.error)
            self.param[1:]=self.param[1:]+(2*self.lr/self.n)*(np.dot(self.x,self.y-self.ypred))
            self.param[:1]=self.param[:1]+(2*self.lr/self.n)*sum(self.y-self.ypred)
        print("parameter values ",self.param)
                    
    
    


class linear_regresssion():
    def __init__(self,x,y):
        self.lr=x
        self.epochs=y

    def generate_data(self,p,coefficents):
        self.x=[]
        self.p=p+1
        random.seed(1)
        self.bias=coefficents[0]
        for j in range(p):
            tempx=scipy.stats.norm.rvs(0,1,100)
            self.x.append(tempx)
        self.y=coefficents[1]*self.x[0]+self.bias
        for j in range(len(coefficents)-2):
            self.y+=coefficents[j+2]*self.x[j+1]
            error=scipy.stats.norm.rvs(0,0.25,100)
            self.y+=error
            self.y=np.array(self.y)
            self.x=np.array(self.x)
    
    def ols(self):
        self.error=0
        for j in range(len(self.y)):
            self.error+=(self.y[0]-self.ypred[0])*(self.y[0]-self.ypred[0])
        self.error/=(2*len(self.y))

    def initialize_parameters(self):
        self.param=[]
        for j in range(self.p):
            self.param.append(0.0)
        self.param=np.array(self.param)
    
    def applyl1regularization(self,lambda1):
        self.n=len(self.y)
        self.initialize_parameters()
        for j in range(self.epochs):
            l1=lambda1*self.param*self.param
            self.ypred=np.dot(self.param[1:],self.x)+self.param[:1]
            self.ols()
            print("iteration ",j," error= ",self.error)
            self.param[1:]=self.param[1:]+(2*self.lr/self.n)*((np.dot(self.x,self.y-self.ypred))-l1[1:])
            self.param[:1]=self.param[:1]+(2*self.lr/self.n)*(sum(self.y-self.ypred)-l1[:1])
        print("parameter values ",self.param)
        
    def applyl2regularization(self,lambda2):
        self.n=len(self.y)
        self.initialize_parameters()
        for j in range(self.epochs):
            l2=abs(lambda2*self.param)
            self.ypred=np.dot(self.param[1:],self.x)+self.param[:1]
            self.ols()
            print("iteration ",j," error= ",self.error)
            self.param[1:]=self.param[1:]+(2*self.lr/self.n)*((np.dot(self.x,self.y-self.ypred))-l2[1:])
            self.param[:1]=self.param[:1]+(2*self.lr/self.n)*(sum(self.y-self.ypred)-l2[:1])
        print("parameter values ",self.param)
             
    

    def gradient_descent(self):
        self.n=len(self.y)
        self.initialize_parameters()
        for j in range(self.epochs):
            self.ypred=np.dot(self.param[1:],self.x)+self.param[:1]
            self.ols()
            print("iteration ",j," error= ",self.error)
            self.param[1:]=self.param[1:]+(2*self.lr/self.n)*((np.dot(self.x,self.y-self.ypred)))
            self.param[:1]=self.param[:1]+(2*self.lr/self.n)*sum(self.y-self.ypred)
        print("parameter values ",self.param)
            
    
        
    def predict(self,x):
        y=np.dot(self.param,x)
        print(y)
        
        

class kmeans():
    def __init__(self,x,y,epoch):
        self.points=x
        self.k=y
        self.epochs=epoch
        
    def generate_data(self):
        self.cordinates=np.random.rand(100,2)
        self.cordinates[:66]+=2
        self.cordinates[33:66,:1]+=4
        self.cordinates[66:]+=8
        np.random.shuffle(self.cordinates)
    
    def euclidean_distance(self,x,y):
        return ((x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1]))
    
    def centroids(self):
        cordinates=self.cordinates
        np.random.shuffle(cordinates)
        self.centroid=cordinates[:self.k]
        
    def clustering(self):
        self.minvar=999999
        self.kcluster={}
        for iteration in range(self.epochs):
            self.centroids()
            self.clusters={}
            for j in range(self.k):
                self.clusters[j]=[]
                
            for j in self.cordinates:
                min_dist=9999
                cluster=0
                for number in range(self.k):
                    dist=self.euclidean_distance(self.centroid[number],j)
                    if(min_dist>dist):
                        min_dist=dist
                        cluster=number
                self.clusters[cluster].append(j)
            
            self.variance=0
            self.mean=0
            for j in range(self.k):
                self.mean+=len(self.clusters[j])
            self.mean/=len(self.clusters)
            
            for j in range(self.k):
                self.variance+=(self.mean-len(self.clusters[j]))*(self.mean-len(self.clusters[j]))
                
            if(self.minvar>self.variance):
                self.minvar=self.variance
                self.kcluster=self.clusters
        
        print(len(self.kcluster[0]))
        print(len(self.kcluster[1]))
        print(len(self.kcluster[2]))
        print(self.minvar)
        


        

#Question-1

#model=climb_steps(60,250,[])
#model.probability_value()
#model=climb_steps(60,250,[1,1,2,2,2,3,3,4,5,6])
#model.probability_value()

#LinearRegression

#model=linear_regresssion(0.01,500)
#model.generate_data(5,[-1,1,0.7,3,-2,-0.3]) 
#model.gradient_descent()
#model.applyl1regularization(0.01)
#model.applyl2regularization(0.01)

#LogisticRegression

#model=logistic_regression(0.01,500)
#model.generate_data(5,[-1,1,0.7,3,-2,-0.3]) 
#model.gradient_descent()
#model.applyl1regularization(0.01)
#model.applyl2regularization(0.01)


#Kmeans

#model=kmeans(100,3,10)
#model.generate_data()
#model.clustering()
        
        

    



    
    
        
    
    
