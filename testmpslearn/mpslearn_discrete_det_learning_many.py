# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 22:06:25 2018

@author: dario
"""

   
import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.insert(0, lib_path)
#from mpslearn.classifier import mpsclassifier
from mpslearn.kerneller import mpskerneller
from numpy import array, eye, ndarray, savetxt, zeros, int_, ones    
from numpy import genfromtxt, loadtxt 
from scipy.linalg import norm
from mpslearn.core import dvec2mps 
from random import randint, random  
from matplotlib import pyplot as plt 

def input_output_prepare(r=1, L=10, M=5000): 
    inpdata = zeros(shape=(M,L)) 
    outpdata = zeros(shape=(M,L))      
    for isam in range(M) : 
        posi = randint(0, L-1) 
        poso = posi + 1
        if poso == L :
            poso = 0 
        inpdata[isam,posi] = 1 
        outpdata[isam,poso] = 1 
    return int_(inpdata), int_(outpdata) 

def input_output_prepare_diffusion(r=1, L=10, M=5000): 
    inpdata = zeros(shape=(M,L)) 
    outpdata = zeros(shape=(M,L))      
    for isam in range(M) : 
        posi = randint(0, L-1) 
        rn = random() 
        mov = 1 
        if rn > r :
            mov = -1 
        poso = posi + mov    
        if poso == -1 :
            poso = L-1 
        if poso == L :
            poso = 0 
        inpdata[isam,posi] = 1 
        outpdata[isam,poso] = 1 
    return int_(inpdata), int_(outpdata) 

def input_output_prepare_diffusion_prob(r=1, L=10, M=5000): 
    inpdata = zeros(shape=(M,L)) 
    outpdata = zeros(shape=(M,L))      
    for isam in range(M) : 
        sinp = 0
        for ipos in range(L) : 
            inpdata[isam,ipos] = random() 
            sinp += inpdata[isam,ipos] 
        for jpos in range(L) :  
            inpdata[isam,jpos] /= sinp 
        for kpos in range(L-2) : 
            outpdata[isam,kpos+1] = r*inpdata[isam,jpos] + (1-r)*inpdata[isam,jpos+2] 
        outpdata[isam,0] = r*inpdata[isam,L-1] + (1-r)*inpdata[isam,1]  
        outpdata[isam,L-1] = r*inpdata[isam,L-2] + (1-r)*inpdata[isam,0]    
    return inpdata, outpdata 

def diffusion_evolution(r=1, L=10, M=5000): 
#### this needs fixing   
    inpdata = zeros(shape=(M,L)) 
    outpdata = zeros(shape=(M,L))      
    for isam in range(M) : 
        posi = randint(0, L-1) 
        rn = random() 
        mov = 1 
        if rn > 0.5 :
            mov = -1 
        poso = posi + mov    
        if poso == -1 :
            poso = L-1 
        if poso == L :
            poso = 0 
        inpdata[isam,posi] = 1 
        outpdata[isam,poso] = 1 
    return int_(inpdata), int_(outpdata) 

def input_output_prepare_FAEastModel(r=1, L=10, M=5000): 
    r_param = r    
    inpdata = zeros(shape=(M,L)) 
    outpdata = zeros(shape=(M,L))      
    for isam in range(M) : 
        for ipos in range(L) :
            inpdata[isam,ipos] = randint(0,1) 
            outpdata[isam,ipos] = inpdata[isam,ipos]     
        for ipos in range(L-1) : 
            if (1-inpdata[isam,ipos+1] > 0) :
                jp = random()
                if (jp > r_param) :
                    outpdata[isam, ipos] = 1 - outpdata[isam, ipos] 
    return int_(inpdata), int_(outpdata) 

def input_output_prepare_FAEastModel_det(L=10, M=5000): 
    inpdata = zeros(shape=(M,L)) 
    outpdata = zeros(shape=(M,L))      
    for isam in range(M) : 
        for ipos in range(L) :
            inpdata[isam,ipos] = randint(0,1) 
            outpdata[isam,ipos] = inpdata[isam,ipos]     
        for ipos in range(L-1) : 
            if (1-inpdata[isam,ipos+1] > 0) :
                outpdata[isam, ipos] = 1 - inpdata[isam, ipos]
    return int_(inpdata), int_(outpdata) 

def input_output_prepare_FAEastModel_det_err(L=10, M=5000, er=0.02): 
    inpdata = zeros(shape=(M,L)) 
    outpdata = zeros(shape=(M,L))      
    for isam in range(M) : 
        for ipos in range(L) :
            inpdata[isam,ipos] = randint(0,1) 
            outpdata[isam,ipos] = inpdata[isam,ipos]     
        for ipos in range(L-1) : 
            if (1-inpdata[isam,ipos+1] > 0) :
                outpdata[isam, ipos] = 1 - inpdata[isam, ipos]
        if random()<er : 
            for ipos in range(L):
                outpdata[isam,ipos] = randint(0,1) 
    return int_(inpdata), int_(outpdata) 

def input_output_prepare_FAEastModel_det_longrange(L=10, M=5000, dd=2): 
    inpdata = zeros(shape=(M,L)) 
    outpdata = zeros(shape=(M,L))      
    for isam in range(M) : 
        for ipos in range(L) :
            inpdata[isam,ipos] = randint(0,1) 
            outpdata[isam,ipos] = inpdata[isam,ipos]     
        for ipos in range(L-dd) : 
            if (1-inpdata[isam,ipos+dd] > 0) :
                outpdata[isam, ipos] = 1 - inpdata[isam, ipos]
    return int_(inpdata), int_(outpdata) 

def input_output_prepare_FAEastModel_det_longrange_err(L=10, M=5000, dd=2, er=0.02): 
    inpdata = zeros(shape=(M,L)) 
    outpdata = zeros(shape=(M,L))      
    for isam in range(M) : 
        for ipos in range(L) :
            inpdata[isam,ipos] = randint(0,1) 
            outpdata[isam,ipos] = inpdata[isam,ipos]     
        for ipos in range(L-dd) : 
            if (1-inpdata[isam,ipos+dd] > 0) :
                outpdata[isam, ipos] = 1 - inpdata[isam, ipos]
        if random()<er : 
            for ipos in range(L):
                outpdata[isam,ipos] = randint(0,1)
    return int_(inpdata), int_(outpdata) 



if __name__ == '__main__':

   L = 20  
   M = 15000   
   dd = 3     
   Devolve = 10
   
   inpdata, outpdata = input_output_prepare_FAEastModel_det_longrange_err(L, M, dd, er)   
   print(inpdata)
   xtrain = inpdata.tolist() 
   ytrain = outpdata.tolist() 
   

   er = 0.4       

######## training #########
#
#   D = 4  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       
       
####### training #########
#
#   D = 5  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#              
#       
######## training #########
#
#   D = 6  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       

####### training #########

   D = 7  
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
   
   # save mpo
   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       
    
####### training #########

   D = 8  
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
   
   # save mpo
   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       

####### training #########

   D = 9   
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
   
   # save mpo
   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)

####### training #########

   D = 10   
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
   
   # save mpo
   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)       

############################################
############################################    
############################################  


   er = 0.3       

######## training #########
#
#   D = 4  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       
       
####### training #########

   D = 5  
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
   
   # save mpo
   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
              
       
####### training #########

   D = 6  
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
   
   # save mpo
   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       

####### training #########

   D = 7  
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
   
   # save mpo
   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       
    
####### training #########

   D = 8  
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
   
   # save mpo
   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       

####### training #########

   D = 9   
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
   
   # save mpo
   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       
####### training #########

   D = 10   
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
   
   # save mpo
   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)              

############################################
############################################    
############################################  


#   er = 0.2      
#
######## training #########
#
#   D = 4  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#       
######## training #########
#
#   D = 5  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#              
#       
######## training #########
#
#   D = 6  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#
######## training #########
#
#   D = 7  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#    
######## training #########
#
#   D = 8  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#
######## training #########
#
#   D = 9   
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
######## training #########
#
#   D = 10   
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)       
#
#############################################
#############################################    
#############################################  
#
#
#   er = 0.2      
#
######## training #########
#
#   D = 4  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#       
######## training #########
#
#   D = 5  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#              
#       
######## training #########
#
#   D = 6  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#
######## training #########
#
#   D = 7  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#    
######## training #########
#
#   D = 8  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#
######## training #########
#
#   D = 9   
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
######## training #########
#
#   D = 10   
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)       
#
#
#############################################
#############################################    
#############################################  
#
#
#   er = 0.005      
#
######## training #########
#
#   D = 4  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#       
######## training #########
#
#   D = 5  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#              
#       
######## training #########
#
#   D = 6  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#
######## training #########
#
#   D = 7  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#    
######## training #########
#
#   D = 8  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#
######## training #########
#
#   D = 9   
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
######## training #########
#
#   D = 10   
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)       
#
#
#############################################
#############################################    
#############################################  
#
#
#   er = 0.001      
#
######## training #########
#
#   D = 4  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#       
######## training #########
#
#   D = 5  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#              
#       
######## training #########
#
#   D = 6  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#
######## training #########
#
#   D = 7  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#    
######## training #########
#
#   D = 8  
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#
######## training #########
#
#   D = 9   
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#
######## training #########
#
#   D = 10   
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#   
#   # save mpo
#   path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(D) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)       
#
#
#
#
#
#
