# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 06:03:41 2018

@author: dario
"""


   
import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.insert(0, lib_path)
#from mpslearn.classifier import mpsclassifier
from mpslearn.kernellercont import mpskerneller
from numpy import array, eye, ndarray, savetxt, zeros, int_, ones, sqrt, exp          
from numpy import genfromtxt, loadtxt 
from scipy.linalg import norm
from mpslearn.core import dvec2mps 
from random import randint, random  
from matplotlib import pyplot as plt 



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

def input_output_prepare_diffusion_prob_nonlinear_error(r=1, L=10, M=5000, m=2, er=0.01): 
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
            outpdata[isam,kpos+1] = r*inpdata[isam,kpos]**m + (1-r)*inpdata[isam,kpos+2]**m -inpdata[isam,kpos+1]**m +inpdata[isam,kpos+1] 
        outpdata[isam,0] = (1-r)*inpdata[isam,1]**m - r*inpdata[isam,0]**m + inpdata[isam,0]    
        outpdata[isam,L-1] = r*inpdata[isam,L-2]**m - (1-r)*inpdata[isam,L-1]**m + inpdata[isam,L-1]   
        erch = random() 
        if erch < er : 
            sout = 0. 
            for jpos in range(L) : 
                outpdata[isam,jpos]=random() 
                sout += outpdata[isam,jpos] 
            for jpos in range(L) : 
                outpdata[isam,jpos] /= sout 
            
    return inpdata, outpdata 

def input_output_prepare_diffusion_prob_nonlinear_lr(r=1, L=10, M=5000, m=3, m2=3, g=-0.1, g2=0.5): 
    inpdata = zeros(shape=(M,L)) 
    outpdata = zeros(shape=(M,L))      
    for isam in range(M) : 
        sinp = 0
        for ipos in range(L) : 
            inpdata[isam,ipos] = random() 
            sinp += inpdata[isam,ipos] 
        for jpos in range(L) :  
            inpdata[isam,jpos] /= sinp 
        for jpos in range(L) :   
            jpos_l = jpos - 1 
            jpos_r = jpos + 1 
            jpos_ll = jpos - 2 
            jpos_rr = jpos + 2 
            if jpos_l < 0 :
                jpos_l = L-1 
            if jpos_r > L-1 : 
                jpos_r = 0 
            if jpos_ll < 0 : 
                jpos_ll = L+jpos_ll 
            if jpos_rr > L-1 : 
                jpos_rr = jpos_rr-L 
            #outpdata[isam,kpos+1] = r*inpdata[isam,kpos]**m + (1-r)*inpdata[isam,kpos+2]**m -inpdata[isam,kpos+1]**m +inpdata[isam,kpos+1] 
            outpdata[isam,jpos] = g*(r*inpdata[isam,jpos_l]**m+ (1-r)*inpdata[isam,jpos_r]**m - inpdata[isam,jpos]**m)  +inpdata[isam,jpos]    
            outpdata[isam,jpos] += g2*((1-r)*inpdata[isam,jpos_ll]**m2 + r*inpdata[isam,jpos_rr]**m2 - inpdata[isam,jpos]**m2)   
        
    return inpdata, outpdata  

def input_output_prepare_diffusion_prob_nonlinear_lr_err(r=1, L=10, M=5000, m=3, m2=3, g=-0.1, g2=0.5, er=0.2): 
    inpdata = zeros(shape=(M,L)) 
    outpdata = zeros(shape=(M,L))      
    for isam in range(M) : 
        sinp = 0
        for ipos in range(L) : 
            inpdata[isam,ipos] = random() 
            sinp += inpdata[isam,ipos] 
        for jpos in range(L) :  
            inpdata[isam,jpos] /= sinp 
        for jpos in range(L) :   
            jpos_l = jpos - 1 
            jpos_r = jpos + 1 
            jpos_ll = jpos - 2 
            jpos_rr = jpos + 2 
            if jpos_l < 0 :
                jpos_l = L-1 
            if jpos_r > L-1 : 
                jpos_r = 0 
            if jpos_ll < 0 : 
                jpos_ll = L+jpos_ll 
            if jpos_rr > L-1 : 
                jpos_rr = jpos_rr-L 
            #outpdata[isam,kpos+1] = r*inpdata[isam,kpos]**m + (1-r)*inpdata[isam,kpos+2]**m -inpdata[isam,kpos+1]**m +inpdata[isam,kpos+1] 
            outpdata[isam,jpos] = g*(r*inpdata[isam,jpos_l]**m+ (1-r)*inpdata[isam,jpos_r]**m - inpdata[isam,jpos]**m)  +inpdata[isam,jpos]    
            outpdata[isam,jpos] += g2*((1-r)*inpdata[isam,jpos_ll]**m2 + r*inpdata[isam,jpos_rr]**m2 - inpdata[isam,jpos]**m2)   
        if random() < er : 
            sout = 0.  
            for jpos in range(L) : 
                outpdata[isam,jpos]=random() 
                sout += outpdata[isam,jpos] 
            for jpos in range(L) : 
                outpdata[isam,jpos] /= sout                     
    return inpdata, outpdata  


if __name__ == '__main__':

   r=0.5  
   L=20  
   var = 2 
   M=15000 
   D = 20  
   nsteps = 200 
   Devolve = D 
   m = 3 
   m2 = 2 
   g = -0.1 
   g2 = 0.5  
   er = 0.02 
   
   inpdata, outpdata = input_output_prepare_diffusion_prob_nonlinear_lr_err(r, L, M, m, m2, g, g2, er) 
   print(inpdata)
   xtrain = inpdata.tolist() 
   ytrain = outpdata.tolist() 
   
######## training #########
   
   D = 5 
   
   clf = mpskerneller(D=D) 
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
    
   # save mpo
   path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(D) + "_m=" + str(er) + "_m2=" + str(er) + "_g=" + str(er) + "_g2=" + str(er) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       
######## training #########
   
   D = 10 
   
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
    
   # save mpo
   path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(D) + "_m=" + str(er) + "_m2=" + str(er) + "_g=" + str(er) + "_g2=" + str(er) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       
######### training #########
   
   D = 20 
   
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
    
   # save mpo
   path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(D) + "_m=" + str(er) + "_m2=" + str(er) + "_g=" + str(er) + "_g2=" + str(er) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       
######### training #########
#   
#   D = 40 
#   
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
#    
#   # save mpo
#   path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(D) + "_m=" + str(er) + "_m2=" + str(er) + "_g=" + str(er) + "_g2=" + str(er) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       
#####################################################################        
       
   M = 5000     

   inpdata, outpdata = input_output_prepare_diffusion_prob_nonlinear_lr_err(r, L, M, m, m2, g, g2, er) 
   print(inpdata)
   xtrain = inpdata.tolist() 
   ytrain = outpdata.tolist() 


######## training #########
   
   D = 5 
   
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
    
   # save mpo
   path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(D) + "_m=" + str(m) + "_m2=" + str(m2) + "_g=" + str(g) + "_g2=" + str(g2) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       
######## training #########
   
   D = 10 
   
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
    
   # save mpo
   path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(D) + "_m=" + str(m) + "_m2=" + str(m2) + "_g=" + str(g) + "_g2=" + str(g2) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       
######## training #########
   
   D = 20 
   
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
    
   # save mpo
   path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(D) + "_m=" + str(m) + "_m2=" + str(m2) + "_g=" + str(g) + "_g2=" + str(g2) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       
######## training #########
   
#   D = 40 
#   
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
#    
#   # save mpo
#   path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(D) + "_m=" + str(er) + "_m2=" + str(er) + "_g=" + str(er) + "_g2=" + str(er) + "_er=" + str(er) +".txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       
#####################################################################

       
   M = 40000     

   inpdata, outpdata = input_output_prepare_diffusion_prob_nonlinear_lr_err(r, L, M, m, m2, g, g2, er) 
   print(inpdata)
   xtrain = inpdata.tolist() 
   ytrain = outpdata.tolist() 

######## training #########
   
   D = 5 
   
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
    
   # save mpo
   path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(D) + "_m=" + str(m) + "_m2=" + str(m2) + "_g=" + str(g) + "_g2=" + str(g2) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       
######## training #########
   
   D = 10 
   
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
    
   # save mpo
   path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(D) + "_m=" + str(m) + "_m2=" + str(m2) + "_g=" + str(g) + "_g2=" + str(g2) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       
######## training #########
   
   D = 20 
   
   clf = mpskerneller(D=D)
   clf.train(xtrain, ytrain, alpha=0.001, kmax=20, tol=1.0e-5, less_memory=False, phyx=2, phyy=2, verbose=2)
    
   # save mpo
   path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(D) + "_m=" + str(m) + "_m2=" + str(m2) + "_g=" + str(g) + "_g2=" + str(g2) + "_er=" + str(er) +".txt" 
   with open(path, 'wb') as f: 
       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)


######## training #########
   
#   D = 5 
#   
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#    
#   # save mpo
#   path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(D) + "_m=" + str(m) + "_m2=" + str(m2) + "_g=" + str(g) + "_g2=" + str(g2) + "_er=" + str(er) + "_kmax=40_tol=-7.txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
######### training #########
#   
#   D = 10 
#   
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#    
#   # save mpo
#   path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(D) + "_m=" + str(m) + "_m2=" + str(m2) + "_g=" + str(g) + "_g2=" + str(g2) + "_er=" + str(er) + "_kmax=40_tol=-7.txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
######### training #########
#   
#   D = 20 
#   
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#    
#   # save mpo
#   path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(D) + "_m=" + str(m) + "_m2=" + str(m2) + "_g=" + str(g) + "_g2=" + str(g2) + "_er=" + str(er) + "_kmax=40_tol=-7.txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
#
######################################################################
#       
#   M = 40000     
#
#   inpdata, outpdata = input_output_prepare_diffusion_prob_nonlinear_lr_err(r, L, M, m, m2, g, g2, er) 
#   print(inpdata)
#   xtrain = inpdata.tolist() 
#   ytrain = outpdata.tolist() 
#
#
######### training #########
#   
#   D = 5 
#   
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#    
#   # save mpo
#   path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(D) + "_m=" + str(m) + "_m2=" + str(m2) + "_g=" + str(g) + "_g2=" + str(g2) + "_er=" + str(er) + "_kmax=40_tol=-7.txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
######### training #########
#   
#   D = 10 
#   
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#    
#   # save mpo
#   path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(D) + "_m=" + str(m) + "_m2=" + str(m2) + "_g=" + str(g) + "_g2=" + str(g2) + "_er=" + str(er) + "_kmax=40_tol=-7.txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
#       
######### training #########
#   
#   D = 20 
#   
#   clf = mpskerneller(D=D)
#   clf.train(xtrain, ytrain, alpha=0.001, kmax=40, tol=1.0e-7, less_memory=False, phyx=2, phyy=2, verbose=2)
#    
#   # save mpo
#   path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(D) + "_m=" + str(m) + "_m2=" + str(m2) + "_g=" + str(g) + "_g2=" + str(g2) + "_er=" + str(er) + "_kmax=40_tol=-7.txt" 
#   with open(path, 'wb') as f: 
#       pickle.dump(clf.mpo, f, pickle.HIGHEST_PROTOCOL)
       
