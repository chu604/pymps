# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:34:40 2018

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
from mpslearn.core import simple_uniform_lattice, overlap, dvec2mps, mps2dvec, mps2dvec_up, mps2rvec




def FAEastModel_evolution_det_longrange(yinit, nsteps=5, L=10, dd=2): 
    outpdata = zeros(shape=(nsteps,L))  
    for ipos in range(L) : 
        outpdata[0,ipos] = yinit[ipos] 
    for istep in range(nsteps-1) :   
        for ipos in range(L) : 
            outpdata[istep+1,ipos] = outpdata[istep, ipos]
        for ipos in range(L-dd) : 
            if (1-outpdata[istep,ipos+dd] > 0) :
                outpdata[istep+1, ipos] = 1 - outpdata[istep, ipos]
#                jp = random()
#                if r < jp :
#                    outpdata[isam, ipos] = 1 - inpdata[isam, ipos] 
    return int_(outpdata)

def evolve(mpo, xinit, nsteps, phyx, Devolve):
    #this returns the evolved mps and the local probabilities for the evolved mps 
    #the input should be a vector which is going to be translated to an mps with D=1   
    mps_x = dvec2mps(xinit,phyx)
    mps_x.svdCompress(maxbonddimension=Devolve)   
    x_obs_list =[]
    x_obs_list.append(mps2dvec(mps_x))
    for istep in range(0,nsteps-1) :
        xnext_mps = mpo.dot(mps_x) 
        xnext_mps.svdCompress(maxbonddimension=Devolve)      
        mps_x = xnext_mps
        x_obs_list.append(mps2dvec(mps_x))
    return x_obs_list, mps_x     


if __name__ == '__main__':

   L=20  
   M=15000      
   D = 8        
   dd = 3 
   Devolve = D 
   er = 0.4 
   n_ic = 100 
   nsteps = 150 

###### load mpo ########  

   for Dd in [10, 9, 8, 7, 6, 5, 4]: 
       
       path = "mpo_FAEd_L=" + str(L) + "_M=" + str(M) + "_nsteps=2500_D=" + str(Dd) + "_dd=" + str(dd) + "_er=" + str(er) +".txt" 
       with open(path, 'rb') as f: 
           tr_mpo = pickle.load(f)    
    
          
    #### iterate on initial conditions #####  
    
       for i_ic in range(n_ic): 
           
           xinit = int_(zeros(L)).tolist() 
    
        ###### initial condition #########    
           #xinit[int_(L/2)] = 1    
           #xinit[L-1] = 1 
           for ii in range(L) : 
               xinit[ii] = randint(0,1) 
           phyx = 2 
           Devolve = Dd 
           yevolve_prob, yevolve_mps = evolve(tr_mpo, xinit, nsteps, phyx, Devolve)
           prob_evo = array(yevolve_prob)  
           print(prob_evo)   
           #print(int_(prob_evo))
           
           plt.figure() 
           plt.imshow(prob_evo) 
           plt.show() 
        
        
        ####### evolution #########    
           
           print(xinit)
           exact_evo = FAEastModel_evolution_det_longrange(xinit, nsteps, L, dd)  
        
           plt.figure() 
           plt.imshow(exact_evo) 
           plt.show() 
           
           #print(exact_evo)    
           predict_evo_file = "load_predict_evo_file_FAEd_er_L=" + str(L) + "_M=" + str(M) + "_nsteps=" + str(nsteps) + "_D=" + str(Dd) + "_dd=" + str(dd) + "_er=" + str(er) + "_ic=" + str(i_ic) +".txt"   
           exact_evo_file = "load_exact_evo_file_FAEd_er_L=" + str(L) + "_M=" + str(M) + "_nsteps=" + str(nsteps) + "_D=" + str(Dd) + "_dd=" + str(dd) + "_er=" + str(er) + "_ic=" + str(i_ic) +".txt"   
        
           savetxt(predict_evo_file, prob_evo) 
           savetxt(exact_evo_file, exact_evo)
           
           nerrors = 0 
           for istep in range(nsteps) : 
               for ipos in range(L) : 
                   nerrors += abs(prob_evo[istep,ipos] - exact_evo[istep,ipos])   
           print(nerrors)    
           
           #print(exact_evo + prob_evo)