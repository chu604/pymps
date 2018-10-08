# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 06:33:47 2018

@author: dario
"""


   
   
import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.insert(0, lib_path)
#from mpslearn.classifier import mpsclassifier
from mpslearn.kernellercont import mpskerneller
from mpslearn.corecont import simple_uniform_lattice, overlap, dvec2mps, mps2dvec, rvec2mps, mps2rvec
from numpy import array, eye, ndarray, savetxt, zeros, int_, ones, sqrt, exp, cos, pi          
from numpy import genfromtxt, loadtxt 
from scipy.linalg import norm
from mpslearn.core import dvec2mps 
from random import randint, random, uniform        
from matplotlib import pyplot as plt 




def diffusion_prob_evolution_nonlinear_lr(yinit, r=0.5, nsteps=5000, L=10, m=3, m2=3, g=-0.1, g2=0.5): 
    outpdata = zeros(shape=(nsteps,L))   
    for ipos in range(L) : 
        outpdata[0,ipos] = yinit[ipos]
    for istep in range(nsteps-1) : 
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
            outpdata[istep+1,jpos] = g*(r*outpdata[istep,jpos_l]**m+ (1-r)*outpdata[istep,jpos_r]**m - outpdata[istep,jpos]**m)  +outpdata[istep,jpos]    
            outpdata[istep+1,jpos] += g2*((1-r)*outpdata[istep,jpos_ll]**m2 + r*outpdata[istep,jpos_rr]**m2 - outpdata[istep,jpos]**m2)      
    return outpdata   




#def evolve(mpo, xinit, nsteps, phyx, Devolve):
#    #this returns the evolved mps and the local probabilities for the evolved mps 
#    #the input should be a vector which is going to be translated to an mps with D=1   
#    mps_x = dvec2mps(xinit,phyx)
#    mps_x.svdCompress(maxbonddimension=Devolve)   
#    x_obs_list =[]
#    x_obs_list.append(mps2dvec(mps_x))
#    for istep in range(0,nsteps-1) :
#        xnext_mps = mpo.dot(mps_x) 
#        xnext_mps.svdCompress(maxbonddimension=Devolve)      
#        mps_x = xnext_mps
#        x_obs_list.append(mps2dvec(mps_x))
#    return x_obs_list, mps_x     

def evolve(mpo, xinit, nsteps, phyx, Devolve):
    #this returns the evolved mps and the local probabilities for the evolved mps 
    #the input should be a vector which is going to be translated to an mps with D=1   
    mps_x = rvec2mps(xinit)
    mps_x.svdCompress(maxbonddimension=Devolve)   
    x_obs_list =[]
    x_obs_list.append(mps2rvec(mps_x))
    for istep in range(0,nsteps-1) :
        xnext_mps = mpo.dot(mps_x) 
        xnext_mps.svdCompress(maxbonddimension=Devolve)      
        mps_x = xnext_mps
        x_obs_list.append(mps2rvec(mps_x))   
    return x_obs_list, mps_x     



if __name__ == '__main__':

   L=20  
   #M=15000      
   #D = [5, 10, 20]        
   D = [5, 10, 20] 
   m = 3 
   m2 = 2 
   g = -0.1 
   g2 = 0.5 
   Devolve = D 
   er = 0.  
   n_ic = 100 
   nsteps = 100 

###### load mpo ########  
   for M in [20000, 40000]: 
       for Dd in D: 
           
           path = "mpo_prob_L=" + str(L) + "_M=" + str(M) + "_D=" + str(Dd) + "_m=" + str(m) + "_m2=" + str(m2) + "_g=" + str(g) + "_g2=" + str(g2) + "_er=" + str(er) +"_kmax=40_tol=-7.txt"        
           with open(path, 'rb') as f: 
               tr_mpo = pickle.load(f)    
        
              
        #### iterate on initial conditions #####  
        
           for i_ic in range(n_ic): 
               
            ####### initial condition #########    
               xinit = zeros(L).tolist() 
               ################################     
    #           # for gaussian initial condition 
    #           pos =  randint(0, L-1) 
    #           ss = 0 
    #           for ipos in range(L) : 
    #               xinit[ipos] = exp(-((ipos-pos)**2)/(2*var**2)) 
    #               #xinit[ipos] = 1  #### uniform initial condition     
    #               ss += xinit[ipos] 
    #           for jpos in range(L) : 
    #               xinit[jpos] /= ss 
               ################################     
               # for sinusoidal initial condition with gaussian  
               pos =  randint(0, L-1) 
               nlambda = randint(1, 5) 
               var = uniform(1,5) 
               ss = 0 
               for ipos in range(L) : 
                   xinit[ipos] = 1/2*( 1+cos(2*pi/L*ipos*nlambda) )*exp(-((ipos-pos)**2)/(2*var**2)) 
                   #xinit[ipos] = 1  #### uniform initial condition     
                   ss += xinit[ipos] 
               for jpos in range(L) : 
                   xinit[jpos] /= ss                     
               ##############################    
               # for random initial condition 
            #   sinp = 0
            #   for ipos in range(L) : 
            #       xinit[ipos] = random() 
            #       sinp += xinit[ipos] 
            #   for jpos in range(L) :  
            #       xinit[jpos] /= sinp 
               ####################################    
               ### for delta pick initial condition     
            #   xinit[L-3] = 1 
               phyx = 2 
               Devolve = Dd 
               yevolve_prob, yevolve_mps = evolve(tr_mpo, xinit, nsteps, phyx, Devolve)
               prob_evo = array(yevolve_prob)  
               #print(prob_evo)   
               
               plt.figure() 
               plt.imshow(prob_evo) 
               plt.show() 
            
            
            ####### evolution #########    
               
               #print(xinit)
               exact_evo = diffusion_prob_evolution_nonlinear_lr(xinit, r, nsteps, L, m, m2, g, g2)  
            
               plt.figure() 
               plt.imshow(exact_evo) 
               plt.show() 
               
               #print(exact_evo)    
               predict_evo_file = "./Load_data/load_predict_evo_file_prob_er_L=" + str(L) + "_M=" + str(M) + "_nsteps=" + str(nsteps) + "_D=" + str(Dd) + "_m=" + str(m) + "_m2=" + str(m2) + "_g=" + str(g) + "_g2=" + str(g2) + "_er=" + str(er) + "_ic=" + str(i_ic) +"_kmax=40_tol=-7.txt"   
               exact_evo_file = "./Load_data/load_exact_evo_file_prob_er_L=" + str(L) + "_M=" + str(M) + "_nsteps=" + str(nsteps) + "_D=" + str(Dd) + "_m=" + str(m) + "_m2=" + str(m2) + "_g=" + str(g) + "_g2=" + str(g2) + "_er=" + str(er) + "_ic=" + str(i_ic) +"_kmax=40_tol=-7.txt"   
            
               savetxt(predict_evo_file, prob_evo) 
               savetxt(exact_evo_file, exact_evo)
               
               nerrors = 0 
               for istep in range(nsteps) : 
                   for ipos in range(L) : 
                       nerrors += abs(prob_evo[istep,ipos] - exact_evo[istep,ipos])   
               print(nerrors)    
               
               #print(exact_evo + prob_evo)