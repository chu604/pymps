# -*- coding: utf-8 -*-
# @Author: dario
# @Date:   2017-12-12 10:08:05
# @Last Modified by:   guochu
# @Last Modified time: 2018-01-13 10:24:53
from .core import simple_uniform_lattice, overlap, dvec2mps, mps2dvec
from .core import minimizeMPOOneSite, minimizeMPOTwoSite
from numpy import eye 
from nsymmps.lattice import simple_uniform_lattice, generateSpinOperator
from nsymmps import uniformLocalObservers 
#from nsymmps.measurement import openNorm
import sys 

class mpskerneller:
    """docstring for mpskerneller"""
    def __init__(self, D):
        self.D = D

#    def __get_different_y(self, y):
#        ys = []
#        for i in range(len(y)):
#            exist=False
#            for j in range(len(ys)):
#                if (y[i] == ys[j]):
#                    exist = True
#                    break
#            if (exist==False):
#                ys.append(y[i])
#        return ys

    def train(self, x, y, singleSite=True, alpha=0.01, kmax=10, tol=1.0e-9, less_memory=True, phyx=2, phyy=2, verbose=2):
        if verbose >= 2:
            print('use single site minimization algorithm')
            print('maximum bonddimension', self.D)
            print('maximum number of iteration:', kmax, 'tolerance', tol)
#        ys = self.__get_different_y(y)
        N = len(x)
        L = len(x[0])
        Ny = len(y)
        Ly = len(y[0]) 
        
        if (Ny != N):
            sys.exit("number of samples is different!") 
        if (Ly != L):
            sys.exit("length of vectors is different!")             
        
        if (verbose >= 1):
            print('number of samples:', N, ', length of each sample:', L)
#        if (verbose >= 2):
#            print('all the different y labels:', ys)
#        assert(L >= len(ys))
#        inc = L//len(ys)
#        assert(inc > 0)
#        ytargets = {}
#        for j in range(len(ys)):
#            temp = [0]*L
#            temp[j*inc] = 1
#            lattice = simple_uniform_lattice(L, eye(2))
#            ytargets[ys[j]] = lattice.generateProdMPS(temp)
        if (verbose >= 2):
            print('convert x and y into mps...')
        #print([xj for xj in x])
        mpsxs = [dvec2mps(xj,phyx) for xj in x]
        mpsys = [dvec2mps(yj,phyy) for yj in y]
        #mpsys = [ytargets[y[j]] for j in range(N)]
        # learning process
        if (verbose >= 2):
            print('training the mpo...')
        if singleSite==True:
            mpo, itr, err, kvals_all = minimizeMPOOneSite(mpsxs, mpsys, 
                alpha=alpha, D=self.D, kmax=kmax, tol=tol, less_memory=less_memory, verbose=verbose)
        else:
            mpo, itr, err, kvals_all = minimizeMPOTwoSite(mpsxs, mpsys, 
                alpha=alpha, D=self.D, kmax=kmax, tol=tol, verbose=verbose)
        # mpo, itr, err, kvals_all = minimizeMPOOneSite2(mpsxs, mpsys, decay=1.,
        #     D=self.D, kmax=kmax, tol=tol, verbose=verbose)
        self.mpo = mpo
        self.iterations = itr
        self.error = err
        self.kvals = kvals_all
        #self.ytargets = ytargets

    def __predict_single(self, xj):
        mpsx = rvec2mps(xj)
        mpsy = self.mpo.dot(mpsx)
        yres = {}
        for key, value in self.ytargets.items():
            temp = overlap(mpsy, value)
            yres[key] = temp*temp
        s = 0
        pos = None
        for key, value in yres.items():
            if (value > s):
                pos = key
                s = value
        return pos

    def predict(self, x):
        return [self.__predict_single(x[j]) for j in range(len(x))]

    def evolve(self, xinit, nsteps, phyx, Devolve):
        #this returns the evolved mps and the local probabilities for the evolved mps 
        #the input should be a vector which is going to be translated to an mps with D=1   
        mps_x = dvec2mps(xinit,phyx)
        mps_x.svdCompress(maxbonddimension=Devolve)   
        #x_obs_list=[]*nsteps 
        #x_mps_list=[]*nsteps 
        #x_obs_list.append(xinit) 
        #x_obs_list.append(mps_x)
        # L=len(xinit)
        # op = generateSpinOperator()
        # lattice = simple_uniform_lattice(L, op['id'])
        	# generate the observables
      	# local observables measured in a continous chunk 
        # ob = uniformLocalObservers(0, L)
        # ob.set('su', op['su']) #only valid for dimension 2 
        # ob.measure(mps_x) 
        x_obs_list =[]
        x_obs_list.append(mps2dvec(mps_x))
        for istep in range(0,nsteps-1) :
            xnext_mps = self.mpo.dot(mps_x) 
            xnext_mps.svdCompress(maxbonddimension=Devolve)      
            mps_x = xnext_mps
            # ob.measure(mps_x)
            x_obs_list.append(mps2dvec(mps_x))
            #compress xnext_mps     
            #xnexr_obs must be computed     
            #x_mps_list.append(xnext_mps) 
        # x_obs_list = ob.getResults()         
        return x_obs_list, mps_x     
        #return x_mps_list, ob.getResults()     



        