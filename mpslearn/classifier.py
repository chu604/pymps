# -*- coding: utf-8 -*-
# @Author: 1000787
# @Date:   2017-08-31 16:54:05
# @Last Modified by:   guochu
# @Last Modified time: 2018-01-13 11:52:19
from .core import simple_uniform_lattice, overlap, rvec2mps
from .core import minimizeMPOOneSite, minimizeMPOTwoSite
from numpy import eye

class mpsclassifier:
	"""docstring for mpsclassifier"""
	def __init__(self, D):
		self.D = D

	def __get_different_y(self, y):
		ys = []
		for i in range(len(y)):
			exist=False
			for j in range(len(ys)):
				if (y[i] == ys[j]):
					exist = True
					break
			if (exist==False):
				ys.append(y[i])
		return ys

	def train(self, x, y, singleSite=True, alpha=0.01, kmax=10, tol=1.0e-9, less_memory=True, verbose=2):
		if verbose >= 2:
			if singleSite==True:
				print('use single site minimization algorithm')
			else:
				print('use two site minimization algorithm')
			print('maximum bonddimension', self.D)
			print('maximum number of iteration:', kmax, 'tolerance', tol)
		ys = self.__get_different_y(y)
		N = len(x)
		L = len(x[0])
		if (verbose >= 1):
			print('number of samples:', N, ', length of each sample:', L)
		if (verbose >= 2):
			print('all the different y labels:', ys)
		assert(L >= len(ys))
		inc = L//len(ys)
		assert(inc > 0)
		ytargets = {}
		for j in range(len(ys)):
			temp = [0]*L
			temp[j*inc] = 1
			lattice = simple_uniform_lattice(L, eye(2))
			ytargets[ys[j]] = lattice.generateProdMPS(temp)
		if (verbose >= 2):
			print('convert x and y into mps...')
		mpsxs = [rvec2mps(xj) for xj in x]
		mpsys = [ytargets[y[j]] for j in range(N)]
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
		# 	D=self.D, kmax=kmax, tol=tol, verbose=verbose)
		self.mpo = mpo
		self.iterations = itr
		self.error = err
		self.kvals = kvals_all
		self.ytargets = ytargets

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





		