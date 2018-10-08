# -*- coding: utf-8 -*-
# @Author: 1000787
# @Date:   2017-08-31 16:48:20
# @Last Modified by:   1000787
# @Last Modified time: 2018-03-07 17:02:09
import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.insert(0, lib_path)
from nsymmps.DTensor import contract, svdCompress, qrCompress, diag
from nsymmps import MPS, MPO, simple_uniform_lattice, createEmptyMPO, createrandommps
from nsymmps.measurement import overlap
from nsymmps.update import updateCleft, updateCright, updateClefth1h2, updateCrighth1h2
from nsymmps.reduceD import reduceDSingleSite
from numpy.random import rand
from numpy import ones, eye, zeros, sqrt, array, amax, argmax, kron, power
from scipy.linalg import lstsq, svd, solve
from scipy.linalg import norm
from numpy.linalg import cond
from scipy.sparse.linalg import bicg, cg, LinearOperator, lsqr

def initCstorageRight(mpsA, mpsB):
	"""
	auxilliary function for reduceMPSOneSite
	initialize Cstorage from right to left
	"""
	assert(len(mpsA) == len(mpsB))
	L = len(mpsA)
	Cstorage = [None]*(L+1)
	Cstorage[0] = ones((1, 1))
	Cstorage[L] = ones((1, 1))
	for i in range(L-1, 0, -1):
		Cstorage[i] = updateCrighth1h2(Cstorage[i+1], None, mpsA[i], mpsB[i].conj())
	return Cstorage

def reduceMPSIterativeOneSite(mpsA, mpsB, kmax=50, tol=1.0e-10, verbose=1):
	"""
	return mpsB as best approximation of mpsAs
	"""
	L = len(mpsB)
	assert(len(mpsB) == L)
	Cstorage = initCstorageRight(mpsA, mpsB) 
	itr = 0
	while (itr < kmax):
		if verbose >= 2:
			print('we are at', itr, '-th iteraction.')
		kvals = []
		for j in range(L-1):
			sitempsB = reduceDSingleSite(mpsA[j], Cstorage[j], Cstorage[j+1])
			kvals.append(norm(sitempsB))
			mpsB[j], u = qrCompress('L', sitempsB, (2,))
			Cstorage[j+1] = updateClefth1h2(Cstorage[j], None, mpsA[j], mpsB[j])
		for j in range(L-1, 0, -1):
			sitempsB = reduceDSingleSite(mpsA[j], Cstorage[j], Cstorage[j+1])
			kvals.append(norm(sitempsB))
			u, mpsB[j] = qrCompress('R', sitempsB, (1,2))
			Cstorage[j] = updateCrighth1h2(Cstorage[j+1], None, mpsA[j], mpsB[j])
		kvals = array(kvals)
		err = abs(kvals.std() / kvals.mean())
		if verbose >= 2:
			print('the error is', err)
		if err < tol:
			if verbose >= 2:
				print('converges after', itr+1, 'sweeps.')
			break
		itr += 1
	mpsB[0] = contract(mpsB[0], u, ((2,), (0,)))
	return mpsB

def reduceMPSOneSite(mpsAs, mpsB, kmax=50, tol=1.0e-10, verbose=1):
	"""
	mpsAs is a list of mps
	return mpsB as best approximation of mpsAs
	"""
	mpsB.svdCompress()
	L = len(mpsB)
	N = len(mpsAs)
	for j in range(N):
		assert(len(mpsAs[j]) == L)
	Cstorage = [initCstorageRight(mps, mpsB) for mps in mpsAs]
	itr = 0
	while (itr < kmax):
		if verbose >= 2:
			print('we are at', itr, '-th iteraction.')
		kvals = []
		for j in range(L-1):
			sitempsB = zeros((Cstorage[0][j].shape[0], mpsB[j].shape[1], Cstorage[0][j+1].shape[0]))
			for n in range(N):
				sitempsB += reduceDSingleSite(mpsAs[n][j], Cstorage[n][j], Cstorage[n][j+1])
			kvals.append(norm(sitempsB))
			mpsB[j], s, u, bet = svdCompress(sitempsB, (2,), -1, tol, useIterSolver=0, verbose=verbose)
			for n in range(N):
				Cstorage[n][j+1] = updateClefth1h2(Cstorage[n][j], None, mpsAs[n][j], mpsB[j])
		for j in range(L-1, 0, -1):
			sitempsB = zeros((Cstorage[0][j].shape[0], mpsB[j].shape[1], Cstorage[0][j+1].shape[0]))
			for n in range(N):
				sitempsB += reduceDSingleSite(mpsAs[n][j], Cstorage[n][j], Cstorage[n][j+1])
			kvals.append(norm(sitempsB))
			u, mpsB.svectors[j], mpsB[j], bet = svdCompress(sitempsB, (1, 2), -1, tol, useIterSolver=0, verbose=verbose)
			for n in range(N):
				Cstorage[n][j] = updateCrighth1h2(Cstorage[n][j+1], None, mpsAs[n][j], mpsB[j])
		kvals = array(kvals)
		err = abs(kvals.std() / kvals.mean())
		if verbose >= 2:
			# print(kvals)
			print('the error is', err)
		if err < tol:
			if verbose >= 2:
				print('converges after', itr+1, 'sweeps.')
			break
		itr += 1
	u = contract(u, diag(mpsB.svectors[1]), ((1,), (0,)))
	mpsB[0] = contract(mpsB[0], u, ((2,), (0,)))
	return mpsB

def map_scalar_to_vector(s):
	assert(abs(s) <= 1.)
	# y = (-s + sqrt(2.-s*s))/2.
	return [sqrt(1.-s*s), s]

def map_vector_to_scalar(v):
	assert(len(v)==2)
	if v[0] >= 0.:
		return v[1]
	else:
		return -v[1]

def generateProdMPS(x):
	L = len(x)
	assert(L > 0)
	d = len(x[0])
	for i in range(1, L):
		assert(len(x[i])==d)
	lattice = simple_uniform_lattice(L, eye(d))
	return lattice.generateProdMPS(x)

def rvec2mps(x):
	"""
	turn a list of integers into mps
	each element has to be positive real number
	"""
	L = len(x)
	s = 0.
	for j in range(L):
		s = s + float(x[j])*float(x[j])
	s = sqrt(s)
	mpsx = [None]*(len(x))
	xp = [None]*L
	lattice = simple_uniform_lattice(L, eye(2))
	for j in range(L):
		temp = x[j]/s
		# assert(temp >= 0.)
		# xp[j] = [sqrt(1-temp*temp), temp]
		xp[j] = map_scalar_to_vector(temp)
	return lattice.generateProdMPS(xp)*s

def mps2rvec(mps):
	L = len(mps)
	assert(L > 0)
	for i in range(L):
		assert(mps[i].shape[1] == 2)
	sz = array([[1., 0.], [0., -1.]])
	v = zeros((L,))
	n = mps.norm(False)
	mps /= n
	mps.svdCompress(maxbonddimension=1)
	# mpsB = createrandommps(L, 2, 1)
	# mpsB.svdCompress()
	# mpsB = reduceMPSIterativeOneSite(mps, mpsB)
	# mps = mpsB
	n = mps.norm(True)
	# if (abs(n) < 1.0e-12):
	# 	# the norm is too close to 0
	# 	return [0]*L
	mps /= n
	for i in range(L):
		v[i] = map_vector_to_scalar([mps[i][0,0,0], mps[i][0,1,0]])
	return n*v

def dvec2mps(x, phyx):
	"""
	turn a list of integers into mps
	"""
	L = len(x)
	lattice = simple_uniform_lattice(L, eye(phyx))
	return lattice.generateProdMPS(x);
	# mpsx = [None]*(len(x))
	# for j in range(len(x)):
	# 	mpsx[j] = lattice.generateProdMPS(x[j])
	# return mpsx

def imax(l):
	s = 0.
	j = 0
	for i in range(len(l)):
		if (l[i] >= s):
			j = i
			s = l[i]
	return j

def mps2dvec(mps):
	L = len(mps)
	assert(L > 0)
	d = mps[0].shape[1]
	for i in range(L):
		assert(mps[i].shape[1]==d)
	# mps.svdCompress(maxbonddimension=1)
	mpsB = createrandommps(L, d, 1)
	mpsB.svdCompress()
	mpsB = reduceMPSIterativeOneSite(mps, mpsB)
	mps = mpsB
	n = mps.norm(True)
	# if (abs(n) < 1.0e-12):
	# 	# the norm is too close to 0
	# 	return [0]*L
	# assert(not mps.trivial())
	mps /= n
	v = [None]*L
	for i in range(L):
		v[i] = imax([abs(mps[i][0, j, 0]) for j in range(d)])
	return v

# def prodmps2rvec(mps):
# 	# mps.svdCompress()
# 	s = mps.norm(True)
# 	mps /= s
# 	for m in mps:
# 		assert(m.shape[0]==1 and m.shape[2]==1 and m.shape[1]==2)
# 	r = zeros((len(mps),))
# 	for i in range(len(mps)):
# 		r[i] = map_vector_to_scalar([mps[i][0,0,0], mps[i][0, 1, 0]])
# 	return s*r

def createrandommpo(L, dx, dy, D):
	mpo = [None]*L
	assert(L > 1)
	mpo[0] = rand(dy, 1, D, dx)/D
	mpo[L-1] = rand(dy, D, 1, dx)/D
	for i in range(1, L-1):
		mpo[i] = rand(dy, D, D, dx)/D
	# for m in mpo:
		# m /= norm(m.reshape(m.size))
	return MPO(mpo)

def updateCCleft(hold, mpoj, mpsj):
	hnew = contract(mpoj, mpsj, ((3,), (1,)))
	hnew1 = contract(hold, hnew, ((2,3), (1,3)))
	hnew1 = contract(hnew.conj(), hnew1, ((3,1,0),(0,1,2)))
	return hnew1.transpose((1,0,2,3))

def updateCCright(hold, mpoj, mpsj):
	hnew = contract(mpoj, mpsj, ((3,), (1,)))
	hnew1 = contract(hold, hnew, ((2,3),(2,4)))
	hnew1 = contract(hnew.conj(), hnew1, ((4,2,0),(0,1,2)))
	return hnew1.transpose((1,0,2,3))

def updateMPOleft(hold, mpoj):
	hnew = contract(hold, mpoj, ((1,), (1,)))
	return contract(mpoj.conj(), hnew, ((0,1,3), (1,0,3)))

def updateMPOright(hold, mpoj):
	hnew = contract(mpoj.conj(), hold, ((2,), (0,)))
	return contract(hnew, mpoj, ((0,2,3),(0,3,2)))

def initHstorageRight(mpo, mpsx, mpsy):
	assert(len(mpo) == len(mpsx))
	assert(len(mpo) == len(mpsy))
	L = len(mpo)
	assert(L > 0)
	hstorage = [None]*(L+1)
	hstorage[0] = ones((1,1,1))
	hstorage[L] = ones((1,1,1))
	for i in range(L-1, 0, -1):
		hstorage[i] = updateCright(hstorage[i+1], mpo[i], mpsx[i], mpsy[i].conj())
	return hstorage

def computeHstorageLeft(mpo, mpsx, mpsy, site):
	"""
	update Hstorage from left till site
	"""
	assert(len(mpo) == len(mpsx))
	assert(len(mpo) == len(mpsy))
	L = len(mpo)
	assert(L > 0)
	assert(site >= 0 and site < L)
	h = ones((1,1,1))
	for i in range(site):
		h = updateCleft(h, mpo[i], mpsx[i], mpsy[i].conj())
	return h

def computeHstorageRight(mpo, mpsx, mpsy, site):
	"""
	update Hstorage from right till site-1
	"""
	assert(len(mpo) == len(mpsx))
	assert(len(mpo) == len(mpsy))
	L = len(mpo)
	assert(L > 0)
	assert(site >= 0 and site < L)
	h = ones((1,1,1))
	for i in range(L-1, site, -1):
		h = updateCright(h, mpo[i], mpsx[i], mpsy[i].conj())
	return h

def initHHstorageRight(mpo, mps):
	assert(len(mpo) == len(mps))
	L = len(mpo)
	assert(L > 1)
	hstorage = [None]*(L+1)
	hstorage[0] = ones((1,1,1,1))
	hstorage[L] = ones((1,1,1,1))
	for i in range(L-1, 0, -1):
		hstorage[i] = updateCCright(hstorage[i+1], mpo[i], mps[i])
	return hstorage

def computeHHstorageLeft(mpo, mps, site):
	assert(len(mpo) == len(mps))
	L = len(mpo)
	assert(L > 1)
	assert(site >= 0 and site < L)
	h = ones((1,1,1,1))
	for i in range(site):
		h = updateCCleft(h, mpo[i], mps[i])
	return h

def computeHHstorageRight(mpo, mps, site):
	assert(len(mpo) == len(mps))
	L = len(mpo)
	assert(L > 1)
	assert(site >= 0 and site < L)
	h = ones((1,1,1,1))
	for i in range(L-1, site, -1):
		h = updateCCright(h, mpo[i], mps[i])
	return h

def initMPOstorageRight(mpo):
	L = len(mpo)
	assert(L > 1)
	hstorage = [None]*(L+1)
	hstorage[0] = ones((1,1))
	hstorage[L] = ones((1,1))
	for i in range(L-1, 0, -1):
		hstorage[i] = updateMPOright(hstorage[i+1], mpo[i])
	return hstorage

def computeMPOstorageLeft(mpo, site):
	L = len(mpo)
	assert(L > 1)
	assert(site >= 0 and site < L)
	h = ones((1,1))
	for i in range(site):
		h = updateMPOleft(h, mpo[i])
	return h

def computeMPOstorageRight(mpo, site):
	L = len(mpo)
	assert(L > 1)
	assert(site >= 0 and site < L)
	h = ones((1,1))
	for i in range(L-1, site, -1):
		h = updateMPOright(h, mpo[i])
	return h

def HSingleSite(mpsxj, mpsyj, hleft, hright):
	"""
	output
	*******0*******
	***1---M----3
	*******2*******
	"""
	hnew = contract(hleft, mpsxj, ((2,),(0,)))
	hnew = contract(hnew, hright, ((3,),(2,)))
	hnew = contract(mpsyj, hnew, ((0,2),(0,3)))
	return hnew

def HHSingleSite(mpsj, hleft, hright):
	"""
	output
	*******0*******
	***1---M----4
	***2---M----5
	*******3*******
	"""

	hnew = contract(mpsj.conj(), hleft, ((0,), (0,)))
	hnew = contract(hnew, mpsj, ((4,), (0,)))
	hnew = contract(hnew, hright, ((1,5),(0,3)))
	return hnew

def error_func(heff, xeff, mpoj):
	hnew = contract(mpoj, heff, ((1,2,3),(1,4,0))) - xeff
	return contract(hnew, mpoj, ((0,1,2,3),(0,1,3,2)))

# def renormalizeMPO(mpo):
# 	L = len(mpo)
# 	s = 1.
# 	nrms = [None]*L
# 	for i in range(L):
# 		nrm = norm(mpo[i].reshape(mpo[i].size))
# 		nrms[i] = nrm
# 		s = power(s, i/(i+1))*power(nrm, 1/(i+1))
# 	print('norm of mpo is', s)
# 	for i in range(L):
# 		mpo[i] *= (s/nrms[i])

def minimizeMPOOneSite(mpsxs, mpsys, alpha=0.1, D=20, kmax=50, tol=1.0e-9, less_memory=False, verbose=1):
	assert(len(mpsxs) == len(mpsys))
	N = len(mpsxs)
	assert(N > 0)
	L = len(mpsxs[0])
	dx = mpsxs[0][0].shape[1]
	dy = mpsys[0][0].shape[1]
	for i in range(N):
		for j in range(1,L):
			assert(mpsxs[i][j].shape[1] == dx)
			assert(mpsys[i][j].shape[1] == dy)
	# create an initial random mpo
	mpo = createrandommpo(L, dx, dy, D)
	# renormalizeMPO(mpo)
	hms = initMPOstorageRight(mpo)
	if less_memory == False:
		hs = [initHstorageRight(mpo, mpsxs[j], mpsys[j]) for j in range(N)]
		hhs = [initHHstorageRight(mpo, mpsxs[j]) for j in range(N)]
	itr = 0
	kvals_all = []
	while (itr < kmax):
		if (verbose >= 1):
			print('we are at', itr, '-th iteration.')
		kvals = []
		if less_memory == True:
			h = [computeHstorageLeft(mpo, mpsxs[n], mpsys[n], 0) for n in range(N)]
			hh = [computeHHstorageLeft(mpo, mpsxs[n], 0) for n in range(N)]
		for j in range(L-1):
			if (verbose >= 2):
				print('sweeping from left to right on site', j)
			xeff = zeros((dy, mpo[j].shape[1], dx, mpo[j].shape[2]))
			heff = zeros((dx, mpo[j].shape[1], mpo[j].shape[1], dx, 
				mpo[j].shape[2], mpo[j].shape[2]))
			if less_memory==False:
				for n in range(N):
					heff += HHSingleSite(mpsxs[n][j], hhs[n][j], hhs[n][j+1])
					xeff += HSingleSite(mpsxs[n][j], mpsys[n][j], hs[n][j], hs[n][j+1])
			else:
				for n in range(N):
					tmp = computeHHstorageRight(mpo, mpsxs[n], j)
					heff += HHSingleSite(mpsxs[n][j], hh[n], tmp)
					tmp = computeHstorageRight(mpo, mpsxs[n], mpsys[n], j)
					xeff += HSingleSite(mpsxs[n][j], mpsys[n][j], h[n], tmp)
			hefft = heff.transpose((1,0,4,2,3,5))
			# add regularization term for mpo to make sure 
			# the elements of mpo are not too large
			mpoeff = contract(hms[j], hms[j+1], ((), ()))
			mpoeff = contract(mpoeff, eye(dx), ((), ()))
			mpoeff = mpoeff.transpose((0,4,2,1,5,3))
			hefft += alpha*mpoeff
			# -------------------------------------
			xefft = xeff.transpose((1,2,3,0))
			d = hefft.shape[0]*hefft.shape[1]*hefft.shape[2]
			xeff2 = xefft.reshape((d, xefft.shape[3]))
			heff2 = hefft.reshape((d, d))
			# if (verbose >= 2):
			# 	print('conditional number of heff:', cond(heff2))
			# 	print('norm of heff', norm(heff2))
			# 	print('norm of xeff', norm(xeff2))
			# s = svd(a=heff2, compute_uv=False)
			# print('heff', heff2)
			# print('xeff', xeff2)
			# print('the singular values', s)
			# linear solver
			# (1) using lstsq
			# mpoj, r, k, s = lstsq(heff2, xeff2)
			# if (verbose >= 2):
			# 	print('rank: ', heff2.shape[1], '; effective rank:', k)
			# 	print('norm of mpoj', norm(mpoj.reshape(mpoj.size)))
			mpoj = solve(heff2, xeff2)
			# if (verbose >= 2):
			# 	print('norm of mpoj', norm(mpoj.reshape(mpoj.size)))
			# end of (1)
			# (2) using bicg
			# A = LinearOperator(shape=(d*dy, d*dy), 
			# 	matvec=lambda v:heff2.dot(v.reshape((d, dy))).reshape(d*dy),
			# 	rmatvec=lambda v:heff2.T.dot(v.reshape((d, dy))).reshape(d*dy))
			# x0 = mpo[j].transpose((1,2,3,0))
			# res = bicg(A=A, b = xeff2.reshape(d*dy), x0=x0.reshape(x0.size))
			# mpoj = res[0].reshape((d, dy))
			# end of (2)
			distance = contract(heff2.dot(mpoj)-2.*xeff2, mpoj, ((0,1),(0,1)))
			kvals.append(distance)
			if (verbose >= 2):
				print('error after updating', distance)
			mpoj = mpoj.reshape(xefft.shape)
			mpoj = mpoj.transpose((3,0,2,1))
			# print('error after updating', error_func(heff, 2.*xeff, mpoj))
			#-------do svd to mpoj----------
			# mpoj, s, u, bet = svdCompress(mpoj, (2,), 10000, 1.0e-10, useIterSolver=0, verbose=verbose)
			# mpo[j] = mpoj.transpose((0,1,3,2))
			# u = contract(diag(s), u, ((1,),(0,)))
			# mpo[j+1] = contract(u, mpo[j+1], ((1,), (1,)))
			# mpo[j+1] = mpo[j+1].transpose((1,0,2,3))
			# ------end of do svd to mpoj---------
			#------do qr to mpoj--------
			mpoj, u = qrCompress('L', mpoj, (2,))
			mpo[j] = mpoj.transpose((0,1,3,2))
			mpo[j+1] = contract(u, mpo[j+1], ((1,), (1,)))
			mpo[j+1] = mpo[j+1].transpose((1,0,2,3))
			#------end of do qr to mpoj---------
			#------do nothing to mpoj--------
			# mpo[j] = mpoj
			# update hs and hhs
			if less_memory==False:
				for n in range(N):
					hs[n][j+1] = updateCleft(hs[n][j], mpo[j], mpsxs[n][j], mpsys[n][j].conj())
					hhs[n][j+1] = updateCCleft(hhs[n][j], mpo[j], mpsxs[n][j])
			else:
				for n in range(N):
					h[n] = updateCleft(h[n], mpo[j], mpsxs[n][j], mpsys[n][j].conj())
					hh[n] = updateCCleft(hh[n], mpo[j], mpsxs[n][j])
			# update hms(mpostorage)
			hms[j+1] = updateMPOleft(hms[j], mpo[j])
		if less_memory == True:
			h = [computeHstorageRight(mpo, mpsxs[n], mpsys[n], L-1) for n in range(N)]
			hh = [computeHHstorageRight(mpo, mpsxs[n], L-1) for n in range(N)]
		for j in range(L-1, 0, -1):
			if (verbose >= 2):
				print('sweeping from right to left on site', j)
			xeff = zeros((dy, mpo[j].shape[1], dx, mpo[j].shape[2]))
			heff = zeros((dx, mpo[j].shape[1], mpo[j].shape[1], dx, 
				mpo[j].shape[2], mpo[j].shape[2]))
			if less_memory==False:
				for n in range(N):
					heff += HHSingleSite(mpsxs[n][j], hhs[n][j], hhs[n][j+1])
					xeff += HSingleSite(mpsxs[n][j], mpsys[n][j], hs[n][j], hs[n][j+1])
			else:
				for n in range(N):
					tmp = computeHHstorageLeft(mpo, mpsxs[n], j)
					heff += HHSingleSite(mpsxs[n][j], tmp, hh[n])
					tmp = computeHstorageLeft(mpo, mpsxs[n], mpsys[n], j)
					xeff += HSingleSite(mpsxs[n][j], mpsys[n][j], tmp, h[n])
			hefft = heff.transpose((1,0,4,2,3,5))
			# add regularization term for mpo to make sure 
			# the elements of mpo are not too large
			mpoeff = contract(hms[j], hms[j+1], ((), ()))
			mpoeff = contract(mpoeff, eye(dx), ((), ()))
			mpoeff = mpoeff.transpose((0,4,2,1,5,3))
			hefft += alpha*mpoeff
			# -------------------------------------
			xefft = xeff.transpose((1,2,3,0))
			d = hefft.shape[0]*hefft.shape[1]*hefft.shape[2]
			xeff2 = xefft.reshape((d, xefft.shape[3]))
			heff2 = hefft.reshape((d, d))
			# if (verbose >= 2):
			# 	print('conditional number of heff:', cond(heff2))
			# 	print('norm of heff', norm(heff2))
			# 	print('norm of xeff', norm(xeff2))
			# print('heff', heff2)
			# print('xeff', xeff2)
			# linear solver
			# (1) using lstsq
			# mpoj, r, k, s = lstsq(heff2, xeff2)
			# if (verbose >= 2):
			# 	print('rank: ', heff2.shape[1], '; effective rank:', k)
			# 	print('norm of mpoj', norm(mpoj.reshape(mpoj.size)))
			mpoj = solve(heff2, xeff2)
			# if (verbose >= 2):
			# 	print('norm of mpoj', norm(mpoj.reshape(mpoj.size)))
			# end of (1)
			# (2) using bicg
			# A = LinearOperator(shape=(d*dy, d*dy), 
			# 	matvec=lambda v:heff2.dot(v.reshape((d, dy))).reshape(d*dy),
			# 	rmatvec=lambda v:heff2.T.dot(v.reshape((d, dy))).reshape(d*dy))
			# x0 = mpo[j].transpose((1,2,3,0))
			# res = bicg(A=A, b = xeff2.reshape(d*dy), x0=x0.reshape(x0.size))
			# mpoj = res[0].reshape((d, dy))
			# end of (2)
			distance = contract(heff2.dot(mpoj)-2.*xeff2, mpoj, ((0,1),(0,1)))
			kvals.append(distance)
			if (verbose >= 2):
				print('error after updating', distance)
			mpoj = mpoj.reshape((xeff.shape[1], xeff.shape[2], xeff.shape[3], xeff.shape[0]))
			mpoj = mpoj.transpose((3,0,2,1))
			# print('error after updating', error_func(heff, 2.*xeff, mpoj))
			#--------do svd to mpoj-----------
			# u, s, mpoj, bet = svdCompress(mpoj, (0,2,3), 10000, 1.0e-10, useIterSolver=0, verbose=verbose)
			# mpo[j] = mpoj.transpose((1,0,2,3))
			# u = contract(u, diag(s), ((1,), (0,)))
			# mpo[j-1] = contract(mpo[j-1], u, ((2,), (0,)))
			# mpo[j-1] = mpo[j-1].transpose((0,1,3,2))
			# --------end of do svd to mpoj--------
			# --------do qr to mpoj------------
			u, mpoj = qrCompress('R', mpoj, (0,2,3))
			mpo[j] = mpoj.transpose((1,0,2,3))
			mpo[j-1] = contract(mpo[j-1], u, ((2,), (0,)))
			mpo[j-1] = mpo[j-1].transpose((0,1,3,2))
			# --------end of do qr to mpoj--------
			#------do nothing to mpoj--------
			# mpo[j] = mpoj
			# update hs and hhs
			if less_memory==False:
				for n in range(N):
					hs[n][j] = updateCright(hs[n][j+1], mpo[j], mpsxs[n][j], mpsys[n][j].conj())
					hhs[n][j] = updateCCright(hhs[n][j+1], mpo[j], mpsxs[n][j])
			else:
				for n in range(N):
					h[n] = updateCright(h[n], mpo[j], mpsxs[n][j], mpsys[n][j].conj())
					hh[n] = updateCCright(hh[n], mpo[j], mpsxs[n][j])
			# update hms(mpostorage)
			hms[j] = updateMPOright(hms[j+1], mpo[j])
		# renormalize mpo
		# if verbose>=2:
		# 	print('start to renormalize mpo and reinitialize hstorages')
		# renormalizeMPO(mpo)
		# hs = [initHstorageRight(mpo, mpsxs[j], mpsys[j]) for j in range(N)]
		# hhs = [initHHstorageRight(mpo, mpsxs[j]) for j in range(N)]
		# hms = initMPOstorageRight(mpo)
		# if verbose>=2:
		# 	print('finish renormalizing mpo and reinitialize hstorages')
		if (verbose >= 3):
			print('the kvalues are:', kvals)
		kvals_all = kvals_all + kvals
		kvals = array(kvals)
		err = abs(kvals.std() / kvals.mean())
		if verbose >= 1:
			# print(kvals)
			print('the standard error is', err)
		if err < tol:
			if verbose >= 1:
				print('converges after', itr+1, 'sweeps.')
			break
		itr += 1
	# u = contract(u, diag(s), ((1,),(0,)))
	# mpo[0] = contract(mpo[0], u, ((2,),(0,)))
	# mpo[0] = mpo[0].transpose((0,1,3,2))
	return mpo, itr, err, kvals_all

# def minimizeMPOOneSite2(mpsxs, mpsys, D=20, kmax=50, tol=1.0e-9, verbose=1):
# 	assert(len(mpsxs) == len(mpsys))
# 	N = len(mpsxs)
# 	assert(N > 0)
# 	L = len(mpsxs[0])
# 	dx = mpsxs[0][0].shape[1]
# 	dy = mpsys[0][0].shape[1]
# 	for i in range(N):
# 		for j in range(1,L):
# 			assert(mpsxs[i][j].shape[1] == dx)
# 			assert(mpsys[i][j].shape[1] == dy)
# 	# create an initial random mpo
# 	mpo = createrandommpo(L, dx, dy, D)
# 	hs = [initHstorageRight(mpo, mpsxs[j], mpsys[j]) for j in range(N)]
# 	hhs = [initHHstorageRight(mpo, mpsxs[j]) for j in range(N)]
# 	itr = 0
# 	kvals_all = []
# 	while (itr < kmax):
# 		if (verbose >= 1):
# 			print('we are at', itr, '-th iteraction.')
# 		kvals = []
# 		for j in range(L-1):
# 			if (verbose >= 2):
# 				print('sweeping from left to right on site', j)
# 			xeff = zeros((dy, hs[0][j].shape[1], dx, hs[0][j+1].shape[1]))
# 			heff = zeros((dx, hhs[0][j].shape[1], hhs[0][j].shape[2], dx, 
# 				hhs[0][j+1].shape[1], hhs[0][j+1].shape[2]))
# 			for n in range(N):
# 				heff += HHSingleSite(mpsxs[n][j], hhs[n][j], hhs[n][j+1])
# 				xeff += HSingleSite(mpsxs[n][j], mpsys[n][j], hs[n][j], hs[n][j+1])
# 			# print('error before updating', error_func(heff, 2.*xeff, mpo[j]))
# 			hefft = heff.transpose((1,0,4,2,3,5))
# 			xefft = xeff.transpose((1,2,3,0))
# 			xeff2 = xefft.reshape((xefft.shape[0]*xefft.shape[1]*xefft.shape[2], xefft.shape[3]))
# 			d = hefft.shape[0]*hefft.shape[1]*hefft.shape[2]
# 			heff2 = hefft.reshape((d, d))
# 			if (verbose >= 2):
# 				print('conditional number of heff:', cond(heff2))
# 				print('norm of heff', norm(heff2))
# 				print('norm of xeff', norm(xeff2))
# 			# s = svd(a=heff2, compute_uv=False)
# 			# print('heff', heff2)
# 			# print('xeff', xeff2)
# 			# print('the singular values', s)
# 			# linear solver
# 			# (1) using lstsq
# 			# mpoj, r, k, s = lstsq(heff2, xeff2)
# 			# if (verbose >= 2):
# 			# 	print('rank: ', heff2.shape[1], '; effective rank:', k)
# 			# end of (1)
# 			# (2) using bicg
# 			A = LinearOperator(shape=(d*dy, d*dy), 
# 				matvec=lambda v:heff2.dot(v.reshape((d, dy))).reshape(d*dy),
# 				rmatvec=lambda v:heff2.T.dot(v.reshape((d, dy))).reshape(d*dy))
# 			x0 = mpo[j].transpose((1,2,3,0))
# 			# res = cg(A=A, b = xeff2.reshape(d*dy), x0=x0.reshape(x0.size), maxiter=100)
# 			# res = lsqr(A=A, b = xeff2.reshape(d*dy), x0=x0.reshape(x0.size))
# 			res = lsqr(A=A, b = xeff2.reshape(d*dy))
# 			mpoj = res[0].reshape((d, dy))
# 			# end of (2)
# 			distance = contract(heff2.dot(mpoj)-2.*xeff2, mpoj, ((0,1),(0,1)))
# 			kvals.append(distance)
# 			if (verbose >= 2):
# 				print('error after updating', distance)
# 				print('norm of mpoj', norm(mpoj.reshape(mpoj.size)))
# 			mpoj = mpoj.reshape(xefft.shape)
# 			mpoj = mpoj.transpose((3,0,2,1))
# 			# # print('error after updating', error_func(heff, xeff, mpoj))
# 			# do svd to mpsj
# 			# mpoj, s, u, bet = svdCompress(mpoj, (2,), 10000, 1.0e-10, useIterSolver=0, verbose=verbose)
# 			# # print('singular values', s)
# 			# mpo[j] = mpoj.transpose((0,1,3,2))
# 			# u = contract(diag(s), u, ((1,),(0,)))
# 			# mpo[j+1] = contract(u, mpo[j+1], ((1,), (1,)))
# 			# mpo[j+1] = mpo[j+1].transpose((1,0,2,3))
# 			# end of do svd to mpsj
# 			mpo[j] = mpoj
# 			# update hs and hhs
# 			for n in range(N):
# 				hs[n][j+1] = updateCleft(hs[n][j], mpo[j], mpsxs[n][j], mpsys[n][j].conj())
# 				hhs[n][j+1] = updateCCleft(hhs[n][j], mpo[j], mpsxs[n][j])
# 		for j in range(L-1, 0, -1):
# 			if (verbose >= 2):
# 				print('sweeping from right to left on site', j)
# 			xeff = zeros((dy, hs[0][j].shape[1], dx, hs[0][j+1].shape[1]))
# 			heff = zeros((dx, hhs[0][j].shape[1], hhs[0][j].shape[2], dx, 
# 				hhs[0][j+1].shape[1], hhs[0][j+1].shape[2]))
# 			for n in range(N):
# 				heff += HHSingleSite(mpsxs[n][j], hhs[n][j], hhs[n][j+1])
# 				xeff += HSingleSite(mpsxs[n][j], mpsys[n][j], hs[n][j], hs[n][j+1])
# 			# print('error before updating', error_func(heff, 2.*xeff, mpo[j]))
# 			# print('mpoj', mpo[j])
# 			hefft = heff.transpose((1,0,4,2,3,5))
# 			xefft = xeff.transpose((1,2,3,0))
# 			xeff2 = xefft.reshape((xefft.shape[0]*xefft.shape[1]*xefft.shape[2], xefft.shape[3]))
# 			d = hefft.shape[0]*hefft.shape[1]*hefft.shape[2]
# 			heff2 = hefft.reshape((d, d))
# 			if (verbose >= 2):
# 				print('conditional number of heff:', cond(heff2))
# 				print('norm of heff', norm(heff2))
# 				print('norm of xeff', norm(xeff2))
# 			# print('heff', heff2)
# 			# print('xeff', xeff2)
# 			# linear solver
# 			# (1) using lstsq
# 			# mpoj, r, k, s = lstsq(heff2, xeff2)
# 			# if (verbose >= 2):
# 			# 	print('rank: ', heff2.shape[1], '; effective rank:', k)
# 			# end of (1)
# 			# (2) using bicg
# 			A = LinearOperator(shape=(d*dy, d*dy), 
# 				matvec=lambda v:heff2.dot(v.reshape((d, dy))).reshape(d*dy),
# 				rmatvec=lambda v:heff2.T.dot(v.reshape((d, dy))).reshape(d*dy))
# 			x0 = mpo[j].transpose((1,2,3,0))
# 			# res = cg(A=A, b = xeff2.reshape(d*dy), x0=x0.reshape(x0.size), maxiter=100)
# 			# res = lsqr(A=A, b = xeff2.reshape(d*dy), x0=x0.reshape(x0.size))
# 			res = lsqr(A=A, b = xeff2.reshape(d*dy))
# 			mpoj = res[0].reshape((d, dy))
# 			# end of (2)
# 			distance = contract(heff2.dot(mpoj)-2.*xeff2, mpoj, ((0,1),(0,1)))
# 			kvals.append(distance)
# 			if (verbose >= 2):
# 				print('error after updating', distance)
# 				print('norm of mpoj', norm(mpoj.reshape(mpoj.size)))
# 			mpoj = mpoj.reshape((xeff.shape[1], xeff.shape[2], xeff.shape[3], xeff.shape[0]))
# 			mpoj = mpoj.transpose((3,0,2,1))
# 			# print('error after updating', error_func(heff, xeff, mpoj))
# 			# do svd to mpsj
# 			# u, s, mpoj, bet = svdCompress(mpoj, (0,2,3), 10000, 1.0e-10, useIterSolver=0, verbose=verbose)
# 			# # print('singular values', s)
# 			# mpo[j] = mpoj.transpose((1,0,2,3))
# 			# u = contract(u, diag(s), ((1,), (0,)))
# 			# mpo[j-1] = contract(mpo[j-1], u, ((2,), (0,)))
# 			# mpo[j-1] = mpo[j-1].transpose((0,1,3,2))
# 			# end of do svd to mpsj
# 			mpo[j] = mpoj
# 			# update hs and hhs
# 			for n in range(N):
# 				hs[n][j] = updateCright(hs[n][j+1], mpo[j], mpsxs[n][j], mpsys[n][j].conj())
# 				hhs[n][j] = updateCCright(hhs[n][j+1], mpo[j], mpsxs[n][j])
# 		if (verbose >= 3):
# 			print('the kvalues are:', kvals)
# 		kvals_all = kvals_all + kvals
# 		kvals = array(kvals)
# 		err = abs(kvals.std() / kvals.mean())
# 		if verbose >= 1:
# 			# print(kvals)
# 			print('the standard error is', err)
# 		if err < tol:
# 			if verbose >= 1:
# 				print('converges after', itr+1, 'sweeps.')
# 			break
# 		itr += 1
# 	# u = contract(u, diag(s), ((1,),(0,)))
# 	# mpo[0] = contract(mpo[0], u, ((2,),(0,)))
# 	# mpo[0] = mpo[0].transpose((0,1,3,2))
# 	return mpo, itr, err, kvals_all

def minimizeMPOOneSiteGD(mpsxs, mpsys, alpha=0.1, step_size=100., D=20, kmax=50, tol=1.0e-9, less_memory=False, verbose=1):
	assert(len(mpsxs) == len(mpsys))
	N = len(mpsxs)
	assert(N > 0)
	L = len(mpsxs[0])
	dx = mpsxs[0][0].shape[1]
	dy = mpsys[0][0].shape[1]
	for i in range(N):
		for j in range(1,L):
			assert(mpsxs[i][j].shape[1] == dx)
			assert(mpsys[i][j].shape[1] == dy)
	# create an initial random mpo
	mpo = createrandommpo(L, dx, dy, D)
	hms = initMPOstorageRight(mpo)
	if less_memory == False:
		hs = [initHstorageRight(mpo, mpsxs[j], mpsys[j]) for j in range(N)]
		hhs = [initHHstorageRight(mpo, mpsxs[j]) for j in range(N)]
	itr = 0
	kvals_all = []
	while (itr < kmax):
		if (verbose >= 1):
			print('we are at', itr, '-th iteration.')
		kvals = []
		if less_memory == True:
			h = [computeHstorageLeft(mpo, mpsxs[n], mpsys[n], 0) for n in range(N)]
			hh = [computeHHstorageLeft(mpo, mpsxs[n], 0) for n in range(N)]
		for j in range(L-1):
			if (verbose >= 2):
				print('sweeping from left to right on site', j)
			xeff = zeros((dy, mpo[j].shape[1], dx, mpo[j].shape[2]))
			heff = zeros((dx, mpo[j].shape[1], mpo[j].shape[1], dx, 
				mpo[j].shape[2], mpo[j].shape[2]))
			if less_memory==False:
				for n in range(N):
					heff += HHSingleSite(mpsxs[n][j], hhs[n][j], hhs[n][j+1])
					xeff += HSingleSite(mpsxs[n][j], mpsys[n][j], hs[n][j], hs[n][j+1])
			else:
				for n in range(N):
					tmp = computeHHstorageRight(mpo, mpsxs[n], j)
					heff += HHSingleSite(mpsxs[n][j], hh[n], tmp)
					tmp = computeHstorageRight(mpo, mpsxs[n], mpsys[n], j)
					xeff += HSingleSite(mpsxs[n][j], mpsys[n][j], h[n], tmp)
			mpoeff = contract(hms[j], hms[j+1], ((), ()))
			mpoeff = contract(mpoeff, eye(dx), ((), ()))
			mpoeff = mpoeff.transpose((4,0,1,5,2,3))
			heff += alpha*mpoeff
			# gradient descent method
			dmpoj = contract(mpo[j], heff, ((1,2,3),(2,5,3)))
			dmpoj = xeff.transpose((0,1,3,2)) - dmpoj.transpose((0,2,3,1))
			print('norm of gradient', norm(dmpoj.reshape(dmpoj.size))) 
			mpo[j] += step_size*dmpoj
			distance = error_func(heff, 2.*xeff, mpo[j])
			kvals.append(distance)
			if (verbose >= 2):
				print('error after updating', distance)
			#------do qr to mpoj--------
			mpo[j], u = qrCompress('L', mpo[j], (2,))
			mpo[j] = mpo[j].transpose((0,1,3,2))
			mpo[j+1] = contract(u, mpo[j+1], ((1,), (1,)))
			mpo[j+1] = mpo[j+1].transpose((1,0,2,3))
			#------end of do qr to mpoj---------
			#------do nothing to mpoj--------
			# mpo[j] = mpoj
			# update hs and hhs
			if less_memory==False:
				for n in range(N):
					hs[n][j+1] = updateCleft(hs[n][j], mpo[j], mpsxs[n][j], mpsys[n][j].conj())
					hhs[n][j+1] = updateCCleft(hhs[n][j], mpo[j], mpsxs[n][j])
			else:
				for n in range(N):
					h[n] = updateCleft(h[n], mpo[j], mpsxs[n][j], mpsys[n][j].conj())
					hh[n] = updateCCleft(hh[n], mpo[j], mpsxs[n][j])
			# update hms(mpostorage)
			hms[j+1] = updateMPOleft(hms[j], mpo[j])
		if less_memory == True:
			h = [computeHstorageRight(mpo, mpsxs[n], mpsys[n], L-1) for n in range(N)]
			hh = [computeHHstorageRight(mpo, mpsxs[n], L-1) for n in range(N)]
		for j in range(L-1, 0, -1):
			if (verbose >= 2):
				print('sweeping from right to left on site', j)
			xeff = zeros((dy, mpo[j].shape[1], dx, mpo[j].shape[2]))
			heff = zeros((dx, mpo[j].shape[1], mpo[j].shape[1], dx, 
				mpo[j].shape[2], mpo[j].shape[2]))
			if less_memory==False:
				for n in range(N):
					heff += HHSingleSite(mpsxs[n][j], hhs[n][j], hhs[n][j+1])
					xeff += HSingleSite(mpsxs[n][j], mpsys[n][j], hs[n][j], hs[n][j+1])
			else:
				for n in range(N):
					tmp = computeHHstorageLeft(mpo, mpsxs[n], j)
					heff += HHSingleSite(mpsxs[n][j], tmp, hh[n])
					tmp = computeHstorageLeft(mpo, mpsxs[n], mpsys[n], j)
					xeff += HSingleSite(mpsxs[n][j], mpsys[n][j], tmp, h[n])
			mpoeff = contract(hms[j], hms[j+1], ((), ()))
			mpoeff = contract(mpoeff, eye(dx), ((), ()))
			mpoeff = mpoeff.transpose((4,0,1,5,2,3))
			heff += alpha*mpoeff
			# gradient descent method
			dmpoj = contract(mpo[j], heff, ((1,2,3),(2,5,3)))
			dmpoj = xeff.transpose((0,1,3,2)) - dmpoj.transpose((0,2,3,1))
			print('norm of gradient', norm(dmpoj.reshape(dmpoj.size))) 
			mpo[j] += step_size*dmpoj
			distance = error_func(heff, 2.*xeff, mpo[j])
			kvals.append(distance)
			if (verbose >= 2):
				print('error after updating', distance)
			# --------do qr to mpoj------------
			u, mpo[j] = qrCompress('R', mpo[j], (0,2,3))
			mpo[j] = mpo[j].transpose((1,0,2,3))
			mpo[j-1] = contract(mpo[j-1], u, ((2,), (0,)))
			mpo[j-1] = mpo[j-1].transpose((0,1,3,2))
			# --------end of do qr to mpoj--------
			# update hs and hhs
			if less_memory==False:
				for n in range(N):
					hs[n][j] = updateCright(hs[n][j+1], mpo[j], mpsxs[n][j], mpsys[n][j].conj())
					hhs[n][j] = updateCCright(hhs[n][j+1], mpo[j], mpsxs[n][j])
			else:
				for n in range(N):
					h[n] = updateCright(h[n], mpo[j], mpsxs[n][j], mpsys[n][j].conj())
					hh[n] = updateCCright(hh[n], mpo[j], mpsxs[n][j])
			hms[j] = updateMPOright(hms[j+1], mpo[j])
		if (verbose >= 3):
			print('the kvalues are:', kvals)
		kvals_all = kvals_all + kvals
		kvals = array(kvals)
		err = abs(kvals.std() / kvals.mean())
		if verbose >= 1:
			# print(kvals)
			print('the standard error is', err)
		if err < tol:
			if verbose >= 1:
				print('converges after', itr+1, 'sweeps.')
			break
		itr += 1
	return mpo, itr, err, kvals_all

def minimizeMPOOneSite2(mpsxs, mpsys, decay=1.0e-4, D=20, kmax=50, tol=1.0e-9, verbose=1):
	"""
	gradient descent algorithm
	"""
	assert(len(mpsxs) == len(mpsys))
	N = len(mpsxs)
	assert(N > 0)
	L = len(mpsxs[0])
	dx = mpsxs[0][0].shape[1]
	dy = mpsys[0][0].shape[1]
	for i in range(N):
		for j in range(1,L):
			assert(mpsxs[i][j].shape[1] == dx)
			assert(mpsys[i][j].shape[1] == dy)
	# create an initial random mpo
	mpo = createrandommpo(L, dx, dy, D)
	hs = [initHstorageRight(mpo, mpsxs[j], mpsys[j]) for j in range(N)]
	hhs = [initHHstorageRight(mpo, mpsxs[j]) for j in range(N)]
	itr = 0
	while (itr < kmax):
		if (verbose >= 1):
			print('we are at', itr, '-th iteraction.')
		kvals = []
		for j in range(L-1):
			if (verbose >= 2):
				print('sweeping from left to right on site', j)
			xeff = zeros((dy, hs[0][j].shape[1], dx, hs[0][j+1].shape[1]))
			heff = zeros((dx, hhs[0][j].shape[1], hhs[0][j].shape[2], dx, 
				hhs[0][j+1].shape[1], hhs[0][j+1].shape[2]))
			for n in range(N):
				heff += HHSingleSite(mpsxs[n][j], hhs[n][j], hhs[n][j+1])
				xeff += HSingleSite(mpsxs[n][j], mpsys[n][j], hs[n][j], hs[n][j+1])
			print('error before updating', error_func(heff, 2.*xeff, mpo[j]))
			dmpoj = contract(mpo[j], heff, ((1,2,3),(2,5,3)))
			dmpoj = xeff.transpose((0,1,3,2)) - dmpoj.transpose((0,2,3,1))
			# print('norm of gradient', norm(dmpoj.reshape(dmpoj.size))) 
			mpo[j] += decay*dmpoj
			distance = error_func(heff, 2.*xeff, mpo[j])
			kvals.append(distance)
			if (verbose >= 2):
				print('error after updating', distance)
			# mpoj, s, u, bet = svdCompress(mpo[j], (2,), -1, tol, useIterSolver=0, verbose=verbose)
			# mpo[j] = mpoj.transpose((0,1,3,2))
			# u = contract(diag(s), u, ((1,),(0,)))
			# mpo[j+1] = contract(u, mpo[j+1], ((1,), (1,)))
			# mpo[j+1] = mpo[j+1].transpose((1,0,2,3))
			# update hs and hhs
			for n in range(N):
				hs[n][j+1] = updateCleft(hs[n][j], mpo[j], mpsxs[n][j], mpsys[n][j].conj())
				hhs[n][j+1] = updateCCleft(hhs[n][j], mpo[j], mpsxs[n][j])
		for j in range(L-1, 0, -1):
			if (verbose >= 2):
				print('sweeping from right to left on site', j)
			xeff = zeros((dy, hs[0][j].shape[1], dx, hs[0][j+1].shape[1]))
			heff = zeros((dx, hhs[0][j].shape[1], hhs[0][j].shape[2], dx, 
				hhs[0][j+1].shape[1], hhs[0][j+1].shape[2]))
			for n in range(N):
				heff += HHSingleSite(mpsxs[n][j], hhs[n][j], hhs[n][j+1])
				xeff += HSingleSite(mpsxs[n][j], mpsys[n][j], hs[n][j], hs[n][j+1])
			# print('error before updating', error_func(heff, xeff, mpo[j]))
			dmpoj = contract(mpo[j], heff, ((1,2,3),(2,5,3)))
			dmpoj = xeff.transpose((0,1,3,2)) - dmpoj.transpose((0,2,3,1)) 
			# print('norm of gradient', norm(dmpoj.reshape(dmpoj.size))) 
			mpo[j] += decay*dmpoj
			distance = error_func(heff, 2.*xeff, mpo[j])
			kvals.append(distance)
			if (verbose >= 2):
				print('error after updating', distance)
			# u, s, mpoj, bet = svdCompress(mpo[j], (0,2,3), -1, tol, useIterSolver=0, verbose=verbose)
			# mpo[j] = mpoj.transpose((1,0,2,3))
			# u = contract(u, diag(s), ((1,), (0,)))
			# mpo[j-1] = contract(mpo[j-1], u, ((2,), (0,)))
			# mpo[j-1] = mpo[j-1].transpose((0,1,3,2))
			# update hs and hhs
			for n in range(N):
				hs[n][j] = updateCright(hs[n][j+1], mpo[j], mpsxs[n][j], mpsys[n][j].conj())
				hhs[n][j] = updateCCright(hhs[n][j+1], mpo[j], mpsxs[n][j])
		if (verbose >= 3):
			print('the kvalues are:', kvals)
		kvals = array(kvals)
		err = amax(kvals)
		if verbose >= 1:
			# print(kvals)
			print('the maximum error is', err)
		if err < tol:
			if verbose >= 1:
				print('converges after', itr+1, 'sweeps.')
			break
		itr += 1
	# u = contract(u, diag(s), ((1,),(0,)))
	# mpo[0] = contract(mpo[0], u, ((2,),(0,)))
	# mpo[0] = mpo[0].transpose((0,1,3,2))
	return mpo

def HTwoSite(mpsxj1, mpsxj2, mpsyj1, mpsyj2, hleft, hright):
	"""
	output
	*******0**5****
	***1---M--M---4
	*******2**3****
	"""
	hnew = contract(hleft, mpsxj1, ((2,),(0,)))
	hnew = contract(hnew, mpsxj2, ((3,),(0,)))
	hnew = contract(hnew, hright, ((4,),(2,)))
	hnew = contract(mpsyj1, hnew, ((0,),(0,)))
	hnew = contract(hnew, mpsyj2, ((1,5),(0,2)))
	return hnew

def HHTwoSite(mpsj1, mpsj2, hleft, hright):
	"""
	output
	*******0**7*******
	***1---M--M---5
	***2---M--M---6
	*******3**4*******
	"""
	hnew = contract(mpsj1.conj(), hleft, ((0,), (0,)))
	hnew = contract(hnew, mpsj1, ((4,), (0,)))
	hnew = contract(hnew, mpsj2, ((5,), (0,)))
	hnew = contract(hnew, hright, ((6,), (3,)))
	hnew = contract(hnew, mpsj2.conj(), ((1,6),(0,2)))
	return hnew

def minimizeMPOTwoSite(mpsxs, mpsys, alpha=0.1, D=20, kmax=50, 
	tol=1.0e-9, svdcutoff=1.0e-12, verbose=1):
	assert(len(mpsxs) == len(mpsys))
	N = len(mpsxs)
	assert(N > 0)
	L = len(mpsxs[0])
	dx = mpsxs[0][0].shape[1]
	dy = mpsys[0][0].shape[1]
	for i in range(N):
		for j in range(1,L):
			assert(mpsxs[i][j].shape[1] == dx)
			assert(mpsys[i][j].shape[1] == dy)
	# create an initial random mpo
	mpo = createrandommpo(L, dx, dy, D)
	hs = [initHstorageRight(mpo, mpsxs[j], mpsys[j]) for j in range(N)]
	hhs = [initHHstorageRight(mpo, mpsxs[j]) for j in range(N)]
	hms = initMPOstorageRight(mpo)
	itr = 0
	kvals_all = []
	while (itr < kmax):
		if (verbose >= 1):
			print('we are at', itr, '-th iteraction.')
		kvals = []
		for j in range(L-2):
			if (verbose >= 2):
				print('sweeping from left to right on site', j)
			xeff = zeros((dy, hs[0][j].shape[1], dx, dx, hs[0][j+2].shape[1], dy))
			heff = zeros((dx, hhs[0][j].shape[1], hhs[0][j].shape[2], 
				dx, dx, hhs[0][j+2].shape[1], hhs[0][j+2].shape[2], dy))
			for n in range(N):
				heff += HHTwoSite(mpsxs[n][j], mpsxs[n][j+1], hhs[n][j], hhs[n][j+2])
				xeff += HTwoSite(mpsxs[n][j], mpsxs[n][j+1], mpsys[n][j], 
					mpsys[n][j+1], hs[n][j], hs[n][j+2])
			hefft = heff.transpose((1,0,7,5,2,3,4,6))
			mpoeff = contract(hms[j], hms[j+2], ((), ()))
			mpoeff = contract(mpoeff, eye(dx), ((), ()))
			mpoeff = contract(mpoeff, eye(dx), ((), ()))
			mpoeff = mpoeff.transpose((0,4,6,2,1,5,7,3))
			hefft += alpha*mpoeff
			xefft = xeff.transpose((1,2,3,4,0,5))
			d = xefft.shape[0]*xefft.shape[1]*xefft.shape[2]*xefft.shape[3]
			xeff2 = xefft.reshape((d, xefft.shape[4]*xefft.shape[5]))
			heff2 = hefft.reshape((d, d))
			# if (verbose >= 2):
			# 	print('conditional number of heff:', cond(heff2))
			# 	print('norm of heff', norm(heff2))
			# 	print('norm of xeff', norm(xeff2))
			# s = svd(a=heff2, compute_uv=False)
			# print('heff', heff2)
			# print('xeff', xeff2)
			# print('the singular values', s)
			# linear solver
			# (1) using lstsq
			mpoj = solve(heff2, xeff2)
			# if (verbose >= 2):
			# 	print('rank: ', heff2.shape[1], '; effective rank:', k)
				# print('norm of mpoj', norm(mpoj.reshape(mpoj.size)))
			# end of (1)
			# (2) using bicg
			# A = LinearOperator(shape=(d*dy, d*dy), 
			# 	matvec=lambda v:heff2.dot(v.reshape((d, dy))).reshape(d*dy),
			# 	rmatvec=lambda v:heff2.T.dot(v.reshape((d, dy))).reshape(d*dy))
			# x0 = mpo[j].transpose((1,2,3,0))
			# res = bicg(A=A, b = xeff2.reshape(d*dy), x0=x0.reshape(x0.size))
			# mpoj = res[0].reshape((d, dy))
			# end of (2)
			distance = contract(heff2.dot(mpoj)-2.*xeff2, mpoj, ((0,1),(0,1)))
			kvals.append(distance)
			if (verbose >= 2):
				print('error is', distance)
			mpoj = mpoj.reshape(xefft.shape)
			mpoj, s, u, bet = svdCompress(mpoj, (2,3,5), D, 
				svdcutoff*abs(distance), useIterSolver=0, verbose=verbose)
			mpo[j] = mpoj.transpose((2,0,3,1))
			u = contract(diag(s), u, ((1,),(0,)))
			mpo[j+1] = u.transpose((3,0,2,1))
			# update hs and hhs
			for n in range(N):
				hs[n][j+1] = updateCleft(hs[n][j], mpo[j], mpsxs[n][j], mpsys[n][j].conj())
				hhs[n][j+1] = updateCCleft(hhs[n][j], mpo[j], mpsxs[n][j])
			# update hms(mpostorage)
			hms[j+1] = updateMPOleft(hms[j], mpo[j])
		for j in range(L-2, 0, -1):
			if (verbose >= 2):
				print('sweeping from right to left on site', j)
			xeff = zeros((dy, hs[0][j].shape[1], dx, dx, hs[0][j+2].shape[1], dy))
			heff = zeros((dx, hhs[0][j].shape[1], hhs[0][j].shape[2], 
				dx, dx, hhs[0][j+2].shape[1], hhs[0][j+2].shape[2], dy))
			for n in range(N):
				heff += HHTwoSite(mpsxs[n][j], mpsxs[n][j+1], hhs[n][j], hhs[n][j+2])
				xeff += HTwoSite(mpsxs[n][j], mpsxs[n][j+1], mpsys[n][j], 
					mpsys[n][j+1], hs[n][j], hs[n][j+2])
			hefft = heff.transpose((1,0,7,5,2,3,4,6))
			mpoeff = contract(hms[j], hms[j+2], ((), ()))
			mpoeff = contract(mpoeff, eye(dx), ((), ()))
			mpoeff = contract(mpoeff, eye(dx), ((), ()))
			mpoeff = mpoeff.transpose((0,4,6,2,1,5,7,3))
			hefft += alpha*mpoeff
			xefft = xeff.transpose((1,2,3,4,0,5))
			d = xefft.shape[0]*xefft.shape[1]*xefft.shape[2]*xefft.shape[3]
			xeff2 = xefft.reshape((d, xefft.shape[4]*xefft.shape[5]))
			heff2 = hefft.reshape((d, d))
			# if (verbose >= 2):
			# 	print('conditional number of heff:', cond(heff2))
			# 	print('norm of heff', norm(heff2))
			# 	print('norm of xeff', norm(xeff2))
			# print('heff', heff2)
			# print('xeff', xeff2)
			# linear solver
			# (1) using lstsq
			# mpoj, r, k, s = lstsq(heff2, xeff2)
			mpoj = solve(heff2, xeff2)
			# if (verbose >= 2):
			# 	print('rank: ', heff2.shape[1], '; effective rank:', k)
			# 	print('norm of mpoj', norm(mpoj.reshape(mpoj.size)))
			# end of (1)
			# (2) using bicg
			# A = LinearOperator(shape=(d*dy, d*dy), 
			# 	matvec=lambda v:heff2.dot(v.reshape((d, dy))).reshape(d*dy),
			# 	rmatvec=lambda v:heff2.T.dot(v.reshape((d, dy))).reshape(d*dy))
			# x0 = mpo[j].transpose((1,2,3,0))
			# res = bicg(A=A, b = xeff2.reshape(d*dy), x0=x0.reshape(x0.size))
			# mpoj = res[0].reshape((d, dy))
			# end of (2)
			distance = contract(heff2.dot(mpoj)-2.*xeff2, mpoj, ((0,1),(0,1)))
			kvals.append(distance)
			if (verbose >= 2):
				print('error is', distance)
			mpoj = mpoj.reshape(xefft.shape)
			u, s, mpoj, bet = svdCompress(mpoj, (2,3,5), D, 
				svdcutoff*abs(distance), useIterSolver=0, verbose=verbose)
			# print('singular values', s)
			mpo[j+1] = mpoj.transpose((3,0,2,1))
			u = contract(u, diag(s), ((3,), (0,)))
			mpo[j] = u.transpose((2,0,3,1))
			# update hs and hhs
			for n in range(N):
				hs[n][j+1] = updateCright(hs[n][j+2], mpo[j+1], mpsxs[n][j+1], mpsys[n][j+1].conj())
				hhs[n][j+1] = updateCCright(hhs[n][j+2], mpo[j+1], mpsxs[n][j+1])
			# update hms(mpostorage)
			hms[j+1] = updateMPOright(hms[j+2], mpo[j+1])
		if (verbose >= 3):
			print('the kvalues are:', kvals)
		kvals_all = kvals_all + kvals
		kvals = array(kvals)
		err = abs(kvals.std() / kvals.mean())
		if verbose >= 1:
			# print(kvals)
			print('the standard error is', err)
		if err < tol:
			if verbose >= 1:
				print('converges after', itr+1, 'sweeps.')
			break
		itr += 1
	# u = contract(u, diag(s), ((1,),(0,)))
	# mpo[0] = contract(mpo[0], u, ((2,),(0,)))
	# mpo[0] = mpo[0].transpose((0,1,3,2))
	return mpo, itr, err, kvals_all

# def minimizeMPOTwoSite2(mpsxs, mpsys, alpha=1.0e-4, D=20, kmax=50, tol=1.0e-9, verbose=1):
# 	assert(len(mpsxs) == len(mpsys))
# 	N = len(mpsxs)
# 	assert(N > 0)
# 	L = len(mpsxs[0])
# 	dx = mpsxs[0][0].shape[1]
# 	dy = mpsys[0][0].shape[1]
# 	for i in range(N):
# 		for j in range(1,L):
# 			assert(mpsxs[i][j].shape[1] == dx)
# 			assert(mpsys[i][j].shape[1] == dy)
# 	# create an initial random mpo
# 	mpo = createrandommpo(L, dx, dy, D)
# 	mpo.svdCompress()
# 	hs = [initHstorageRight(mpo, mpsxs[j], mpsys[j]) for j in range(N)]
# 	hhs = [initHHstorageRight(mpo, mpsxs[j]) for j in range(N)]
# 	# for h in hhs:
# 	# 	print('norm', amax(h[50]))
# 	# assert(False)
# 	itr = 0
# 	kvals_all = []
# 	while (itr < kmax):
# 		if (verbose >= 1):
# 			print('we are at', itr, '-th iteraction.')
# 		kvals = []
# 		for j in range(L-2):
# 			if (verbose >= 2):
# 				print('sweeping from left to right on site', j)
# 			xeff = zeros((dy, hs[0][j].shape[1], dx, dx, hs[0][j+2].shape[1], dy))
# 			heff = zeros((dx, hhs[0][j].shape[1], hhs[0][j].shape[2], 
# 				dx, dx, hhs[0][j+2].shape[1], hhs[0][j+2].shape[2], dy))
# 			for n in range(N):
# 				heff += HHTwoSite(mpsxs[n][j], mpsxs[n][j+1], hhs[n][j], hhs[n][j+2])
# 				xeff += HTwoSite(mpsxs[n][j], mpsxs[n][j+1], mpsys[n][j], 
# 					mpsys[n][j+1], hs[n][j], hs[n][j+2])
# 			hefft = heff.transpose((1,0,7,5,2,3,4,6))
# 			xefft = xeff.transpose((1,2,3,4,0,5))
# 			d = xefft.shape[0]*xefft.shape[1]*xefft.shape[2]*xefft.shape[3]
# 			xeff2 = xefft.reshape((d, xefft.shape[4]*xefft.shape[5]))
# 			heff2 = hefft.reshape((d, d))
# 			print('conditional number of heff:', cond(heff2))
# 			# s = svd(a=heff2, compute_uv=False)
# 			# print('heff', heff2)
# 			# print('xeff', xeff2)
# 			# print('the singular values', s)
# 			# linear solver
# 			# (1) using lstsq
# 			mpoj, r, k, s = lstsq(heff2, xeff2)
# 			if (verbose >= 2):
# 				print('rank: ', heff2.shape[1], '; effective rank:', k)
# 			# end of (1)
# 			# (2) using bicg
# 			# A = LinearOperator(shape=(d*dy, d*dy), 
# 			# 	matvec=lambda v:heff2.dot(v.reshape((d, dy))).reshape(d*dy),
# 			# 	rmatvec=lambda v:heff2.T.dot(v.reshape((d, dy))).reshape(d*dy))
# 			# x0 = mpo[j].transpose((1,2,3,0))
# 			# res = bicg(A=A, b = xeff2.reshape(d*dy), x0=x0.reshape(x0.size))
# 			# mpoj = res[0].reshape((d, dy))
# 			# end of (2)
# 			distance = contract(heff2.dot(mpoj)-2.*xeff2, mpoj, ((0,1),(0,1)))
# 			kvals.append(distance)
# 			if (verbose >= 2):
# 				print('error is', distance)
# 			mpoj = mpoj.reshape(xefft.shape)
# 			mpoj, s, u, bet = svdCompress(mpoj, (2,3,5), D, 1.0e-10, useIterSolver=0, verbose=verbose)
# 			# print('singular values', s)
# 			mpo[j] = mpoj.transpose((2,0,3,1))
# 			u = contract(diag(s), u, ((1,),(0,)))
# 			mpo[j+1] = u.transpose((3,0,2,1))
# 			# update hs and hhs
# 			for n in range(N):
# 				hs[n][j+1] = updateCleft(hs[n][j], mpo[j], mpsxs[n][j], mpsys[n][j].conj())
# 				hhs[n][j+1] = updateCCleft(hhs[n][j], mpo[j], mpsxs[n][j])
# 		for j in range(L-2, 0, -1):
# 			if (verbose >= 2):
# 				print('sweeping from right to left on site', j)
# 			xeff = zeros((dy, hs[0][j].shape[1], dx, dx, hs[0][j+2].shape[1], dy))
# 			heff = zeros((dx, hhs[0][j].shape[1], hhs[0][j].shape[2], 
# 				dx, dx, hhs[0][j+2].shape[1], hhs[0][j+2].shape[2], dy))
# 			for n in range(N):
# 				heff += HHTwoSite(mpsxs[n][j], mpsxs[n][j+1], hhs[n][j], hhs[n][j+2])
# 				xeff += HTwoSite(mpsxs[n][j], mpsxs[n][j+1], mpsys[n][j], 
# 					mpsys[n][j+1], hs[n][j], hs[n][j+2])
# 			hefft = heff.transpose((1,0,7,5,2,3,4,6))
# 			xefft = xeff.transpose((1,2,3,4,0,5))
# 			d = xefft.shape[0]*xefft.shape[1]*xefft.shape[2]*xefft.shape[3]
# 			xeff2 = xefft.reshape((d, xefft.shape[4]*xefft.shape[5]))
# 			heff2 = hefft.reshape((d, d))
# 			print('conditional number of heff:', cond(heff2))
# 			# print('heff', heff2)
# 			# print('xeff', xeff2)
# 			# linear solver
# 			# (1) using lstsq
# 			mpoj, r, k, s = lstsq(heff2, xeff2)
# 			if (verbose >= 2):
# 				print('rank: ', heff2.shape[1], '; effective rank:', k)
# 			# end of (1)
# 			# (2) using bicg
# 			# A = LinearOperator(shape=(d*dy, d*dy), 
# 			# 	matvec=lambda v:heff2.dot(v.reshape((d, dy))).reshape(d*dy),
# 			# 	rmatvec=lambda v:heff2.T.dot(v.reshape((d, dy))).reshape(d*dy))
# 			# x0 = mpo[j].transpose((1,2,3,0))
# 			# res = bicg(A=A, b = xeff2.reshape(d*dy), x0=x0.reshape(x0.size))
# 			# mpoj = res[0].reshape((d, dy))
# 			# end of (2)
# 			distance = contract(heff2.dot(mpoj)-2.*xeff2, mpoj, ((0,1),(0,1)))
# 			kvals.append(distance)
# 			if (verbose >= 2):
# 				print('error is', distance)
# 			mpoj = mpoj.reshape(xefft.shape)
# 			u, s, mpoj, bet = svdCompress(mpoj, (2,3,5), D, 1.0e-10, useIterSolver=0, verbose=verbose)
# 			# print('singular values', s)
# 			mpo[j+1] = mpoj.transpose((3,0,2,1))
# 			u = contract(u, diag(s), ((3,), (0,)))
# 			mpo[j] = u.transpose((2,0,3,1))
# 			# update hs and hhs
# 			for n in range(N):
# 				hs[n][j+1] = updateCright(hs[n][j+2], mpo[j+1], mpsxs[n][j+1], mpsys[n][j+1].conj())
# 				hhs[n][j+1] = updateCCright(hhs[n][j+2], mpo[j+1], mpsxs[n][j+1])
# 		if (verbose >= 3):
# 			print('the kvalues are:', kvals)
# 		kvals_all = kvals_all + kvals
# 		kvals = array(kvals)
# 		err = abs(kvals.std() / kvals.mean())
# 		if verbose >= 1:
# 			# print(kvals)
# 			print('the maximum error is', err)
# 		if err < tol:
# 			if verbose >= 1:
# 				print('converges after', itr+1, 'sweeps.')
# 			break
# 		itr += 1
# 	# u = contract(u, diag(s), ((1,),(0,)))
# 	# mpo[0] = contract(mpo[0], u, ((2,),(0,)))
# 	# mpo[0] = mpo[0].transpose((0,1,3,2))
# 	return mpo, itr, err, kvals_all

# svd approximation

# def mpo_from_pair_mps(mpsx, mpsy):
# 	"""
# 	generate |x><y| (mpo form) from |x> and |y>
# 	"""
# 	assert(len(mpsx) == len(mpsy))
# 	L = len(mpsx)
# 	mpo = [None]*L
# 	for j in range(L):
# 		dx = mpsx[j].size
# 		dy = mpsy[j].size
# 		mpo[j] = reshape(kron(mpsy[j], mpsx[j]), (dy, 1, 1, dx))
# 	return MPO(mpo)

# def mps_from_pair_xy(mpsx, mpsy):
# 	"""
# 	generate |x,y> (mps form) from |x> and |y>
# 	"""
# 	assert(len(mpsx) == len(mpsy))
# 	L = len(mpsx)
# 	mps = [None]*L
# 	for j in range(L):
# 		dx = mpsx[j].size
# 		dy = mpsy[j].size
# 		mps[j] = reshape(kron(mpsy[j], mpsx[j]), (1, dy*dx, 1))
# 	return MPS(mps)

# def reshapeMPS2MPO(mps, dx, dy):
# 	"""
# 	mapping |x,y> to |x><y|
# 	"""
# 	L = len(mps)
# 	mpo = [None]*L
# 	for j in range(L):
# 		assert(mps[j].shape[1] == dx*dy)
# 		mpoj = mps[j].reshape((mps[j].shape[0], dy, dx, mps[j].shape[2]))
# 		mpo[j] = mpoj.transpose((1, 0, 3, 2))
# 	return MPO(mpo)

# def reshapeMPO2MPS(mpo, dx, dy):
# 	"""
# 	mapping |x><y| to |x,y>
# 	"""
# 	L = len(mpo)
# 	mps = [None]*L
# 	for j in range(L):
# 		assert(mpo[j].shape[0]==dy)
# 		assert(mpo[j].shape[3]==dx)
# 		mpsj = mpo[j].transpose((1, 0, 3, 2))
# 		mps[j] = mpsj.reshape((mpsj.shape[0], dx*dy, mpsj.shape[3]))
# 	return MPS(mps)

# def validate_mps_input(mpsxs, mpsys):
# 	"""
# 	make sure mpsxs and mpsyx are valid,
# 	and get the information of N, L, dx, dy
# 	"""
# 	N = len(mpsxs)
# 	#N is number of samples
# 	assert(N == len(mpsys))
# 	assert(N > 0)
# 	L = len(mpsxs[0])
# 	# L is number of sites
# 	assert(L > 0)
# 	for j in range(N):
# 		assert(len(mpsxs[j]) == L)
# 		assert(len(mpsys[j]) == L)
# 	dx = mpsxs[0][0].shape[1]
# 	dy = mpsys[0][0].shape[1]
# 	for i in range(N):
# 		for j in range(L):
# 			assert(mpsxs[i][j].shape[1] == dx)
# 			assert(mpsys[i][j].shape[1] == dy)
# 	return N, L, dx, dy

# def train_by_svd_compress(mpsxs, mpsys, D, verbose=0):
# 	"""
# 	mpsxs and mpsyx are two lists of mps
# 	"""
# 	#N is number of samples
# 	N, L, dx, dy = validate_mps_input(mpsxs, mpsys)
# 	# kron mpsx and mpsy
# 	mpsAs=[mps_from_pair_xy(mpsxs[j], mpsys[j]) for j in range(N)]
# 	mpoall = createEmptyMPO(L)
# 	for j in range(N):
# 		mpoall += reshapeMPS2MPO(mpsAs[j], dx, dy)
# 	mpoall.svdCompress(maxbonddimension=D, verbose=verbose)
# 	return mpoall

# def train_by_iterative_compress(mpsxs, mpsys, D, verbose=0):
# 	N, L, dx, dy = validate_mps_input(mpsxs, mpsys)
# 	mpsAs=[mps_from_pair_xy(mpsxs[j], mpsys[j]) for j in range(N)]
# 	mpsB = createrandommps(L, dx*dy, D)
# 	mpsB = reduceMPSOneSite(mpsAs, mpsB, verbose=verbose)
# 	return reshapeMPS2MPO(mpsB, dx, dy)

# def error(mpoall, mpsx, mpsy):
# 	temp = mpoall.dot(mpsx) - mpsy
# 	return temp.norm(False)

# def predict(mpoall, x):
# 	"""
# 	given a x, predict a y
# 	assuming x is an array of integers
# 	"""
# 	assert(len(mpoall) > 0)
# 	dy = mpoall[0].shape[0]
# 	dx = mpoall[0].shape[3]
# 	for j in range(1, len(mpoall)):
# 		assert(dy == mpoall[j].shape[0])
# 		assert(dx == mpoall[j].shape[3])
# 	lattice = simple_uniform_lattice(len(mpoall), eye(dx))
# 	mpsx = lattice.generateProdMPS(x)
# 	mpsy = mpoall.dot(mpsx)
# 	# compress mpsy to bond dimension 1
# 	# print(mpsy)
# 	mpsy.svdCompress(maxbonddimension=1)
# 	# print(mpsy.svectors)
# 	# mpsy may have loss norm after compress, renormalize
# 	# it to be 1 again.
# 	mpsy /= mpsy.norm(False)
# 	mpsy.svdCompress()
# 	# temp = createrandommps(len(mpoall), dy, 2)
# 	# temp.svdCompress(maxbonddimension=1)
# 	for j in range(len(mpoall)):
# 		assert(mpsy[j].shape[1] == dy)
# 		# print(temp[j].shape)
# 	# 	# print(mpsy[j].shape)
# 	# mpsy = reduceOneSite(mpsy, temp)
# 	# print(mpsy[-1])
# 	y = [None]*len(mpsy)
# 	for j in range(len(mpsy)):
# 		temp = mpsy[j].reshape(dy)
# 		y[j] = argmax(temp)
# 	return y
