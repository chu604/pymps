# -*- coding: utf-8 -*-
# @Author: 1000787
# @Date:   2017-05-30 15:25:38
# @Last Modified by:   1000787
# @Last Modified time: 2018-03-07 17:03:11
from .DTensor import DTensor, contract
from numpy import zeros, sqrt

# update heff from right using mpo
def updateCright(hold, mpoj, mpsAj, mpsBj):
	Hnew = contract(mpsBj, hold, axes=((2,),(0,)))
	if mpoj is None:
		return contract(Hnew, mpsAj, axes=((1,3),(1,2)))
	else:
		Hnew = contract(Hnew, mpoj, axes=((1,2),(0,2)))
		return contract(Hnew, mpsAj, axes=((3,1),(1,2)))

# update heff from left using mpo
def updateCleft(hold, mpoj, mpsAj, mpsBj):
	Hnew = contract(hold, mpsAj, axes=((2,),(0,)))
	if mpoj is None:
		return contract(mpsBj, Hnew, axes=((0,1),(0,2)))
	else:
		Hnew = contract(mpoj, Hnew, axes=((1,3),(1,2)))
		return contract(mpsBj, Hnew, axes=((0,1),(2,0)))


def updateClefth1h2(hold, obj, mpsAj, mpsBj):
	if hold is None:
		if obj is None:
			return contract(mpsBj, mpsAj, ((0,1),(0,1)))
		else:
			Hnew = contract(obj, mpsAj, ((1,),(1,)))
			return contract(mpsBj, Hnew, ((0,1),(1,0)))
	else:
		Hnew = contract(hold, mpsAj, ((1,),(0,)))
		if obj is None:
			return contract(mpsBj, Hnew, ((0,1),(0,1)))
		else:
			Hnew = contract(obj, Hnew, ((1,),(1,)))
			return contract(mpsBj, Hnew, ((0,1),(1,0)))

def updateCrighth1h2(hold, obj, mpsAj, mpsBj):
	if hold is None:
		if obj is None:
			return contract(mpsBj, mpsAj, ((1,2),(1,2)))
		else:
			Hnew = contract(obj, mpsAj, ((1,),(1,)))
			return contract(mpsBj, Hnew, ((1,2),(0, 2)))
	else:
		Hnew = contract(mpsBj, hold, ((2,),(0,)))
		if obj is None:
			return contract(Hnew, mpsAj, ((1,2),(1,2)))
		else:
			Hnew = contract(Hnew, obj, ((1,),(0,)))
			return contract(Hnew, mpsAj, ((2,1),(1,2)))

def trace2(m):
	assert(m.ndim==2)
	s1 = m.shape[0]
	s2 = m.shape[1]
	h = zeros(s1, m.dtype)
	d = int(sqrt(s2))
	assert(d*d == s2)
	for i in range(s1):
		temp = 0.
		for j in range(d):
			k = d*j + j
			temp = temp + m[i, k]
		h[i] = temp
	return h

def updateCrightOpenh1h2(hold, obj, mpsj):
	Hnew = contract(mpsj, hold, ((2,),(0,)))
	if obj is not None:
		Hnew = contract(Hnew, obj, ((1,),(1,)))
	return trace2(Hnew)

def updateCleftOpenh1h2(hold, obj, mpsj):
	Hnew = contract(hold, mpsj, ((0,), (0,)))
	if obj is not None:
		Hnew = contract(obj, Hnew, ((1,), (0,)))
	return trace2(Hnew.T)

