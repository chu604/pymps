# -*- coding: utf-8 -*-
# @Author: 1000787
# @Date:   2017-06-03 16:22:43
# @Last Modified by:   1000787
# @Last Modified time: 2018-03-07 17:03:19
from .DTensor import DTensor, kron
from numpy import ones, zeros, sqrt, array, ndarray
from .MPO import MPO
from .MPS import MPS

def generateSpinOperator():
	sx = array([[0., 1.], [1., 0.]])
	sy = array([[0., -1j], [1j, 0.]])
	sz = array([[1., 0.], [0., -1.]])
	su = array([[1., 0.], [0., 0.]])
	sd = array([[0., 0.], [0., 1.]])
	sp = array([[0., 1.], [0., 0.]])
	sm = array([[0., 0.], [1., 0.]])
	id2 = array([[1., 0.], [0., 1.]])
	return {"sx":sx, "sy":sy, "sz":sz, "sp":sp, "sm":sm, \
	"su":su, "sd":sd, "id":id2}

def generateBosonOperator(d):
	a = zeros((d, d))
	for i in range(d - 1):
		a[i, i+1] = sqrt(i+1);
	adag = a.T.copy()
	n = adag.dot(a)
	n2 = n.dot(n)
	idd = eye(d)
	return {"a":a, "adag":adag, "n":n, "n2":n2, "id":idd}

class simple_uniform_lattice:
	"""docstring for openBosonLattice"""
	def __init__(self, L, iden):
		self.iden = iden
		assert(self.iden.ndim == 2)
		assert(self.iden.shape[0] == self.iden.shape[1])
		self.L = L
		self.d = self.iden.shape[0]

	def size(self):
		return self.L

	def generateProdMPO(self, mpostr):
		if max(mpostr) >= self.L:
			raise ValueError("index out of range.")
		mpo = [None]*self.L
		L = self.size()
		temp = mpostr.get(L-1)
		if temp is None:
			temp =  self.iden
		assert(temp.ndim==2)
		assert(temp.shape[0] == temp.shape[1])
		d = temp.shape[0]
		assert(d == self.d)
		mpo[L-1] = array(temp.reshape((d,1,1,d)))
		for i in range(self.size()-2, -1, -1):
			temp = mpostr.get(i)
			if temp is None:
				temp = self.iden
			assert(temp.ndim==2)
			assert(temp.shape[0] == temp.shape[1])
			d = temp.shape[0]
			assert(d == self.d)
			mpo[i] = array(temp.reshape((d,1,1,d)))
		return MPO(mpo)

	def generateProdMPS(self, mpsstr):
		assert(self.size() == len(mpsstr))
		mps = [None]*self.L
		for i in range(self.L-1, -1, -1):
			mps[i] = zeros((1,self.d,1))
			# print(type(mpsstr[i]))
			assert(isinstance(mpsstr[i], (int, list, ndarray)))
			if isinstance(mpsstr[i], int):
				mps[i][0, mpsstr[i], 0] = 1.
			else:
				mps[i][0, : ,0] = mpsstr[i]
		return MPS(mps)

	def generateEmptyMPO(self):
		return MPO([DTensor((0, 0, 0, 0)) for i in range(self.L)])


