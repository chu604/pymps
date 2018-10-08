# -*- coding: utf-8 -*-
# @Author: 1000787
# @Date:   2017-06-03 16:16:38
# @Last Modified by:   1000787
# @Last Modified time: 2018-03-07 17:23:53
from .DTensor import DTensor, contract, directSum, deparallelisationCompress, \
diag, fusion, svdCompress


class MPO(list):
	"""docstring for MPO"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def __add__(self, other):
		assert(isinstance(other, MPO))
		assert(self.__len__() == other.__len__())
		if self.trivial():
			return other.copy()
		if other.trivial():
			return self.copy()
		L = self.__len__()
		res = [None]*L
		res[0] = directSum(self[0], other[0], (2,))
		res[L-1] = directSum(self[L-1], other[L-1], (1,))
		for i in range(1, L-1):
			res[i] = directSum(self[i], other[i], (1, 2))
		return MPO(res)

	def __sub__(self, other):
		assert(isinstance(other, MPO))
		assert(self.__len__() == other.__len__())
		if self.trivial():
			return -other
		if other.trivial():
			return self.copy()
		L = self.__len__()
		res = [None]*L
		res[0] = directSum(self[0], -other[0], (2,))
		res[L-1] = directSum(self[L-1], other[L-1], (1,))
		for i in range(1, L-1):
			res[i] = directSum(self[i], other[i], (1, 2))
		return MPO(res)

	def __iadd__(self, other):
		self = self.__add__(other)
		return self

	def __isub__(self, other):
		self = self.__sub__(other)
		return self

	def __neg__(self):
		res = self.copy()
		res[0] *= (-1.)
		return res

	def __mul__(self, num):
		res = self.copy()
		res[0] *= num
		return res

	def __imul__(self, num):
		self[0] *= num
		return self

	def __rmul__(self, num):
		return self.__mul__(num)

	def __truediv__(self, num):
		res = self.copy()
		res[0] /= num
		return res

	def __itruediv__(self, num):
		self[0] /= num
		return self

	def conj(self):
		return MPO([s.conj() for s in self])

	def dot(self, other):
		if isinstance(other, MPO):
			return self.__dotMPO(other)
		else:
			return self.__dotMPS(other)

	# mpo times mpo, may be useful to also implement mpo times scalar
	def __dotMPO(self, other):
		assert(isinstance(other, MPO))
		assert(self.__len__() == other.__len__())
		if self.trivial():
			return self
		if other.trivial():
			return other
		L = self.__len__()
		assert(L > 0)
		res = [None]*L
		for i in range(L):
			res[i] = contract(self[i], other[i], ((3,),(0,)))
		res[0], temp = fusion(res[0], None, ((1,3),(0,0)))
		res[0] = res[0].transpose((0,4,1,2,3))
		for i in range(L-1):
			res[i], res[i+1] = fusion(res[i], res[i+1], ((2,3),(1,3)))
			res[i] = res[i].transpose((0,1,3,2))
			res[i+1] = res[i+1].transpose((1,0,2,3,4))
		res[L-1], temp = fusion(res[L-1], None, ((2,3),(0,0)))
		res[L-1] = res[L-1].transpose((0,1,3,2))
		return MPO(res)

	def __dotMPS(self, mps):
		assert(self.__len__() == mps.__len__())
		L = mps.__len__()
		res = [None]*L
		for i in range(L):
			res[i] = contract(self[i], mps[i], ((3,),(1,)))
		res[0], temp = fusion(res[0], None, ((1,3),(0,0)))
		res[0] = res[0].transpose((3,0,1,2))
		for i in range(L-1):
			res[i], res[i+1] = fusion(res[i], res[i+1], ((2,3),(1,3)))
		res[L-1], temp = fusion(res[L-1], None, ((2,3),(0,0)))
		return type(mps)(res)

	def check(self):
		for k in self:
			if (not isinstance(k, QTensor)):
				return False
		return True

	def copy(self):
		return MPO([s.copy() for s in self])

	def trivial(self):
		if not self:
			return True
		for s in self:
			if s.size==0:
				return True
		return False

	def deparallelisationLeft(self, tol=1.0e-12, verbose=False):
		for i in range(self.__len__()-1):
			if verbose:
				print('sweep from left to right on site: ', i)
			M, T = deparallelisationCompress(self[i], (2,), tol, verbose)
			if (M.size > 0):
				self[i] = M.transpose((0, 1, 3, 2))
				self[i+1] = contract(T, self[i+1], ((1,),(1,)))
				self[i+1] = self[i+1].transpose((1,0,2,3))
			else:
				if verbose:
					print('mpo becomes zero after deparallelisation left.')
				for s in self:
					s = DTensor((0, 0, 0, 0))
				break

	def deparallelisationRight(self, tol=1.0e-12, verbose=False):
		for i in range(self.__len__()-1, 0, -1):
			if verbose:
				print('sweep from right to left on site: ', i)
			M, T = deparallelisationCompress(self[i], (1,), tol, verbose)
			if (M.size > 0):
				self[i] = M.transpose((0,3,1,2))
				self[i-1] = contract(self[i-1], T, ((2,),(1,)))
				self[i-1] = self[i-1].transpose((0,1,3,2))
			else:
				if verbose:
					print('mpo becomes zero after deparallelisation right.')
				for s in self:
					s = DTensor((0, 0, 0, 0))
				break

	def compress(self, tol=1.0e-12, verbose=False):
		self.deparallelisationLeft(tol, verbose)
		self.deparallelisationRight(tol, verbose)

	def prepareLeft(self, maxbonddimension, svdcutoff, verbose):
		if self.trivial():
			return
		L = self.__len__()
		bond = 0
		error = 0.
		for i in range(L-1):
			if verbose >= 2:
				print('prepare mpo from left to right on site ', i)
			self[i], s, U, bonderror = svdCompress(self[i], 
				(2,), maxbonddimension, svdcutoff, verbose=verbose)
			if s.size==0:
				if verbose >= 1:
					print('mpo becomes zero after cut off.')
				for s in self:
					s = DTensor((0,0,0,0))
				break
			U = contract(diag(s), U, ((1,), (0,)))
			self[i] = self[i].transpose((0, 1, 3, 2))
			self[i+1] = contract(U, self[i+1], ((1,), (1,)))
			self[i+1] = self[i+1].transpose((1, 0, 2, 3))
			bond = max(bond, bonderror[0])
			error = max(error, bonderror[1])
		return bond, error

	def prepareRight(self, maxbonddimension, svdcutoff, verbose):
		if self.trivial():
			return
		L = self.__len__()
		bond = 0
		error = 0.
		for i in range(L-1, 0, -1):
			if verbose >= 2:
				print('prepare mpo from right to left on site ', i)
			U, s, self[i], bonderror=svdCompress(self[i], \
				(0,2,3), maxbonddimension, svdcutoff, verbose=verbose)
			if (s.size==0):
				if verbose >= 1:
					print('mpo becomes zero after cut off.')
				for s in self:
					s = DTensor((0,0,0,0))
				break
			U = contract(U, diag(s), ((1,),(0,)))
			self[i-1] = contract(self[i-1], U, ((2,),(0,)))
			self[i-1] = self[i-1].transpose((0,1,3,2))
			self[i] = self[i].transpose((1,0,2,3))
			bond = max(bond, bonderror[0])
			error = max(error, bonderror[1])
		return bond, error

	def svdCompress(self, maxbonddimension=200, svdcutoff=1.0e-10, verbose=0):
		self.prepareLeft(-1, svdcutoff, verbose)
		return self.prepareRight(maxbonddimension, svdcutoff, verbose)


	def __str__(self):
		L = self.__len__()
		ss = str()
		for i in range(L):
			ss += 'mpo on site: ' +  str(i) + '\n'
			ss += self[i].__str__()
			ss += '\n'
		return ss

def createEmptyMPO(L):
	return MPO([DTensor((0, 0, 0, 0)) for i in range(L)])
