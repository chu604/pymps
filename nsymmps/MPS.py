from .DTensor import DTensor, contract, directSum, diag, svdCompress, qrCompress
from .update import updateCrighth1h2, updateClefth1h2
from numpy import ones, sqrt, real, result_type, array
from numpy.random import rand
from scipy.linalg import norm
from .reduceD import reduceDSingleSite

class MPS(list):
	"""docstring for MPS"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def __mul__(self, num):
		r = self.copy()
		r *= num
		return r

	def __imul__(self, num):
		if (result_type(self[0].dtype, num)==self[0].dtype):
			self[0] *= num
		else:
			self[0] = self[0]*num
		return self

	def __rmul__(self, num):
		return self.__mul__(num)

	def __truediv__(self, num):
		r = self.copy()
		r /= num
		return r

	def __itruediv__(self, num):
		if (result_type(self[0].dtype, num)==self[0].dtype):
			self[0] /= num
		else:
			self[0] = self[0]/num
		return self

	def __add__(self, other):
		assert(isinstance(other, MPS))
		assert(self.__len__() == other.__len__())
		if self.trivial():
			return other.copy()
		if other.trivial():
			return self.copy()
		L = self.__len__()
		assert(L >= 1)
		axes = (0,2)
		res = [None]*L
		res[0] = directSum(self[0], other[0], (2,))
		res[L-1] = directSum(self[L-1], other[L-1], (0,))
		for i in range(1, L-1):
			res[i] = directSum(self[i], other[i], axes)
		return MPS(res)

	def __sub__(self, other):
		assert(isinstance(other, MPS))
		assert(self.__len__() == other.__len__())
		if self.trivial():
			return -other
		if other.trivial():
			return self.copy()
		L = self.__len__()
		assert(L >= 1)
		axes = (0,2)
		res = [None]*L
		res[0] = directSum(self[0], -other[0], (2,))
		res[L-1] = directSum(self[L-1], other[L-1], (0,))
		for i in range(1, L-1):
			res[i] = directSum(self[i], other[i], axes)
		return MPS(res)

	def __iadd__(self, other):
		self = self.__add__(other)
		return self

	def __isub__(self, other):
		self = self.__sub__(other)
		return self

	def __neg__(self):
		res = self.copy()
		res[0] *= -1.
		return res

	def conj(self):
		return MPS([s.conj() for s in self])
	# end of please implement the following functions

	def copy(self):
		r = MPS([s.copy() for s in self])
		# r.svectors = [s.copy() for s in self.svectors]
		return r

	def trivial(self):
		if not self:
			return True
		for s in self:
			if (s.size==0):
				return True
		return False

	def norm(self, isrightprepared=True):
		if isrightprepared==True:
			return norm(self[0])
		else:
			s = self.cross(self)
			assert(abs(s.imag) < 1.0e-12 )
			s = s.real
			if (abs(s) < 1.0e-14):
				return 0.
			return sqrt(s)

	def cross(self, other):
		assert(isinstance(other, MPS))
		assert(self.__len__() == other.__len__())
		if self.trivial() or other.trivial():
			return 0.
		L = self.__len__()
		hold = None
		for i in range(L-1, -1, -1):
			hold = updateCrighth1h2(hold, None, self[i].conj(), other[i])
		if hold:
			return hold.trace()
		return 0.

	def initializeEdgeSvectors(self):
		L = self.__len__()
		self.svectors = [None]*(L+1)
		self.svectors[0] = ones(1)
		self.svectors[L] = ones(1)

	def qrprepareLeft(self, verbose=0):
		if self.trivial():
			return
		L = self.__len__()
		for i in range(L-1):
			if verbose >= 2:
				print('qr prepare mps from left to right on site ', i)
			self[i], r = qrCompress('L', self[i], (2,))
			if (r.size==0):
				if verbose >= 1:
					print('mps becomes zero after cut off.')
				for s in self:
					s = None
				break
			self[i+1] = contract(r, self[i+1], ((1,),(0,)))

	def prepareLeft(self, maxbonddimension=-1, svdcutoff=1.0e-11, verbose=0):
		if self.trivial():
			return
		L = self.__len__()
		self.initializeEdgeSvectors()
		bond = 0
		error = 0.
		for i in range(L-1):
			if verbose >= 2:
				print('prepare mps from left to right on site ', i)
			self[i], s, U, bonderror = svdCompress(self[i], \
				(2,), maxbonddimension, svdcutoff, verbose=verbose)
			# in case s is empty after svd compress, it means the mps is zero.
			# so I set all of them to be None
			if (s.size==0):
				if verbose >= 1:
					print('mps becomes zero after cut off.')
				for s in self:
					s = None
				break
			U = contract(diag(s), U, ((1,),(0,)))
			self[i+1] = contract(U, self[i+1], ((1,),(0,)))
			self.svectors[i+1] = s
			bond = max(bond, bonderror[0])
			error = max(error, bonderror[1])
		return (bond, error)

	def prepareRight(self, maxbonddimension=-1, svdcutoff=1.0e-11, verbose=0):
		if self.trivial():
			return
		L = self.__len__()
		self.initializeEdgeSvectors()
		bond = 0
		error = 0.
		for i in range(L-1, 0, -1):
			if verbose >= 2:
				print('prepare mps from right to left on site ', i)
			U, s, self[i], bonderror=svdCompress(self[i], \
				(1,2), maxbonddimension, svdcutoff, verbose=verbose)
			if (s.size==0):
				if verbose >= 1:
					print('mps becomes zero after cut off.')
				for s in self:
					s = None
				break
			U = contract(U, diag(s), ((1,),(0,)))
			self[i-1] = contract(self[i-1], U, ((2,),(0,)))
			self.svectors[i] = s
			bond = max(bond, bonderror[0])
			error = max(error, bonderror[1])
		return (bond, error)

	def svdCompress(self, maxbonddimension=200, svdcutoff=1.0e-11, verbose=0):
		# bet1 = self.prepareLeft(-1, svdcutoff, verbose)
		self.qrprepareLeft(verbose)
		# bet2 = self.prepareRight(maxbonddimension, svdcutoff, verbose)
		# return (max(bet1[0], bet2[0]), max(bet1[1], bet2[1]))
		return self.prepareRight(maxbonddimension, svdcutoff, verbose)

	def __str__(self):
		L = self.__len__()
		ss = str()
		for i in range(L):
			ss += 'mps on site: ' +  str(i) + '\n'
			ss += self[i].__str__()
			ss += '\n'
		return ss

def createrandommps(L, d, D):
	mps = [None]*L
	assert(L > 1)
	mps[0] = rand(1, d, D)/sqrt(D)
	mps[L-1] = rand(D, d, 1)/sqrt(D)
	for i in range(1, L-1):
		mps[i] = rand(D, d, D)/sqrt(D)
	return MPS(mps)

def initCstorageRight(mpsA, mpsB):
	"""
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

def reduceOneSite(mpsA, mpsB, kmax=50, tol=1.0e-10, verbose=1):
	"""
	approximate mpsA by mpsB
	"""
	assert(len(mpsA) == len(mpsB))
	L = len(mpsB)
	Cstorage = initCstorageRight(mpsA, mpsB)
	itr = 0
	while (itr < kmax):
		if verbose >= 2:
			print('we are at', itr, '-th iteraction.')
		kvals = []
		for j in range(L-1):
			sitempsB = reduceDSingleSite(mpsA[j], Cstorage[j], Cstorage[j+1])
			kvals.append(norm(sitempsB))
			mpsB[j], s, u, bet = svdCompress(sitempsB, (2,), -1, tol, useIterSolver=0, verbose=verbose)
			Cstorage[j+1] = updateClefth1h2(Cstorage[j], None, mpsA[j], mpsB[j].conj())
		for j in range(L-1, 0, -1):
			sitempsB = reduceDSingleSite(mpsA[j], Cstorage[j], Cstorage[j+1])
			kvals.append(norm(sitempsB))
			u, mpsB.svectors[j], mpsB[j], bet = svdCompress(sitempsB, (1, 2), -1, tol, useIterSolver=0, verbose=verbose)
			Cstorage[j] = updateCrighth1h2(Cstorage[j+1], None, mpsA[j], mpsB[j].conj())
		kvals = array(kvals)
		err = abs(kvals.std() / kvals.mean())
		if verbose >= 2:
			# print(kvals)
			print('the error is', err)
		if err < tol:
			if verbose >= 2:
				print('converges after', itr, 'sweeps.')
			break
		itr += 1
	u = contract(u, diag(mpsB.svectors[1]), ((1,), (0,)))
	mpsB[0] = contract(mpsB[0], u, ((2,), (0,)))
	return mpsB
