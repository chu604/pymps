# -*- coding: utf-8 -*-
# @Author: 1000787
# @Date:   2017-06-03 16:19:51
# @Last Modified by:   1000787
# @Last Modified time: 2018-03-07 17:03:28
from .DTensor import DTensor, contract, diag
from .update import updateClefth1h2, updateCrighth1h2
from .update import updateCleftOpenh1h2, updateCrightOpenh1h2 
from .update import updateCright, updateCleft
from numpy import zeros, ones

def expectation(mpo, mpsA, mpsB):
	assert(len(mpo) == len(mpsA))
	assert(len(mpo) == len(mpsB))
	L = len(mpo)
	assert(L > 1)
	h = ones((1,1,1))
	for i in range(L, 0, -1):
		h = updateCright(h, mpo[i-1], mpsA[i-1], mpsB[i-1])
	assert(h.size==1)
	return h[0,0,0]

def overlap(mpsA, mpsB):
	assert(len(mpsA) == len(mpsB))
	L = len(mpsA)
	hold = None
	for j in range(L, 0, -1):
		hold = updateCrighth1h2(hold, None, mpsA[j-1], mpsB[j-1])
	assert(hold.size == 1)
	return hold[0][0]

def initCstorageOpenLeft(mps):
	L = len(mps)
	assert(L > 0)
	Cstorage = [None]*(L+1)
	Cstorage[0] = ones(1)
	for i in range(L):
		Cstorage[i+1] = updateCleftOpenh1h2(Cstorage[i], None, mps[i])
	return Cstorage

def initCstorageOpenRight(mps):
	L = len(mps)
	assert(L > 0)
	Cstorage = [None]*(L+1)
	Cstorage[L] = ones(1)
	for i in range(L-1, -1, -1):
		Cstorage[i] = updateCrightOpenh1h2(Cstorage[i+1], None, mps[i])
	return Cstorage

def openNorm(mps):
	Cstorage = initCstorageOpenLeft(mps)
	assert(Cstorage[-1].size==1)
	return Cstorage[-1][0]

# def getOpenNormFromCstorageRight(Cstorage):
# 	assert(len(Cstorage[0]) == 1)
# 	v0 = Cstorage[0].getOneKeyValuePair[1]
# 	assert(v0.size == 1)
# 	return v0[0]

# def getOpenNormFromCstorageLeft(Cstorage):
# 	assert(len(Cstorage[-1]) == 1)
# 	v0 = Cstorage[-1].getOneKeyValuePair()[1]
# 	assert(v0.size == 1)
# 	return v0[0]

class localObserver(dict):
	"""docstring for localObserver"""
	def __init__(self, site, *args, **kwargs):
		super(localObserver, self).__init__(*args, **kwargs)
		self.site = site
		self.results = {}

	def measure(self, mps, normOrCstorage=None, isunitary=True):
		if isunitary:
			self.measureUnitary(mps, normOrCstorage)
		else:
			self.measureOpen(mps, normOrCstorage)

	def measureUnitary(self, mps, mpsnorm=None):
		if mpsnorm is None:
			mpsnorm = mps.norm(True)
		hold = None
		mpsj = contract(diag(mps.svectors[self.site]), mps[self.site], ((1,),(0,)))
		for key, value in self.items():
			if key not in self.results:
				self.results[key] = []
			Hnew = updateCrighth1h2(hold, value, mpsj, mpsj.conj())
			temp = Hnew.trace()/mpsnorm
			self.results[key].append(temp)

	def measureOpen(self, mps, Cstorage=None):
		if Cstorage is None:
			Cstorage = initCstorageOpenLeft(mps)
		assert(Cstorage[-1].size==1)
		mpsnorm = Cstorage[-1][0]
		# print('mps norm: ', mpsnorm)
		L = len(mps)
		for key, value in self.items():
			if key not in self.results:
				self.results[key] = []
			Hnew = updateCleftOpenh1h2(Cstorage[self.site], value, mps[self.site])
			for j in range(self.site+1, L):
				Hnew = updateCleftOpenh1h2(Hnew, None, mps[j])
			assert(Hnew.size==1)
			temp = Hnew[0]/mpsnorm
			self.results[key].append(temp)

	def nMeasurements(self):
		return len(next(iter(self.results.items()))[1])

	def saveH5(self, h5group):
		for key, values in self.results.items():
			name = key + "_" + str(self.site)
			h5group.create_dataset(name=name, data = values)

	# how to use the save function from the parent class, answer: use super
	def __str__(self):
		ss = 'local observer on site: ' + str(self.site)
		ss += super().__str__()
		return ss

class uniformLocalObservers(list):
	"""docstring for uniformLocalObservers"""
	def __init__(self, start, stop):
		super().__init__([localObserver(site) for site in range(start, stop)])
		assert(start < stop)
		self.start = start
		self.stop = stop

	def size(self):
		return self.stop - self.start

	def set(self, name, op):
		assert(isinstance(name, str))
		if (isinstance(op, DTensor)):
			for j in range(self.size()):
				self[j][name] = op
		else:
			assert(isinstance(op, (tuple, list)))
			assert(len(op) == self.size())
			for j in range(self.size()):
				assert(isinstance(op[j], DTensor))
				self[j][name] = op[j]

	def measure(self, mps, normOrCstorage=None, isunitary=True):
		assert(self.stop <= len(mps))
		if (normOrCstorage is None):
			if isunitary==True:
				normOrCstorage = mps.norm(True)
			else:
				normOrCstorage = initCstorageOpenLeft(mps)
		for j in range(self.__len__()):
			self[j].measure(mps, normOrCstorage, isunitary)	

	def nMeasurements(self):
		assert(self.__len__() > 0)
		return self[0].nMeasurements()

	def getResults(self):
		L = self.__len__()
		assert(L > 0)
		totalstep = self.nMeasurements()
		if totalstep == 0:
			return None
		dtype = type(next(iter(self[0].results.items()))[1][-1])
		results = {}
		for key, value in self[0].items():
			results[key] = zeros((totalstep, L), dtype=dtype)
		for i in range(L):
			for key, value in self[i].results.items():
				for j in range(totalstep):
					results[key][j, i] = value[j]
		return results

	def saveH5(self, h5group):
		results = self.getResults()
		for key, value in results.items():
			h5group.create_dataset(name=key, data=value)

	def clearResults(self):
		for j in range(self.__len__()):
			del self[j].results[:]

class correlation:
	"""docstring for correlation"""
	def __init__(self, start, stop, distance, name, isfermionic=False):
		self.start = start
		self.stop = stop
		self.distance = distance
		self.name = name
		self.isfermionic = isfermionic
		L = self.size()
		self.op1 = [None]*L
		self.op2 = [None]*L
		# for fermion we need to add sz operator in between
		# here I trivially initialize it, but if it is fermionic, the 
		# middle operator should be correctly set when measure
		self.opm = [None]*L
		self.results = {}

	def size(self):
		return self.stop-self.start

	def measure(self, mps, normOrCstorage=None, isunitary=True):
		if isunitary:
			self.measureUnitary(mps, normOrCstorage)
		else:
			self.measureOpen(mps, normOrCstorage)

	def measureUnitary(self, mps, mpsnorm=None):
		assert(self.stop <= len(mps))
		if mpsnorm is None:
			mpsnorm = mps.norm(True)
		if self.isfermionic:
			for m in opm:
				assert(m is not None)
		for i in range(self.start, self.stop-1):
			mpsi = contract(diag(mps.svectors[i]), mps[i], ((1,),(0,)))
			Oeff = updateClefth1h2(None, self.op1[i-self.start], mpsi, mpsi.conj())
			if i not in self.results:
				self.results[i] = {}
			for j in range(i+1, min(self.stop, i+self.distance)):
				if j not in self.results[i]:
					self.results[i][j] = []
				Hnew = updateClefth1h2(Oeff, self.op2[j-self.start], mps[j], mps[j].conj())
				temp = Hnew.trace()/mpsnorm
				self.results[i][j].append(temp)
				Oeff = updateClefth1h2(Oeff, self.opm[j-self.start], mps[j], mps[j].conj())

	def measureOpen(self, mps, Cstorage=None):
		assert(self.stop <= len(mps))
		if Cstorage is None:
			Cstorage = initCstorageOpenLeft(mps)
		mpsnorm = Cstorage[-1][0]
		# print('mps norm is: ', mpsnorm)
		if self.isfermionic:
			for m in opm:
				assert(m is not None)
		L = len(mps)
		for i in range(self.start, self.stop-1):
			Oeff = updateCleftOpenh1h2(Cstorage[i], self.op1[i-self.start], mps[i])
			if i not in self.results:
				self.results[i] = {}
			for j in range(i+1, min(self.stop, i+self.distance)):
				if j not in self.results[i]:
					self.results[i][j] = []
				Oeff1 = updateCleftOpenh1h2(Oeff, self.op2[j-self.start], mps[j])
				for k in range(j+1, L):
					Oeff1 = updateCleftOpenh1h2(Oeff1, None, mps[k])
				assert(Oeff1.size == 1)
				temp = Oeff1[0]/mpsnorm
				self.results[i][j].append(temp)
				Oeff = updateCleftOpenh1h2(Oeff, self.opm[j-self.start], mps[j])

	def saveH5(self, h5group):
		for k1, v1 in self.results.items():
			n1 = self.name + "_" + str(k1)
			for k2, v2 in v1.items():
				n2 = n1 + "_" + str(k2)
				h5group.create_dataset(name=n2, data = v2)

	def __str__(self):
		ss = str()
		if self.isfermionic:
			ss += 'fermionic two body correlation: \n'
		else:
			ss += 'two body correlation: \n'
		ss += ('start: ' + str(self.start) + " stop: " \
		+ str(self.stop) + " distance: " + self.distance)
		L = self.size()
		for i in range(L):
			ss += ('first operator on site ' + str(i))
			ss += self.op1[i].__str__()
			ss += "\n"
			ss += ('second operator on site ' + str(i))
			ss += self.op2[i].__str__()
			if self.isfermionic:
				ss += ('sz operator on site' + str(i))
				ss + self.opm[i].__str__()
		return ss

class ObserverList(dict):
	"""docstring for ObserverList"""
	def __init__(self, *arg, **kwargs):
		super().__init__(*arg, **kwargs)
		self.results = {}

	def measure(self, mps, normOrCstorage=None, isunitary=True):
		if isunitary:
			self.measureUnitary(mps, normOrCstorage)
		else:
			self.measureOpen(mps, normOrCstorage)

	def measureOpen(self, mps, Cstorage=None):
		if Cstorage:
			Cstorage = initCstorageOpenLeft(mps)
		for key, value in self.items():
			if key not in self.results:
				self.results[key] = []
			self.results[key].append(self.__measureOpenSingle(value, mps, Cstorage))

	def measureUnitary(self, mps, mpsnorm=None):
		if mpsnorm is None:
			mpsnorm = mps.norm(True)
		for key, value in self.items():
			if key not in self.results:
				self.results[key] = []
			self.results[key].append(self.__measureUnitarySingle(value, mps, mpsnorm))

	def __measureUnitarySingle(self, ob, mps, mpsnorm):
		start = min(ob)
		stop = max(ob)
		assert(stop < len(mps))
		mpsj = contract(diag(mps.svectors[start]), mps[start], ((1,), (0,)))
		Hnew = updateClefth1h2(None, ob[start], mpsj, mpsj.conj())
		for i in range(start+1, stop+1):
			opi = ob.get(i)
			Hnew = updateClefth1h2(Hnew, ob.get(i), mps[i], mps[i].conj())
		return Hnew.trace()/mpsnorm

	def __measureOpenSingle(self, ob, mps, Cstorage):
		assert(len(Cstorage) == len(mps)+1)
		assert(Cstorage[-1].size == 1)
		mpsnorm = Cstorage[-1][0]
		start = min(ob)
		Hnew = updateCleftOpenh1h2(Cstorage[start], ob[start], mps[start])
		for i in range(start+1, len(mps)):
			Hnew = updateCleftOpenh1h2(Hnew, ob.get(i), mps[i])
		assert(Hnew.size==1)
		temp = Hnew[0]
		return temp/mpsnorm

	def saveH5(self, h5group):
		for key, value in self.results.items():
			assert(isinstance(key, str) and isinstance(value, list))
			h5group.create_dataset(name=key, data=value)

