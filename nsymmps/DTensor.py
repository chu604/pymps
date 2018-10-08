# -*- coding: utf-8 -*-
# @Author: 1000787
# @Date:   2017-07-18 20:23:35
# @Last Modified by:   1000787
# @Last Modified time: 2018-03-07 17:24:48
from numpy import result_type, kron, diag, ndarray
from numpy import tensordot as contract
from numpy import ndarray as DTensor
from scipy.linalg import expm, svd, qr
from numpy import zeros, ones

def prodTuple(l):
	# assert(len(l) > 0)
	s = 1
	for i in l:
		s *= i
	return s

def moveSelectedIndexForward(a, I):
	na = len(a)
	nI = len(I)
	b = [None]*na
	k1 = 0
	k2 = nI
	for i in range(na):
		s = 0
		while s != nI:
			if i == I[s]:
				b[s] = a[k1]
				k1 += 1
				break
			s += 1
		if s == nI:
			b[k2] = a[k1]
			k1 += 1
			k2 += 1
	return type(a)(b)

def moveSelectedIndexBackward(a, I):
	na = len(a)
	nI = len(I)
	nr = na - nI
	b = [None]*na
	k1 = 0
	k2 = 0
	for i in range(na):
		s = 0
		while s != nI:
			if i == I[s]:
				b[nr + s] = a[k1]
				k1 += 1
				break
			s += 1
		if s == nI:
			b[k2] = a[k1]
			k2 += 1
			k1 += 1
	return type(a)(b)

# the desired svd
def svd2(A):
	"""
	stable singular value decomposition.
	gesdd is faster but sometimes do not converge, 
	gesvd always converges in my experience.
	therefore, try using gesdd first, if not converge,
	then use gesvd instead.
	"""
	try:
		return svd(A, full_matrices=False, lapack_driver='gesdd')
	except LinAlgError:
		return svd(A, full_matrices=False, lapack_driver='gesvd')

def qr2(dir, a):
	"""
	QR decomposition
	"""
	assert(dir == 'L' or dir == 'R')
	if dir == 'L':
		return qr(a, mode='economic', pivoting=False)
	else:
		q, r = qr(a.T, mode='economic', pivoting=False)
		return r.T, q.T

def measure_entropy(v):
	"""
	v is a one dimensional tensor with non-negative elements, 
	return the Von Neumann entropy
	"""
	assert(isinstance(v, ndarray))
	assert(v.ndim==1)
	a = v * v
	s = a.sum()
	a /= s
	return -inner(a, log(a))

# clear the column of m that are all zeros and return the index
def getRidOfZeroCol(m, tol=1.0e-12, verbose=False):
	assert(m.ndim==2)
	s1 = m.shape[0]
	s2 = m.shape[1]
	zerocols = []
	for j in range(s2):
		allzero = True
		for i in range(s1):
			if (abs(m[i,j]) > tol):
				allzero=False
				break
		if (allzero == True):
			if verbose:
				print('all elements of column ', j, ' are zero.')
			zerocols.append(j)
	ns = s2 - len(zerocols)
	if (ns == 0 and verbose == True):
		print('all the columns are zero.')
	mout = zeros((s1, ns), dtype=m.dtype)
	j = 0
	for i in range(s2):
		if i not in zerocols:
			mout[:, j] = m[:, i]
			j += 1
	return mout, zerocols

def matrixDeparallelisationNoZeroCols(m, tol=1.0e-12, verbose=False):
	s1 = m.shape[0]
	s2 = m.shape[1]
	K = []
	T = zeros((s2, s2), dtype=m.dtype)
	for j in range(s2):
		exist = False
		for i in range(len(K)):
			p, factor = isTwoColumnParallel(K[i], m[:, j], tol)
			if p==True:
				if verbose:
					print('column ', i, 'is in parallel with column ', j)
				T[i, j] = factor
				exist = True
				break
		if not exist:
			K.append(m[:, j])
			nK = len(K)	
			T[nK-1, j] = 1
	nK = len(K)
	M = zeros((s1, nK), dtype=m.dtype)
	for j in range(nK):
		M[:, j] = K[j]
	return M, T[:nK, :]

def matrixDeparallelisation(m, tol=1.0e-12, verbose=False):
	assert(m.ndim==2)
	mnew, zerocols = getRidOfZeroCol(m, tol, verbose)
	M, T = matrixDeparallelisationNoZeroCols(mnew, tol, verbose)
	if (M.size == 0):
		if verbose == True:
			print('all the elements of the matrix M are 0.')
		return M, T
	Tnew = zeros((T.shape[0], m.shape[1]), dtype=T.dtype)
	j = 0
	for i in range(Tnew.shape[1]):
		if i not in zerocols:
			Tnew[:, i] = T[:, j]
			j += 1
	# assert(allclose(dot(M, Tnew), m))
	return M, Tnew

def svdTruncate(U, S, V, maxbonddimension, svdcutoff, verbose):
	"""
	truncate the u, s, v resulting from svd. 
	try to truncate with the threshold first,
	if the result dimension less than D, return.
	Otherwise, force truncating to D
	"""
	assert(isinstance(U, ndarray))
	assert(isinstance(S, ndarray))
	assert(isinstance(V, ndarray))
	assert(maxbonddimension > 0)
	sizem = S.size 
	dim = sizem
	assert(S.ndim == 1)
	assert(U.shape[U.ndim-1] == S.shape[0] and S.shape[0] == V.shape[0])
	for i in range(sizem):
		if S[i] < svdcutoff:
			dim = i
			break;
	if (dim == sizem and sizem <= maxbonddimension):
		if (verbose >= 2):
			print('sum:', sizem, '->', dim)
		return (U, S, V, (dim, 0.))
	if (dim > maxbonddimension):
		if (verbose > 0):
			print('sum:', sizem, '->', dim, '(exceed the max bonddimension', maxbonddimension, \
				', cut off error is:', S[maxbonddimension], ')')
		dim = maxbonddimension
	s = 0.
	for i in range(dim, sizem):
		s = s + S[i]
	if verbose >= 2:
		print('sum:', sizem, '->', dim)
	return (U[..., 0:dim], S[0:dim], V[0:dim, :], (dim, s))

def svdDecompose(a, axes, k=200, useIterSolver=1, v0=None):
	"""
	the tensor index specified by axes will be moved to the end
	a new tensor is obtained by transposing the original one
	according to this new index sequence
	the output u, s, v will arrange the index according to
	the new tensor
	"""
	assert(a.size > 0)
	n = a.ndim
	nI = len(axes)
	dim = [i for i in range(n)]
	dim = moveSelectedIndexBackward(dim, axes)
	b = a.transpose(dim)
	s1 = s2 = 1
	# print(b.shape)
	ushape = [b.shape[i] for i in range(n-nI)]
	vshape = [b.shape[i] for i in range(n-nI, n)]
	for i in range(n-nI):
		s1 *= b.shape[i]
	for i in range(n-nI, n):
		s2 *= b.shape[i]
	if (useIterSolver == 1 and (k > 0) and (min(s1, s2) > k*5) ):
		u, s, v = svd2s(b.reshape((s1, s2)), k=k, v0=v0)
	else:
		u, s, v = svd2(b.reshape((s1, s2)))
	md = len(s)
	ushape = ushape + [md]
	vshape = [md] + vshape
	u = u.reshape(ushape)
	v = v.reshape(vshape)
	return u, s, v

def svdCompress(a, axes, maxbonddimension=-1, svdcutoff=1.0e-10, \
 useIterSolver=1, v0=None, verbose=False):
	assert(useIterSolver==1 or useIterSolver == 0)
	bonderror = (0, 0.)
	if maxbonddimension > 0:
		u, s, v = svdDecompose(a, axes, maxbonddimension+1, useIterSolver, v0)
		u, s, v, bonderror = svdTruncate(u, s, v, maxbonddimension, svdcutoff, verbose)
	else:
		u, s, v = svdDecompose(a, axes, maxbonddimension, 0, v0)
	return u, s, v, bonderror

def qrCompress(dir, a, axes):
	"""
	QR decomposition
	"""
	assert(dir == 'L' or dir == 'R')
	s1 = 1
	s2 = 1
	N1 = len(axes)
	newindex = [i for i in range(a.ndim)]
	dimu=[0]*(a.ndim-N1+1)
	dimv=[0]*(N1+1)
	newindex = moveSelectedIndexBackward(newindex, axes)
	a1 = a.transpose(newindex)
	for i in range(a.ndim-N1):
		dimu[i] = a1.shape[i]
		s1 *= a1.shape[i]
	for i in range(a.ndim-N1, a.ndim):
		dimv[i-a.ndim+N1+1] = a1.shape[i]
		s2 *= a1.shape[i]
	u, v = qr2(dir, a1.reshape(s1, s2))
	s = v.shape[0]
	dimu[-1] = s
	dimv[0] = s
	return u.reshape(dimu), v.reshape(dimv) 

def deparallelisationCompress(a, axes, tol=1.0e-12, verbose=False):
	n = a.ndim
	nI = len(axes)
	dim = [i for i in range(n)]
	dim = moveSelectedIndexBackward(dim, axes)
	b = a.transpose(dim)
	s1 = s2 = 1
	# print(b.shape)
	ushape = [b.shape[i] for i in range(n-nI)]
	vshape = [b.shape[i] for i in range(n-nI, n)]
	for i in range(n-nI):
		s1 *= b.shape[i]
	for i in range(n-nI, n):
		s2 *= b.shape[i]
	b = b.reshape((s1, s2))
	u, v = matrixDeparallelisation(b, tol, verbose)
	assert(u.shape[1] == v.shape[0])
	md = u.shape[1]
	ushape = ushape + [md]
	vshape = [md] + vshape
	u = u.reshape(ushape)
	v = v.reshape(vshape)
	return u, v

def directSum(a, b, axes):
	if a is None:
		return b.copy()
	if b is None:
		return a.copy()
	# assert(a.dtype==b.dtype)
	assert(a.ndim == b.ndim)
	dimc = [None]*(a.ndim)
	dim = [0]*(a.ndim)
	for i in range(a.ndim):
		if i in axes:
			dim[i] = a.shape[i]
			dimc[i] = a.shape[i] + b.shape[i]
		else:
			dimc[i] = a.shape[i]
	c = zeros(dimc, dtype = result_type(a.dtype, b.dtype))
	r = [slice(0, a.shape[i]) for i in range(a.ndim)]
	c[r] = a
	for i in range(a.ndim):
		r[i] = slice(dim[i], dimc[i])
	c[r] = b
	return c

def fusion(a, b, axes):
	assert(len(axes)==2)
	assert(len(axes[0]) == len(axes[1]))
	nI = len(axes[0])
	assert(a is not None)
	a1 = None
	b1 = None
	ranka = a.ndim
	assert(nI <= ranka)
	indexa = [i for i in range(ranka)]
	indexa = moveSelectedIndexBackward(indexa, axes[0])
	a1 = a.transpose(indexa)
	# sizem = 1
	# for i in range(ranka-nI, ranka):
	# 	sizem *= a1.shape[i]
	sizem = prodTuple(a1.shape[(ranka-nI):])
	a1 = a1.reshape(a1.shape[:(ranka-nI)] + (sizem,))
	if b is not None:
		rankb = b.ndim
		assert(nI <= rankb)
		indexb = [i for i in range(rankb)]
		indexb = moveSelectedIndexForward(indexb, axes[1])
		b1 = b.transpose(indexb)
		# sizem = 1
		# for i in range(nI):
		# 	sizem *= b1.shape[i]
		sizem = prodTuple(b1.shape[:nI])
		b1 = b1.reshape((sizem,)+b1.shape[nI:])
	return a1, b1
