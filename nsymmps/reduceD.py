# -*- coding: utf-8 -*-
# @Author: 1000787
# @Date:   2017-07-19 09:13:59
# @Last Modified by:   1000787
# @Last Modified time: 2018-03-07 17:04:43
from .DTensor import contract

def reduceDSingleSite(A, Cleft, Cright):
	B = contract(Cleft, A, ((1,), (0,)))
	return contract(B, Cright, ((2,), (1,)))

def reduceDSingleBond(A, Cleft, Cright):
	B = contract(Cleft, A, ((1,), (0,)))
	return contract(B, Cright, ((3,), (1,)))
