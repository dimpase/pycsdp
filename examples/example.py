#!/usr/bin/env python
from __future__ import division
import numpy as np
from pycsdp import _csdp
from sys import getrefcount
c1 = np.array([[2.0, 1.0],
               [1.0, 2.0]])
c2 = np.array([[3.0, 0.0, 1.0],
               [0.0, 2.0, 0.0],
               [1.0, 0.0, 3.0]])
c3 = np.array([[0.0, 0.0]])
# ct = np.random.random((800,800))
# c4 = -1/2.*(ct.T + ct)

C = _csdp.Objective([c1,c2,c3])
# print C.data()

def csdpindices(x):
    if x is None:
        return None
    # First check if x is square and get dimension of the block
    (n,n1) = x.shape
    assert n==n1, "x needs to be a square 2d-array"
    # Now copy upper triangular part
    xt = np.triu(x)
    # now get all the nonzero indices
    i,j = np.nonzero(xt)
    # now get the corresponding values
    val = xt[(i,j)]
    # Indexing starts at one
    return (n,i+1,j+1,val)

a11 = np.array([[3.0, 1.0],
               [1.0, 3.0]])
a12 = None
a13 = np.array([[1.0, 0.0],
                [0.0, 0.0]])
A1 = (csdpindices(a11),csdpindices(a12),csdpindices(a13))
a21 = None
a22 = np.array([[3.0, 0.0, 1.0],
                [0.0, 4.0, 0.0],
                [1.0, 0.0, 5.0]])
a23 = np.array([[0.0, 0.0],
                [0.0, 1.0]])
A2 = (csdpindices(a21),csdpindices(a22),csdpindices(a23))

A = _csdp.Constraints([A1,A2])
# b = np.array([1.0, 2.0])
b = [1.0, 2.0]
sol = _csdp.Solver(C,A,b)
(status,X,Z,obj,y) = sol.solution()
(X1,X2,X3) = X.data()
(Z1,Z2,Z3) = Z.data()
print status
print X1
print X2
print X3
print Z1
print Z2
print Z3
print obj
print y
