def _theta_graph(n, e): # this should work in the same way as theta in csdp
                        # number of edges equals len(e)
# graph is given by a list of edges (not checked in any way), from 0 to n-1
# e.g. 
# _theta_graph(5,[[0,1],[1,2],[2,3],[3,4],[0,4]]) should return sqrt(5.)==2.23606...
#
# return (status,X,Z,obj,y)

   import numpy as np
   from pycsdp import _csdp
   from sys import getrefcount # why do we need this?

   c0 = np.empty([n,n]) # objective
   for i in range(n):
      for j in range(n):
         c0[i][j] = 1.0
   C = _csdp.Objective([c0])
      
   A1 = (n,np.array(range(1,n+1)), np.array(range(1,n+1)), 
         np.array([1.0]*n)) # constraint Tr(X)==1
   cA = [(A1,None)]
   for i,j in e:
      if i<j:
         cA.append(((n,np.array([i+1]), np.array([j+1]), np.array([1.0])),None))
      else:
         cA.append(((n,np.array([j+1]), np.array([i+1]), np.array([1.0])),None))

   A = _csdp.Constraints(cA)
   b = [1.0]+[0.0]*len(e)
   sol = _csdp.Solver(C,A,b)
   return sol.solution()

def _theta_cayley_graph(invs, perms):
# Cayley graph of an n-element group is given by its generators, 
# involutory(invs)+noninvolutory(perms); each generator is given as a list
# of numbers from 1 to n 
# e.g. 1-obj(_theta_cayley_graph([], [[2,3,4,5,1]])) should give sqrt(5.0),
# the value of theta for the 5-gon
#
   import numpy as np
   from pycsdp import _csdp
   from sys import getrefcount # why do we need this?
   if invs == []:
      n = len(perms[0])
   else:
      n = len(invs[0])
   
   b = [-1.0*float(n)]*len(invs)+[-float(n)*2.0]*len(perms)
   c0 = -np.identity(n)/float(n)
   C = _csdp.Objective([c0]) # objective

   cA = []
   for p in invs: 
      p1=[]
      p2=[]
      for i in range(n):
         if i+1<p[i]:
            p1.append(i+1)   
            p2.append(p[i])
      cA.append((
          (n,np.array(p1), np.array(p2), np.array([1.0]*(n/2))), None))

   for p in perms: 
      p1=[]
      p2=[]
      for i in range(n):
         if i+1<p[i]:
            p1.append(i+1)   
            p2.append(p[i])
         else:
            p2.append(i+1)   
            p1.append(p[i])
      cA.append((
          (n,np.array(p1), np.array(p2), np.array([1.0]*n)), None))

   A = _csdp.Constraints(cA)
   sol = _csdp.Solver(C,A,b)
 
   return sol.solution()
