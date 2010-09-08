/*  Python wrapper to Coin-or CSDP library. 
    Copyright (C) 2010  Benjamin Kern <benjamin.kern@ovgu.de>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "_csdp.h" 

Csdp::Csdp() : Py::ExtensionModule<Csdp>("_csdp"){
  Solver::init_type();
  Constraints::init_type();
  Objective::init_type();
  add_varargs_method("Solver", &Csdp::new_solver, "Solver(C,A,b)");
  add_varargs_method("Constraints", &Csdp::new_constraints, "Constraints");
  add_varargs_method("Objective", &Csdp::new_objective, "Objective");
  initialize("Input documentation of Csdp module in here");
}

void Constraints::init_type() {
  behaviors().name("Constraints Object");
  behaviors().doc("Put Docu in here");
}

void Objective::init_type() {
  behaviors().name("Objective Object");
  behaviors().doc("Put Docu in here");
  add_noargs_method("data", &Objective::get_data, "data()");
}


void Solver::init_type() {
  behaviors().name("Solver Object");
  behaviors().doc("Put Docu in here");
  add_noargs_method("solution", &Solver::get_solution, "solution()");
}

Objective::Objective(const struct blockmatrix& X) : dimension(0) {
  size_t nblocks = static_cast<size_t>(X.nblocks);
  C.nblocks = nblocks;
  C.blocks = (struct blockrec*)malloc((nblocks + 1)*sizeof(struct blockrec));
  if (C.blocks == NULL)
    throw Py::MemoryError("Error allocating memory for blocks");
  // Indexing starts at 1. 0-th index is omitted
  for(size_t i = 1; i <= nblocks; i++) {
    size_t blocksize = static_cast<size_t>(X.blocks[i].blocksize);
    // Sum up the columns to get the overall dimension of the C Matrix.
    dimension = dimension + blocksize;
    C.blocks[i].blocksize = blocksize;
    if (X.blocks[i].blockcategory == DIAG) {
      C.blocks[i].blockcategory = DIAG;
      // If we allocate data for vectors, we need one extra entry!!
      C.blocks[i].data.vec = (double*)malloc((blocksize + 1)*sizeof(double));
      if (C.blocks[i].data.vec == NULL)
        throw Py::MemoryError("Error allocating memory for blocks");
      for (size_t j = 1; j <= blocksize; j++)
        C.blocks[i].data.vec[j] = X.blocks[i].data.vec[j];
    } else {
      C.blocks[i].blockcategory = MATRIX;
      C.blocks[i].data.mat =
        (double*)malloc(blocksize*blocksize*sizeof(double));
      if (C.blocks[i].data.mat == NULL)
        throw Py::MemoryError("Error allocating memory for blocks");
      for (size_t j = 1; j <= blocksize; j++) {
        for (size_t k = 1; k <= blocksize; k++) {
          C.blocks[i].data.mat[ijtok(j,k,blocksize)] = 
            X.blocks[i].data.mat[ijtok(j,k,blocksize)];
        }
      }
    }
  }
}

Objective::Objective(const Py::List &py_list) : dimension(0) {
  // Each element of py_list contains an array corresponding to the blocks
  const size_t nblocks = py_list.length();
  C.nblocks = nblocks;
  C.blocks = (struct blockrec*)malloc((nblocks + 1)*sizeof(struct blockrec));
  if (C.blocks == NULL)
    throw Py::MemoryError("Error allocating memory for blocks");
  PyArrayObject* py_C = NULL;
  // Indexing starts at 1. 0-th index is omitted
  for(size_t i = 1; i <= nblocks; i++) {
    // Every array we pass needs to have rank 2. Python indexing starts a 0.
    py_C = (PyArrayObject*)PyArray_ContiguousFromObject(py_list[i-1].ptr(),
                                                        PyArray_DOUBLE, 2, 2);
    if (py_C == NULL) {
      Py_XDECREF(py_C);
      throw Py::ValueError("C needs to be a 2D Array, with at least one row");
    }
    int rows = PyArray_DIM(py_C,0);
    // blocksize is equal to the number of columns
    int blocksize = PyArray_DIM(py_C, 1);
    // Sum up the columns to get the overall dimension of the C Matrix.
    dimension = dimension + blocksize;
    if (rows == 1) {
      // One row always implies a diagonal Block.
      C.blocks[i].blockcategory = DIAG;
      C.blocks[i].blocksize = blocksize;
      // If we allocate data for vectors, we need one extra entry!!
      C.blocks[i].data.vec = (double*)malloc((blocksize + 1)*sizeof(double));
      if (C.blocks[i].data.vec == NULL) {
        Py_XDECREF(py_C);
        // TODO Memory leaks when wrong input is provided?????
        throw Py::MemoryError("Error allocating memory for blocks");
      }
      // Python indexing starts a 0.
      for (int j = 1; j <= blocksize; j++)
        C.blocks[i].data.vec[j] = *(double*)PyArray_GETPTR2(py_C, 0, j-1);
    } else {
      if (rows != blocksize) {
        Py_XDECREF(py_C);
        // TODO Memory leaks when wrong input is provided?????
        throw Py::ValueError("Matrix blocks need to be square arrays");
      }
      C.blocks[i].blockcategory = MATRIX;
      C.blocks[i].blocksize = blocksize;
      C.blocks[i].data.mat =
        (double*)malloc(blocksize*blocksize*sizeof(double));
      if (C.blocks[i].data.mat == NULL) {
        // TODO Memory leaks when wrong input is provided?????
        Py_XDECREF(py_C);
        throw Py::MemoryError("Error allocating memory for blocks");
      }
      // Python indexing starts a 0.
      for (int j = 1; j <= blocksize; j++)
        for (int k = 1; k <= blocksize; k++)
          C.blocks[i].data.mat[ijtok(j,k,blocksize)] =
              *(double*)PyArray_GETPTR2(py_C, j-1, k-1);
    }
    Py_DECREF(py_C);
  }
}

Py::Object Solver::get_solution(void) {
  // status == 0. Success
  // status == 1. Success: Primal infeasible
  // status == 2. Success: Dual infeasible
  // status == 3. Partial Success: Solution found, but accuracy not achieved
  // status == 4. Failure: Maximum iterations reached
  // status == 5. Failure: Stuck at edge of primal feasibility
  // status == 6. Failure: Stuck at edge of dual feasibility
  // status == 7. Failure: Lack of progress
  // status == 8. Failure: X, Z or O was singular
  // status == 9. Failure: Detected NaN or Inf values
  Py::List py_list(5);
  py_list[0] = Py::Long(status);
  if (status == 0 || status == 3) {
    py_list[1] = Py::asObject(new Objective(X)); 
    py_list[2] = Py::asObject(new Objective(Z));
    py_list[3] = Py::Float(obj);
    PyArrayObject* x = NULL;
    npy_intp y_dims[] = {ncons};
    x = (PyArrayObject*)PyArray_SimpleNew(1, y_dims, PyArray_DOUBLE);
    if (x == NULL) {
      Py_XDECREF(x);
      throw Py::ValueError("Problem with creating the return object");
    }
    double* x_ptr = (double*)PyArray_DATA(x);
    // Indexing starts at one
    for(int i = 1; i <= y_dims[0]; i++)
      *x_ptr++ = *(y+i);
    py_list[4] = Py::asObject(PyArray_Return(x));
  } else {
    py_list[1] = Py::None(); 
    py_list[2] = Py::None();
    py_list[3] = Py::None();
    py_list[4] = Py::None();
  }
  return py_list;
}

Py::Object Objective::get_data(void) {
  size_t nblocks = C.nblocks;
  Py::List py_list(nblocks);
  PyArrayObject* x = NULL;
  for(size_t i = 1; i <= nblocks; i++) {
    // C.blocks[0] is a dummy block
    int blocksize = C.blocks[i].blocksize;
    if (C.blocks[i].blockcategory == DIAG) {
      npy_intp m_dims[2] = {1, blocksize};
      x = (PyArrayObject*)PyArray_SimpleNew(2, m_dims, PyArray_DOUBLE);
      if (x == NULL) {
        Py_XDECREF(x);
        throw Py::ValueError("Problem with creating the vector return object");
      }
      double* x_ptr = (double*)PyArray_DATA(x);
      // vec[0] is a dummy entry
      for (int j = 1; j <= blocksize; j++)
        *x_ptr++ = C.blocks[i].data.vec[j];
      py_list[i-1] = Py::asObject(PyArray_Return(x));
    } else {
      npy_intp m_dims[2] = {blocksize, blocksize};
      x = (PyArrayObject*)PyArray_SimpleNew(2, m_dims, PyArray_DOUBLE);
      if (x == NULL) {
        Py_XDECREF(x);
        throw Py::ValueError("Problem with creating the matrix return object");
      }
      double* x_ptr = (double*)PyArray_DATA(x);
      // mat[0][0] is a dummy entry
      for (int j=1; j<=blocksize; j++)
        for (int k=1; k<=blocksize; k++)
          *x_ptr++ = C.blocks[i].data.mat[ijtok(j,k,blocksize)];
      py_list[i-1] = Py::asObject(PyArray_Return(x));
    }
  }
  return py_list;
}

Constraints::Constraints(const Py::List &py_list) : ncons(py_list.length()) {
  /* Constructor comments:
   * As input only a list of tuples is accepted. The number of
   * constraints is specified by the number of entries. Each tuple defines the
   * structure of a particular block. This structure is again described by
   * either a tuple or by Py_None. If the entry of the inner tuple is Py_None
   * then a zero block is assumed. Otherwise the innertuple needs to have the
   * following (n,j,i,v), where
   * n specifies the blocksize (Integer)
   * j spefifies the j-th nonzero uppertriangual entry of the block (PyArray)
   * i spefifies the i-th nonzero uppertriangual entry of the block (PyArray)
   * v speficies the corresponding value of this block. (PyArray)
   * Indexing for i and j starts at 1.
   * Example: An example for setting up 2 Constraint Matrices that have 
   * 3 Blocks might look like
   * A = [((n,j,i,v),(n2,j2,i2,v2), Py_None),(Py_None,(n3,j3,i3,v3),Py_None)]
   * Keep also in mind that the blocksizes need to be consistent, meaning that
   * in this case (n3==n2) needs to be true.
   */
  A = NULL;
  A = (constraintmatrix*)malloc((ncons + 1)*sizeof(constraintmatrix));
  if (A == NULL) {
    throw Py::MemoryError("Error allocating memory for blocks");
  }
  // Helper to setup the linked list
  struct sparseblock* blockptr=NULL;
  // Setup the blocks for each constraint
  for (size_t i = 1; i <= ncons; i++) {
    // Array is a list of tuples. Python index starts at 0.
    Py::Tuple temp = py_list[i-1];
    size_t nblocks = temp.length();
    // Terminate the linked list with a NULL pointer. A[0].blocks is a dummy
    // block
    A[i].blocks = NULL;
    // We will setup the blocks in reversed order (like in example.c)
    for (size_t j = nblocks; j >= 1; j--) {
      // If the element of the tuple is none, we have a zero block.
      if (temp[j-1].ptr()!=Py_None) {
        Py::Tuple py_aij(temp[j-1]);
        py_aij.verify_length(4);
        Py::Long py_blocksize(py_aij[0]);
        PyArrayObject* py_i=NULL;
        PyArrayObject* py_j=NULL;
        PyArrayObject* py_a=NULL;
        py_i = (PyArrayObject*)PyArray_ContiguousFromObject(py_aij[1].ptr(),
                                                            PyArray_LONG, 1, 1);
        if (py_i == NULL) {
        // TODO Memory leaks when wrong input is provided?????
          Py_XDECREF(py_i);
          throw Py::ValueError("Provide compatible 1D arrays");
        }
        int nnz = PyArray_DIM(py_i, 0);
        py_j = (PyArrayObject*)PyArray_ContiguousFromObject(py_aij[2].ptr(),
                                                            PyArray_LONG, 1, 1);
        if (py_j == NULL || PyArray_DIM(py_j,0) != nnz) {
        // TODO Memory leaks when wrong input is provided?????
          Py_XDECREF(py_i);
          Py_XDECREF(py_j);
          throw Py::ValueError("Provide compatible 1D arrays");
        }
        py_a = (PyArrayObject*)PyArray_ContiguousFromObject(py_aij[3].ptr(), 
                                                        PyArray_DOUBLE, 1, 1);
        if (py_a == NULL || PyArray_DIM(py_a,0) != nnz) {
        // TODO Memory leaks when wrong input is provided?????
          Py_XDECREF(py_i);
          Py_XDECREF(py_j);
          Py_XDECREF(py_a);
          throw Py::ValueError("Provide compatible 1D arrays");
        }
        // So now we handle the j-th block in the list
        blockptr=(struct sparseblock*)malloc(sizeof(struct sparseblock));
        if (blockptr == NULL) {
        // TODO Memory leaks when wrong input is provided?????
          throw Py::MemoryError("Error allocating memory for blocks");
        }
        // Setup the j-th block of the i-th A constraint Matrix
        blockptr->blocknum = j;
        blockptr->blocksize = long(py_blocksize);
        // The constraint enumerating starts from 1.
        blockptr->constraintnum = i;
        blockptr->next = NULL;
        blockptr->nextbyblock = NULL;
        // Indexing starts from 1, so nnz+1 (0-th entry is not needed)
        blockptr->entries = (double*)malloc((nnz + 1)*sizeof(double));
        if (blockptr->entries == NULL) {
        // TODO Memory leaks when wrong input is provided?????
          throw Py::MemoryError("Error allocating memory for blocks");
        }
        blockptr->iindices = (int*)malloc((nnz + 1)*sizeof(int));
        if (blockptr->iindices == NULL) {
        // TODO Memory leaks when wrong input is provided?????
          throw Py::MemoryError("Error allocating memory for blocks");
        }
        blockptr->jindices = (int*)malloc((nnz + 1)*sizeof(int));
        if (blockptr->jindices == NULL) {
        // TODO Memory leaks when wrong input is provided?????
          throw Py::MemoryError("Error allocating memory for blocks");
        }
        blockptr->numentries = nnz;
        for (size_t k = 1; k <= static_cast<size_t>(nnz); k++) {
          // iindices[0], jindices[0], entries[0] dont need initialization
          // Python index starts from zero
          blockptr->iindices[k] = *(long*)PyArray_GETPTR1(py_i, k-1);
          blockptr->jindices[k] = *(long*)PyArray_GETPTR1(py_j, k-1);
          blockptr->entries[k] = *(double*)PyArray_GETPTR1(py_a, k-1);
        }
        // Insert the j-th block of the i-th A constraint Matrix into the
        // linked list
        blockptr->next = A[i].blocks;
        A[i].blocks = blockptr;
        Py_DECREF(py_i);
        Py_DECREF(py_j);
        Py_DECREF(py_a);
      }
    }
  }
}

Solver::Solver(const Py::Object& py_c, const Py::Object& py_a,
    const Py::Object& py_b) : obj(0.0), status(0) {
  if (!Objective::check(py_c))
    throw Py::TypeError("Provide a Objective Object");
  Objective* C = static_cast<Objective*>(py_c.ptr());
  if (!Constraints::check(py_a))
    throw Py::TypeError("Provide a Constraints Object");
  Constraints* A = static_cast<Constraints*>(py_a.ptr());
  ncons = A->length();
  PyArrayObject* py_arr = NULL;
  py_arr = (PyArrayObject*)PyArray_ContiguousFromObject(py_b.ptr(),
                                                        PyArray_DOUBLE, 1, 1);
  if (py_arr == NULL) {
    Py_XDECREF(py_arr);
    throw Py::ValueError("b needs to be a 1D Array");
  }
  double* b = NULL;
  b = (double*)malloc((ncons + 1)*sizeof(double));
  if (b == NULL) {
    throw Py::MemoryError("Error allocating memory for blocks");
  }
  for (size_t i = 1; i <= ncons; i++)
    b[i] = *(double*)PyArray_GETPTR1(py_arr,i-1);
  Py_DECREF(py_arr);
  initsoln(C->dim(), ncons, C->Matrix(), b, A->Matrix(), &X, &y, &Z);
//   write_prob("prob.dat-s",C->dim(),ncons,C->Matrix(),b,A->Matrix());
  double pobj, dobj;
  const double offset = 0.0;
  status = easy_sdp(C->dim(), ncons, C->Matrix(), b, A->Matrix(), offset, &X, &y,
      &Z, &pobj, &dobj);
//   write_sol("prob.sol",C->dim(),ncons,X,y,Z);
  obj = (dobj + pobj)/2;
  free(b);
  
}

Solver::~Solver(){
  free(y);
}

Constraints::~Constraints(){
  struct sparseblock* ptr;
  struct sparseblock* oldptr;
  if (A != NULL) {
    for (size_t i = 1; i <= ncons; i++) {
      ptr = A[i].blocks;
      while (ptr != NULL) {
        free(ptr->entries);
        free(ptr->iindices);
        free(ptr->jindices);
        oldptr = ptr;
        ptr = ptr->next;
        free(oldptr);
      }
    }
    free(A);
  }
}

Py::Object Csdp::new_solver(const Py::Tuple& args) {
  args.verify_length(3);
  return Py::asObject(new Solver(args[0], args[1], args[2]));
}

Py::Object Csdp::new_constraints(const Py::Tuple& args) {
  args.verify_length(1);
  return Py::asObject(new Constraints(args[0]));
}

Py::Object Csdp::new_objective(const Py::Tuple& args) {
  args.verify_length(1);
  return Py::asObject(new Objective(args[0]));
}

extern "C" void init_csdp() {
  static Csdp* Csdp_module = NULL;
  Csdp_module = new Csdp();
  import_array();
}
