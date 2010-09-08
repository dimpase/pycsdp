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

#ifndef _CSDP_H
#define _CSDP_H


#include "CXX/Extensions.hxx"
#include "CXX/Objects.hxx"
#include "numpy/arrayobject.h"

extern "C" {
#include "declarations.h"
};

typedef struct blockmatrix blockmatrix;
typedef struct constraintmatrix constraintmatrix;

class Objective : public Py::PythonExtension<Objective> {
  public:
    static void init_type(void);
    Objective(const struct blockmatrix& X);
    Objective(const Py::List& array);
    const blockmatrix& Matrix(void) {
      return C;
    };
    size_t dim(void) {
      return dimension;
    };
    virtual ~Objective() {
      free_mat(C);
    };
    Py::Object get_data(void);
  private:
    blockmatrix C;
    size_t dimension;
};

class Constraints : public Py::PythonExtension<Constraints> {
  public:
    static void init_type(void);
    Constraints(const Py::List& array);
    virtual ~Constraints();
    constraintmatrix* const& Matrix(void) {
      return A;
    };
    size_t length(void) {
      return ncons;
    };
  private:
    size_t ncons;
    constraintmatrix* A;
};

class Solver : public Py::PythonExtension<Solver> {
  public:
    static void init_type(void);
    Solver(const Py::Object& py_c, const Py::Object& py_a,
        const Py::Object& py_b);
    virtual ~Solver();
    Py::Object get_solution(void);
  private:
    // Primal Solution
    blockmatrix X;
    // Dual Solution
    blockmatrix Z;
    // Dual Solution y vector
    double* y;
    // Objective Value
    double obj;
    int status;
    size_t ncons;
};

class Csdp : public Py::ExtensionModule<Csdp> {
  public:
    Csdp();
    virtual ~Csdp() {
    };
  private:
    Py::Object new_solver(const Py::Tuple& args);
    Py::Object new_constraints(const Py::Tuple& args);
    Py::Object new_objective(const Py::Tuple& args);
};

#endif
