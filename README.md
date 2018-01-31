# `pygsvd.py`

A Python wrapper to the LAPACK generalized singular value decomposition.

(C) 2017 Benjamin Naecker bnaecker@fastmail.com

## Overview

The `pygsvd` module exports a single function `gsvd`, which computes the
generalized singular value decomposition (GSVD) of a pair of matrices,
`A` and `B`. The [GSVD](https://en.wikipedia.org/wiki/Generalized_singular_value_decomposition)
is a joint decomposition useful in for computing regularized solutions
to ill-posed least-squares problems, as well as dimensionality reduction
and clustering.

The `pygsvd` module is very simple: it just wraps the underlying LAPACK
routine `ggsvd3`, both the double-precision (`dggsvd3`) and complex-double
precision versions (`zggsvd3`).

## Building

Because the `pygsvd` module wraps a LAPACK routine itself, it is provded
as a Python and NumPy extension module. The module must be compiled,
and doing so requires a LAPACK header and a shared library. The module
currently supports both the standard C bindings to LAPACK (called 
[LAPACKE](http://www.netlib.org/lapack/lapacke.html),
and those provided by Intel's Math Kernel Library. Notably it does *not*
support Apple's Accelerate framework, which seems to be outdated and
differs in several subtle and annoying ways.

You can build against either of the supported implementations, by editing 
the `setup.cfg` file. Set the `define=` line in the file to be one of 
`USE_LAPACK` (the default) or `USE_MKL`.

You must also add the include and library directories for these. The
build process already searches `/usr/local/{include,lib}`, but if these
don't contain the header and library, add the directory containing these
to the `include_dirs=` and `library_dirs=` line. Multiple directories are
separated by a `:`. You can also set these on the command line when building.

For example, to use the LAPACK library, with a header in `/some/dir/`
and the library in `/some/libdir/`, you could run:

	$ python3 setup.py build_ext --include-dirs="/some/dir" --library_dirs="/some/libdir"

Then you can install the module either as usual or in develop mode as:

 	$ python3 setup.py install/develop

Or via `pip` as:

	$ pip3 install .

## Usage

The GSVD of a pair of NumPy ndarrays `a` and `b` can be computed as:

	>>> c, s, r = pygsvd.gsvd(a, b)

This returns the generalized singular values, in `c` and `s`, and the
upper triangular matrix `r`. Optionally, the transformation matrices
`u`, `v`, and `q` may also be computed. E.g.:

	>>> c, s, r, q = pygsvd.gsvd(a, b, extras='q')

also returns the right generalized singular vectors of `a` and `b`.

## The generalized singular value decomposition

The GSVD is a joint decomposition of a pair of matrices. Given matrices
`A` with shape `(m, n)` and `B` with shape `(p, n)`, it computes:

        U.T A Q = D_1 (0 R)
        V.T B Q = D_2 (0 R)

where `U`, `V` and `Q` are unitary matrices, with shapes `(m, m)`, `(p, p)`,
and `(n, n)`, respectively. `D_1` and `D_2` are "diagonal" (possibly non-square) 
matrices containing the generalized singular value pairs, and `R` is 
upper-triangular and non-singular, with shape `(r, r)`, where `r` is the
effective numerical rank of `(A.T; B.T).T`.

This decomposition has many uses, including least-squares fitting of ill-posed
problems. For example, letting `B` be the "second derivative" operator one can
solve the equation

	min_x ||Ax - b||^2 + \lambda ||Bx||^2

using the GSVD, which achieves a smoother solution as `\lambda` is increased.
Similarly, setting `B` to the identity matrix, this becomes the standard
ridge regression problem. These are both versions of the Tichonov regularization
problem, for which the GSVD provides a useful and efficient solution.
