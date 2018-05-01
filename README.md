# `pygsvd.py`

A Python wrapper to the LAPACK generalized singular value decomposition.

(C) 2017 Benjamin Naecker bnaecker@fastmail.com

## Overview

The `pygsvd` module exports a single function `gsvd`, which computes the
generalized singular value decomposition (GSVD) of a pair of matrices,
`A` and `B`. The [GSVD](https://en.wikipedia.org/wiki/Generalized_singular_value_decomposition)
is a joint decomposition useful for computing regularized solutions
to ill-posed least-squares problems, as well as dimensionality reduction
and clustering.

The `pygsvd` module is very simple: it just wraps the underlying LAPACK
routine `ggsvd3`, both the double-precision (`dggsvd3`) and complex-double
precision versions (`zggsvd3`).

## Building

Because the `pygsvd` module wraps a LAPACK routine itself, it is provided
as a Python and NumPy extension module. The module must be compiled,
and doing so requires a LAPACK header and a shared library. The module
currently supports both the standard C bindings to LAPACK (called 
[LAPACKE](http://www.netlib.org/lapack/lapacke.html)),
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

	$ python3 setup.py build_ext --include-dirs="/some/dir" --library-dirs="/some/libdir"

Then you can install the module either as usual or in develop mode as:

 	$ python3 setup.py {install,develop}

Or via `pip` as:

	$ pip3 install .

## Usage

The GSVD of a pair of NumPy ndarrays `a` and `b` can be computed as:

	>>> c, s, x = pygsvd.gsvd(a, b)

This returns the generalized singular values, in arrays `c` and `s`, and the
right generalized singular vectors in `x`. Optionally, the transformation matrices
`u` and` `v` may also be computed. E.g.:

	>>> c, s, x, u = pygsvd.gsvd(a, b, extras='u')

also returns the left generalized singular vectors of `a`.

By default, the matrices `u` and `v`, if returned, are of shape `(m, n)` and
`(p, n)`. Using the optional argument `full_matrices` is set to `True`, then
the matrices are square, of shape `(m, m)` and `(p, p)`.

## The generalized singular value decomposition

The GSVD is a joint decomposition of a pair of matrices. Given matrices
`A` with shape `(m, n)` and `B` with shape `(p, n)`, it computes:

        A = U*C*X.T
        B = V*S*X.T

where `U` and `V` are unitary matrices, with shapes `(m, m)` and `(p, p)`,
and `X` is shaped as `(n, n)`, respectively. `C` and `S` are diagonal (possibly non-square)
matrices containing the generalized singular value pairs.

This decomposition has many uses, including least-squares fitting of ill-posed
problems. For example, letting `B` be the "second derivative" operator one can
solve the equation

	min_x ||Ax - b||^2 + \lambda ||Bx||^2

using the GSVD, which achieves a smoother solution as `\lambda` is increased.
Similarly, setting `B` to the identity matrix, this becomes the standard
ridge regression problem. These are both versions of the Tichonov regularization
problem, for which the GSVD provides a useful and efficient solution.
