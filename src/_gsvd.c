/* _gsvd.c
 *
 * Implementation of a simple Python/NumPy extension module which
 * wraps the LAPACK routines for computing the generalized singular
 * value decomposition (GSVD) of a pair of matrices.
 *
 * The implementation accepts both double-precision and complex-double
 * arrays as inputs.
 */

#include <complex.h>
#include <stdio.h>

#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#ifdef USE_LAPACK
# include "lapacke.h"
#elif defined USE_MKL
# include "mkl.h"
#else
# error "Cannot find a LAPACK header, define either USE_MKL or USE_LAPACK"
#endif

/* gsvd
 *
 * Implements the single exported routine, computing the 
 * generalized singular value decomposition of a pair of matrices
 * A and B. The routine should be called from Python with the 
 * following signature:
 *
 * gsvd(A, B, U, V, Q, C, S, iwork, 
 * 		compute_u, compute_v, compute_q)
 *
 * Where A and B are the input matrices whose GSVD is computed.
 * On exit from the routine, these arrays store portions of the
 * diagonalized matrix R. U, V, and Q are arrays for storing the 
 * (optional) orthogonal transformation matrices, if the routine 
 * computes them. C and S store the generalized singular values 
 * on exit. compute_{u,v,q} are flags indicating whether to compute 
 * the corresponding transformation matrix.
 */
static PyObject *gsvd(PyObject *self, PyObject *args) {

	/* Converted array objects from inputs/outputs. */
	PyArrayObject *A = NULL; // input array 1, converted from arg1
	PyArrayObject *B = NULL; // input array 2, converted from arg2
	PyArrayObject *U = NULL; // output array, converted from u_out
	PyArrayObject *V = NULL; // output array, converted from v_out
	PyArrayObject *Q = NULL; // output array, converted from q_out
	PyArrayObject *alpha = NULL; // output array, converted from alpha_out
	PyArrayObject *beta = NULL; // output array, converted from beta_out
	PyArrayObject *iwork = NULL; // output array, containing sorting information
    PyObject *ret = NULL; // tuple returned with effective rank information
	int compute_u = 0;
	int compute_v = 0;
	int compute_q = 0;

	/* Shapes of various objects. */
	int m = 0; // # rows of arr1
	int p = 0; // # rows of arr2
	int n = 0; // # cols of both arrays
	int k = 0; // output argument, describes shape of diagonal blocks
	int l = 0; // output argument, describes sahpe of diagonal blocks
	int lda = 0; // # of rows of arr1
	int ldb = 0; // # of rows of arr2
	int ldu = 0; // # rows of U
	int ldv = 0; // # rows of V
	int ldq = 0; // # rows of Q

	/* Flags to LAPACK routine for computing extra matrices. */
	char jobu = 'N';
	char jobv = 'N';
	char jobq = 'N';

	/* Unpack input arguments.
	 * Should be called as:
	 * gsvd(A, B, U, V, Q, C, S, iwork, 
	 * 		compute_u, compute_v, compute_q, call_complex)
	 */
	const char arg_format[] = "O!O!O!O!O!O!O!O!ppp";
	if (!PyArg_ParseTuple(args, arg_format,
				&PyArray_Type, &A,
				&PyArray_Type, &B,
				&PyArray_Type, &U,
				&PyArray_Type, &V,
				&PyArray_Type, &Q,
				&PyArray_Type, &alpha,
				&PyArray_Type, &beta,
				&PyArray_Type, &iwork,
				&compute_u,
				&compute_v,
				&compute_q)) {
		return NULL;
	}

	/* Update variables indicating whether to compute U, V, and Q. */
	if (compute_u)
		jobu = 'U';
	if (compute_v)
		jobv = 'V';
	if (compute_q)
		jobq = 'Q';

	/* Get and verify input array sizes. */
	if ( (PyArray_NDIM(A) != 2) || (PyArray_NDIM(B) != 2) ) {
		PyErr_SetString(PyExc_ValueError, 
				"Arrays must be 2-dimensional");
		return NULL;
	}
	npy_intp *a_dims = PyArray_DIMS(A);
	npy_intp *b_dims = PyArray_DIMS(B);
	m = a_dims[0];
	lda = a_dims[1];
	p = b_dims[0];
	ldb = b_dims[1];
	n = a_dims[1];
	if (n != b_dims[1]) {
		PyErr_SetString(PyExc_ValueError, 
				"Arrays must have the same number of columns");
		return NULL;
	}

	/* Set sizes of U, V, Q. */
	ldu = m;
	ldv = p;
	ldq = n;

	/* Compute GSVD via the LAPACK routine. */
	lapack_int lapack_ret = 0;
	if (PyArray_ISCOMPLEX(A)) {
		lapack_ret = LAPACKE_zggsvd3(LAPACK_ROW_MAJOR,
				jobu, jobv, jobq,
				m, n, p, &k, &l, 
				(lapack_complex_double *) PyArray_DATA(A), lda,
				(lapack_complex_double *) PyArray_DATA(B), ldb,
				(double *) PyArray_DATA(alpha), 
				(double *) PyArray_DATA(beta),
				(lapack_complex_double *) PyArray_DATA(U), ldu,
				(lapack_complex_double *) PyArray_DATA(V), ldv,
				(lapack_complex_double *) PyArray_DATA(Q), ldq,
				(lapack_int *) PyArray_DATA(iwork));
	} else {
		lapack_ret = LAPACKE_dggsvd3(LAPACK_ROW_MAJOR,
				jobu, jobv, jobq,
				m, n, p, &k, &l, 
				(double *) PyArray_DATA(A), lda,
				(double *) PyArray_DATA(B), ldb,
				(double *) PyArray_DATA(alpha), 
				(double *) PyArray_DATA(beta),
				(double *) PyArray_DATA(U), ldu,
				(double *) PyArray_DATA(V), ldv,
				(double *) PyArray_DATA(Q), ldq,
				(lapack_int *) PyArray_DATA(iwork));
	}

	if (lapack_ret != 0) {
		char msg[256] = { '\0' };
		snprintf(msg, 256, "A LAPACK error occurred: parameter %d was invalid.",
				lapack_ret);
		PyErr_SetString(PyExc_RuntimeError, msg);
		return NULL;
	}

    /* Pack up tuple of return arguments, (k, l).
	 * k + l gives the effective numerical rank of (A.T, B.T).T
     */
    ret = Py_BuildValue("ii", k, l);
	return ret;
}

/* The _gsvd module exports a single method, gsvd. */
static PyMethodDef GsvdMethods[] = {
	{
		"gsvd", 
		(PyCFunction) gsvd, 
		METH_VARARGS,
		"Compute the generalized singular value decomposition of two matrices."
	},
	{NULL, NULL, 0, NULL }
};

/* Definition of the _gsvd module object. */
static struct PyModuleDef gsvdmodule = {
	PyModuleDef_HEAD_INIT,
	"_gsvd",
	"Compute the generalized singular value decomposition of two matrices.",
	-1,
	GsvdMethods
};

/* Initialize the module. */
PyMODINIT_FUNC
PyInit__gsvd(void) {
	import_array();
	return PyModule_Create(&gsvdmodule);
};

