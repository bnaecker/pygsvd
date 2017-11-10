import numpy as np
import _gsvd

def gsvd(A, B, extras=''):
    '''Compute the generalized singular value decomposition of
    a pair of matrices ``A`` and ``B``.

    The GSVD is defined as a joint decomposition, as follows.
    Letting ``k + l`` be the effective numerical rank of
    ``(A.T; B.T).T``,

        U.T A Q = D_1 (0 R)
        V.T B Q = D_2 (0 R)

    where ``U``, ``V`` and ``Q`` are unitary matrices,
    ``D_1`` and ``D_2`` are diagonal (possibly non-square)
    matrices containing the generalized singular value pairs,
    and ``R`` is upper-triangular and non-singular.
        
    Parameters
    ----------
    A, B : ndarray
        Input matrices on which to perform the decomposition. Must
        be no more than 2D (and will be promoted if only 1D). The
        matrices must also have the same number of columns.
    extras : str, optional
        A string indicating which of the orthogonal transformation
        matrices should be computed. By default, this only computes 
        the generalized singular values in ``C`` and ``S``, and the 
        diagonalized matrix ``R``. The string may contain any of 
        'u', 'v', or 'q' to indicate that the corresponding matrix 
        is to be computed.

    Returns
    -------
    C : ndarray
        The generalized singular values of ``A``.
    S : ndarray
        The generalized singular values of ``B``.
    R : ndarray
        The diagonalized matrix ``R``.
    U : ndarray
        The left generalized singular vectors of ``A``, with
        shape ``(m, m)``. This is only returned if 
        ``'u' in extras`` is True.
    V : ndarray
        The left generalized singular vectors of ``B``, with
        shape ``(p, p)``. This is only returned if 
        ``'v' in extras`` is True.
    Q : ndarray
        The right generalized singular vectors of ``A`` and ``B``,
        with shape ``(n, n)``. This is only returned if 
        ``'q' in extras`` is True.

    Raises
    ------
    A ValueError is raised if ``A`` and ``B`` do not have the same
    number of columns, or if they are not both 2D (1D input arrays
    will be promoted).

    A RuntimeError is raised if the underlying LAPACK routine fails.

    Notes
    -----
    The LAPACK interface for this routine computes slightly different
    matrices than the MATLAB version. First, the singular values are
    returned in decreasing order in this function, whereas MATLAB's
    returns them in increasing order. Second, the matrix ``X`` in
    MATLAB's version is not computed. It can be computed as:

        X = (R.dot(Q.T).T

    Also note that in many cases columns of the returned matrices
    can differ from those returned by MATLAB's routine by a factor
    of -1.
    '''
    # Copy the input arrays, of the right datatype.
    # The LAPACK routine stores R inside A and/or B, so we copy to
    # avoid modifying the caller's arrays.
    dtype = np.complex if (np.iscomplexobj(A) or np.iscomplexobj(B)) else np.double
    Ac = np.array(A, copy=True, dtype=dtype, order='C', ndmin=2)
    Bc = np.array(B, copy=True, dtype=dtype, order='C', ndmin=2)

    # Compute shape
    m, n = Ac.shape
    p = Bc.shape[0]
    if (n != Bc.shape[1]):
        raise ValueError('A and B must have the same number of columns')

    # Determine which arrays are to be computed
    compute_uvq = tuple(each in extras for each in 'uvq')

    # Allocate arrays on which LAPACK routine operates.
    # If computing the corresponding transformation matrix,
    # create a zero array of the right size. If not, make a
    # dummy array of shape (1, 1).
    sizes = (m, p, n)
    U, V, Q = (np.zeros((size, size), dtype=dtype) if compute 
            else np.zeros((1, 1), dtype=dtype)
            for size, compute in zip(sizes, compute_uvq))
    C = np.zeros((n,), dtype=np.double)
    S = np.zeros((n,), dtype=np.double)
    iwork = np.zeros((n,), dtype=np.int32)

    # Compute GSVD via LAPACK wrapper, returning the effective rank
    k, l = _gsvd.gsvd(Ac, Bc, U, V, Q, C, S, iwork,
            compute_uvq[0], compute_uvq[1], compute_uvq[2])

    # Sort the singular values using the sorting information
    # computed in the LAPACK routine
    for i in range(k, min(m, k + l)):
        ix = iwork[i] - 1
        C[i], C[ix] = C[ix], C[i]
        S[i], S[ix] = S[ix], S[i]

    # Extract the R matrix stored in Ac and Bc
    R = _extract_R(Ac, Bc, k, l)

    # Convert to diagonals
    C = np.diag(C)
    S = np.diag(S)

    # Return outputs
    outputs = (C, S, R) + tuple(arr for arr, compute in 
            zip((U, V, Q), compute_uvq) if compute)
    return outputs

def _extract_R(A, B, k, l):
    '''Extract the diagonalized matrix R from A and/or B.

    The indexing performed here is taken from the LAPACK routine
    help, which can be found here:

    ``http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_gab6c743f531c1b87922eb811cbc3ef645.html#gab6c743f531c1b87922eb811cbc3ef645``
    '''
    m, n = A.shape
    if (m - k - l) >= 0:
        R = np.zeros((k+l, n), dtype=A.dtype)
        R[:, (n-k-l):] = A[:k+l, n-k-l:n]
    else:
        R = np.zeros((k + l, k + l), dtype=A.dtype)
        R[:m, :] = A
        R[m:, m:] = B[(m-k):l, (n+m-k-l):]
    return R

