import numpy as np
import _gsvd

def gsvd(A, B, full_matrices=False, extras='uv'):
    '''Compute the generalized singular value decomposition of
    a pair of matrices ``A`` of shape ``(m, n)`` and ``B`` of
    shape ``(p, n)``

    The GSVD is defined as a joint decomposition, as follows.
        
        A = U*C*X.T
        B = V*S*X.T
    
    where

        C.T*C + S.T*S = I

    where ``U`` and ``V`` are unitary matrices.
        
    Parameters
    ----------
    A, B : ndarray
        Input matrices on which to perform the decomposition. Must
        be no more than 2D (and will be promoted if only 1D). The
        matrices must also have the same number of columns.
    full_matrices : bool, optional
        If ``True``, the returned matrices ``U`` and ``V`` have
        at most ``p`` columns and ``C`` and ``S`` are of length ``p``.
    extras : str, optional
        A string indicating which of the orthogonal transformation
        matrices should be computed. By default, this only computes 
        the generalized singular values in ``C`` and ``S``, and the 
        right generalized singular vectors in ``X``. The string may
        contain either 'u' or 'v' to indicate that the corresponding
        matrix is to be computed.

    Returns
    -------
    C : ndarray
        The generalized singular values of ``A``. These are returned
        in decreasing order.
    S : ndarray
        The generalized singular values of ``B``. These are returned
        in increasing order.
    X : ndarray
        The right generalized singular vectors of ``A`` and ``B``.
    U : ndarray
        The left generalized singular vectors of ``A``, with
        shape ``(m, m)``. This is only returned if 
        ``'u' in extras`` is True.
    V : ndarray
        The left generalized singular vectors of ``B``, with
        shape ``(p, p)``. This is only returned if 
        ``'v' in extras`` is True.

    Raises
    ------
    A ValueError is raised if ``A`` and ``B`` do not have the same
    number of columns, or if they are not both 2D (1D input arrays
    will be promoted).

    A RuntimeError is raised if the underlying LAPACK routine fails.

    Notes
    -----
    This routine is intended to be as similar as possible to the
    decomposition provided by Matlab and Octave. Note that this is slightly
    different from the decomposition as put forth in Golub and Van Loan [1],
    and that this routine is thus not directly a wrapper for the underlying
    LAPACK routine.

    One important difference between this routine and that provided by
    Matlab is that this routine returns the singular values in decreasing
    order, for consistency with NumPy's ``svd`` routine.

    References
    ----------
    [1] Golub, G., and C.F. Van Loan, 2013, Matrix Computations, 4th Ed.
    '''
    # The LAPACK routine stores R inside A and/or B, so we copy to
    # avoid modifying the caller's arrays.
    dtype = np.complex128 if any(map(np.iscomplexobj, (A, B))) else np.double
    Ac = np.array(A, copy=True, dtype=dtype, order='C', ndmin=2)
    Bc = np.array(B, copy=True, dtype=dtype, order='C', ndmin=2)
    m, n = Ac.shape
    p = Bc.shape[0]
    if (n != Bc.shape[1]):
        raise ValueError('A and B must have the same number of columns')

    # Allocate input arrays to LAPACK routine
    compute_uv = tuple(each in extras for each in 'uv')
    sizes = (m, p)
    U, V = (np.zeros((size, size), dtype=dtype) if compute 
            else np.zeros((1, 1), dtype=dtype)
            for size, compute in zip(sizes, compute_uv))
    Q = np.zeros((n, n), dtype=dtype)
    C = np.zeros((n,), dtype=np.double)
    S = np.zeros((n,), dtype=np.double)
    iwork = np.zeros((n,), dtype=np.int32)

    # Compute GSVD via LAPACK wrapper, returning the effective rank
    k, l = _gsvd.gsvd(Ac, Bc, U, V, Q, C, S, iwork,
            compute_uv[0], compute_uv[1])

    # Compute X
    R = _extract_R(Ac, Bc, k, l)
    X = R.dot(Q.T).T

    # Sort and sub-sample if needed
    rank = k + l
    ix = np.argsort(C[:rank])[::-1]
    C = C[ix]
    S = S[ix]
    X[:, :rank] = X[:, ix]
    if compute_uv[0]:
        U[:, :rank] = U[:, ix]
    if compute_uv[1]:
        # Handle rank-deficient inputs
        if k:
            V = np.roll(V, k, axis=1)
        V[:, :rank] = V[:, ix]
    if not full_matrices:
        X = X[:, :rank]
        if compute_uv[0]:
            U = U[:, :rank]
        if compute_uv[1]:
            V = V[:, :rank]

    outputs = (C, S, X) + tuple(arr for arr, compute in 
            zip((U, V), compute_uv) if compute)
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

