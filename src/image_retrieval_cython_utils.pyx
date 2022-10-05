import numpy as np
cimport numpy as np
np.import_array()

from numpy.math cimport INFINITY

import cython


cpdef str _test_hello(str name):
    # Define a test function so we can check everything works
    return f'Hello {name}'


#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef np.ndarray[np.npy_bool] verify_model(
    np.ndarray[np.double_t] errors,
    np.ndarray[np.int64_t, ndim=2] corresp,
    int inlier_threshold,
    int num_words,
):
    cdef np.ndarray[np.npy_bool] mask = np.zeros([corresp.shape[0]], dtype=bool)

    cdef np.ndarray[np.npy_bool] taken = np.zeros([num_words], dtype=bool)
    cdef int i = 0
    cdef int i_max = errors.shape[0]
    cdef int actual_y, best_index
    cdef double best_error
    while i < i_max:
        actual_y = corresp[i, 1]
        
        best_error = INFINITY
        best_index = -1
        while i < i_max and actual_y == corresp[i, 1]:
            if errors[i] < best_error and taken[corresp[i, 0]] is False:
                best_error = errors[i]
                best_index = i
            i += 1
        
        if best_error < inlier_threshold:
            mask[best_index] = True
            taken[corresp[best_index, 0]] = True
    
    return mask


@cython.cdivision(True)
#@cython.wraparound(False)
#@cython.boundscheck(False)
cpdef np.ndarray[np.double_t, ndim=2] affine_local_optimization(
    np.ndarray[np.double_t, ndim=2] A,
    np.ndarray[np.double_t, ndim=2] q_geom,
    np.ndarray[np.double_t, ndim=2] db_geom,
):
    cdef int size = q_geom.shape[0]
    
    if size < 3:
        return A
    
    cdef double weight = 2 * np.pi if size < 11 else 0
    cdef double r2h = 50.0  # half of squared circle radius is integrated over
    
    # Mean values
    cdef double q_mx = 0.
    cdef double q_my = 0.
    cdef double db_mx = 0.
    cdef double db_my = 0.
    cdef int i
    for i in range(size):
        q_mx += q_geom[i, 0]
        q_my += q_geom[i, 1]
        db_mx += db_geom[i, 0]
        db_my += db_geom[i, 1]
    q_mx /= size
    q_my /= size
    db_mx /= size
    db_my /= size
    
    # The computation of AtA, AtB1 and AtB2
    cdef double[3] AtA = [0.0, 0.0, 0.0]
    cdef double[2] AtB1 = [0.0, 0.0]
    cdef double[2] AtB2 = [0.0, 0.0]
    
    cdef double dxq, dyq
    for i in range(size):
        dxq = q_geom[i, 0] - q_mx
        dyq = q_geom[i, 1] - q_my
        
        AtA[0] += (1 + weight) * dxq * dxq + weight * r2h
        AtA[1] += (1 + weight) * dxq * dyq
        AtA[2] += (1 + weight) * dyq * dyq + weight * r2h

        AtB1[0] += (1 + weight) * dxq * (db_geom[i, 0] - db_mx) + weight * A[0, 0] * r2h
        AtB1[1] += (1 + weight) * dyq * (db_geom[i, 0] - db_mx) + weight * A[0, 1] * r2h

        AtB2[0] += (1 + weight) * dxq * (db_geom[i, 1] - db_my) + weight * A[1, 0] * r2h
        AtB2[1] += (1 + weight) * dyq * (db_geom[i, 1] - db_my) + weight * A[1, 1] * r2h

    # Final affine transformation
    cdef double detAtA = AtA[0] * AtA[2] - AtA[1] * AtA[1]
    if detAtA == 0:
        print('[!!!] det(AtA) == 0')
        return A
    
    cdef double norm = 1 / detAtA
    cdef double H0 = (AtA[2] * AtB1[0] - AtA[1] * AtB1[1]) * norm
    cdef double H1 = (-AtA[1] * AtB1[0] + AtA[0] * AtB1[1]) * norm
    cdef double H2 = db_mx - q_mx * H0 - q_my * H1
    cdef double H3 = (AtA[2] * AtB2[0] - AtA[1] * AtB2[1]) * norm
    cdef double H4 = (-AtA[1] * AtB2[0] + AtA[0] * AtB2[1]) * norm
    cdef double H5 = db_my - q_mx * H3 - q_my * H4
    
    cdef np.ndarray[np.double_t, ndim=2] H = np.array([
        [H0, H1, H2],
        [H3, H4, H5],
        [0, 0, 1],
    ])
    
    return H


#@cython.wraparound(False)
#@cython.boundscheck(False)
cpdef np.ndarray[np.int_t, ndim=2] get_tentative_correspondencies_cy(
    np.ndarray[np.uint32_t] q_original,
    np.ndarray[np.uint32_t] q_unique,
    np.ndarray[np.int64_t] q_counts,
    np.ndarray[np.int64_t] q_sorted,
    np.ndarray[np.uint32_t] db_original,
    int max_tc,
    int max_MxN,
):
    # Count all the visual words
    cdef np.ndarray[np.uint32_t] db_unique
    cdef np.ndarray[np.int64_t] db_counts
    db_unique, db_counts = np.unique(db_original, return_counts=True)
    
    # Argsort visual words so we can quickly get indices for final output
    cdef np.ndarray[np.int64_t] db_sorted
    db_sorted = np.argsort(db_original)
    
    # Variables for final output
    cdef list ret = []
    cdef list counts = []
    
    # All needed indices and temp variables
    cdef int qr_i = 0  # query index
    cdef int db_i = 0  # database index
    cdef int s_qr_i = 0  # sorted query index
    cdef int s_db_i = 0  # sorted database index
    cdef int count
    cdef int qr_len = q_unique.shape[0]
    cdef int db_len = db_unique.shape[0]
    cdef int s_qry_len = q_sorted.shape[0]
    cdef int s_rel_len = db_sorted.shape[0]
    cdef int s_qr_i_start
    
    while qr_i < qr_len and db_i < db_len:
        if q_unique[qr_i] == db_unique[db_i]:
            count = q_counts[qr_i] * db_counts[db_i]
            if count <= max_MxN:
                    
                while s_qr_i < s_qry_len and q_original[q_sorted[s_qr_i]] != q_unique[qr_i]:
                    s_qr_i += 1
                while s_db_i < s_rel_len and db_original[db_sorted[s_db_i]] != db_unique[db_i]:
                    s_db_i += 1
                        
                s_qr_i_start = s_qr_i
                while s_db_i < s_rel_len and db_original[db_sorted[s_db_i]] == db_unique[db_i]:
                    s_qr_i = s_qr_i_start
                    while s_qr_i < s_qry_len and q_original[q_sorted[s_qr_i]] == q_unique[qr_i]:
                        ret.append([q_sorted[s_qr_i], db_sorted[s_db_i]])
                        counts.append(count)
                        s_qr_i += 1
                    s_db_i += 1
                    
            qr_i += 1
            db_i += 1
        elif q_unique[qr_i] < db_unique[db_i]:
            qr_i += 1
        else:
            db_i += 1
    
    cdef np.ndarray[np.int64_t, ndim=2] ret_np = np.array(ret, dtype=int, ndmin=2)
    
    # If there are way too many correspondences, crop the result
    cdef np.ndarray[np.int64_t] counts_np, keys
    if ret_np.shape[0] > max_tc:
        counts_np = np.array(counts)
        keys = np.argsort(counts, kind='stable')[:max_tc]
        return ret_np[keys]

    return ret_np
