import cython
import numpy as np
from cpython.array cimport array, clone

@cython.boundscheck(False)
@cython.wraparound(False)
def update_D_c(long[::1] D_row, long[::1] D_col, long[::1] D_length,
long[::1] infeasible_indices_mask, long[::1] corrected_indices, long[::1] O, long[:, ::1] N):
    
    cdef long i, j, l
    cdef long length = 0, new_length = 0, k = 0, new_k = 0, new_D_total_length = 0
    cdef long index_D_length = 0, new_index_D_length = 0

    new_D_row_np = np.empty(10_000_000, dtype=np.int64)
    new_D_col_np = np.empty(10_000_000, dtype=np.int64)
    new_D_length_np = np.empty(1_000_000, dtype=np.int64)
    new_D_row_index_np = np.empty(1_000_000, dtype=np.int64)

    cdef long[::1] new_D_row = new_D_row_np
    cdef long[::1] new_D_col = new_D_col_np
    cdef long[::1] new_D_length = new_D_length_np
    cdef long[::1] new_D_row_index = new_D_row_index_np

    cdef long D_size = corrected_indices[corrected_indices.shape[0] - 1] + 1
    cdef long D_total_length = D_row.shape[0]
    cdef long Oshape = O.shape[0]

    while k < D_total_length:
        i = D_row[k]
        if infeasible_indices_mask[i]:
            k += D_length[index_D_length]
            index_D_length += 1
            continue
        j = D_col[k]
        
        if not infeasible_indices_mask[j]:
            new_D_row[new_k] = corrected_indices[i]
            new_D_col[new_k] = corrected_indices[j]
            new_k += 1
            new_length += 1
        
        length += 1
        
        if D_length[index_D_length] == length:
            for l in range(Oshape):
                if O[l] == corrected_indices[i]:
                    new_D_row[new_k] = corrected_indices[i]
                    new_D_col[new_k] = D_size + l
                    new_k += 1
                    new_length += 1
            new_D_length[new_index_D_length] = new_length
            new_D_row_index[new_index_D_length] = corrected_indices[i]
            new_D_total_length += new_length
            new_length = 0
            length = 0
            index_D_length += 1
            new_index_D_length += 1

        k += 1

    new_length = 0
    for i in range(Oshape):
        new_D_row[new_k] = D_size + i
        new_D_col[new_k] = O[i]
        new_k += 1
        new_length += 1
        for j in range(Oshape):
            if N[i, j]:
                new_D_row[new_k] = D_size + i
                new_D_col[new_k] = D_size + j
                new_k += 1
                new_length += 1
        new_D_length[new_index_D_length] = new_length
        new_D_row_index[new_index_D_length] = D_size + i
        new_D_total_length += new_length
        new_length = 0
        new_index_D_length += 1

    output = {"row": new_D_row_np[:new_D_total_length],
             "col": new_D_col[:new_D_total_length],
             "length": new_D_length[:new_index_D_length],
             "row_index": new_D_row_index[:new_index_D_length],}

    return output