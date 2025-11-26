from mpi4py import MPI
import numpy as np
import time
import resource


def get_local_rows(n, rank, size):
    """Блочное разбиение строк"""
    base = n // size
    rem = n % size
    if rank < rem:
        row_start = rank * (base + 1)
        row_end = row_start + (base + 1)
    else:
        row_start = rem * (base + 1) + (rank - rem) * base
        row_end = row_start + base
    return row_start, row_end


def spmv_local(values, col_idx, row_ptr, x_global, row_start, row_end):
    """Локальный SpMV: считает строки [row_start, row_end)."""
    local_n = row_end - row_start
    y_local = np.empty(local_n, dtype=np.float64)
    for i in range(row_start, row_end):
        s = 0.0
        for j in range(row_ptr[i], row_ptr[i + 1]):
            s += values[j] * x_global[col_idx[j]]
        y_local[i - row_start] = s
    return y_local


def dot_product_mpi(x_local, y_local, comm):
    """Локальный dot + Allreduce."""
    local_sum = np.dot(x_local, y_local)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    return global_sum


def norm2_mpi(x_local, comm):
    """Глобальная евклидова норма."""
    local_sum = np.dot(x_local, x_local)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    return np.sqrt(global_sum)


def get_ru_times():
    """Вернуть (user, sys) для текущего процесса в секундах."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime, ru.ru_stime


def bicgstab_solver_mpi(values, col_idx, row_ptr,
                        b_global, x_global,
                        max_iter, tol, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    n = b_global.size
    row_start, row_end = get_local_rows(n, rank, size)
    local_n = row_end - row_start

    # Локальные векторы
    r = np.empty(local_n, dtype=np.float64)
    r_hat = np.empty(local_n, dtype=np.float64)
    p = np.empty(local_n, dtype=np.float64)
    v = np.empty(local_n, dtype=np.float64)
    s = np.empty(local_n, dtype=np.float64)
    t = np.empty(local_n, dtype=np.float64)

    # counts, displs для Allgatherv
    counts = np.empty(size, dtype=np.int32)
    displs = np.empty(size, dtype=np.int32)
    for rnk in range(size):
        rs, re = get_local_rows(n, rnk, size)
        counts[rnk] = re - rs
        displs[rnk] = rs

    comm.Barrier()
    start_time = time.time()

    # r = b - A x
    temp_local = spmv_local(values, col_idx, row_ptr, x_global, row_start, row_end)
    r[:] = b_global[row_start:row_end] - temp_local
    r_hat[:] = r
    p[:] = r

    rho_old = 1.0
    alpha = 1.0
    omega = 1.0

    for it in range(max_iter):
        rho = dot_product_mpi(r_hat, r, comm)
        if abs(rho) < 1e-30:
            break

        if it > 0:
            beta = (rho / rho_old) * (alpha / omega)
            p[:] = r + beta * (p - omega * v)

        # Собираем p_global
        p_global = np.empty(n, dtype=np.float64)
        comm.Allgatherv(
            [p, MPI.DOUBLE],
            [p_global, (counts, displs), MPI.DOUBLE]
        )

        # v_local = A * p_global
        v[:] = spmv_local(values, col_idx, row_ptr, p_global, row_start, row_end)

        r_hat_v = dot_product_mpi(r_hat, v, comm)
        if abs(r_hat_v) < 1e-30:
            break
        alpha = rho / r_hat_v

        # s_local = r - alpha * v
        s[:] = r - alpha * v

        s_norm = norm2_mpi(s, comm)
        if s_norm < tol:
            # x_local += alpha * p_local
            x_global[row_start:row_end] += alpha * p

            # синхронизируем x_global через временный буфер
            sendbuf = np.ascontiguousarray(x_global[row_start:row_end])
            x_tmp = np.empty_like(x_global)
            comm.Allgatherv(
                [sendbuf, MPI.DOUBLE],
                [x_tmp, (counts, displs), MPI.DOUBLE]
            )
            x_global[:] = x_tmp

            it += 1
            break

        # t_local = A * s_global
        s_global = np.empty(n, dtype=np.float64)
        comm.Allgatherv(
            [s, MPI.DOUBLE],
            [s_global, (counts, displs), MPI.DOUBLE]
        )
        t[:] = spmv_local(values, col_idx, row_ptr, s_global, row_start, row_end)

        t_s = dot_product_mpi(t, s, comm)
        t_t = dot_product_mpi(t, t, comm)
        if t_t < 1e-30:
            break
        omega = t_s / t_t

        # x_local += alpha * p_local + omega * s_local
        x_global[row_start:row_end] += alpha * p + omega * s

        # снова синхронизируем x_global через временный буфер
        sendbuf = np.ascontiguousarray(x_global[row_start:row_end])
        x_tmp = np.empty_like(x_global)
        comm.Allgatherv(
            [sendbuf, MPI.DOUBLE],
            [x_tmp, (counts, displs), MPI.DOUBLE]
        )
        x_global[:] = x_tmp

        # r_local = s_local - omega * t_local
        r[:] = s - omega * t

        r_norm = norm2_mpi(r, comm)
        if r_norm < tol:
            it += 1
            break

        rho_old = rho

    comm.Barrier()
    elapsed = time.time() - start_time
    return it, elapsed


def build_tridiag_csr(n):
    #трёхдиагональная матрица
    nnz = 3 * n - 2
    values = np.empty(nnz, dtype=np.float64)
    col_idx = np.empty(nnz, dtype=np.int32)
    row_ptr = np.empty(n + 1, dtype=np.int32)

    idx = 0
    for i in range(n):
        row_ptr[i] = idx
        if i > 0:
            values[idx] = 1.0
            col_idx[idx] = i - 1
            idx += 1
        values[idx] = -2.0
        col_idx[idx] = i
        idx += 1
        if i < n - 1:
            values[idx] = 1.0
            col_idx[idx] = i + 1
            idx += 1
    row_ptr[n] = idx
    return values, col_idx, row_ptr


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n = 200000
    values, col_idx, row_ptr = build_tridiag_csr(n)

    # b и x храним в каждом процессе целиком
    b = np.ones(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)

    max_iter = 10000
    tol = 1e-10

    # замеры до
    u0, s0 = get_ru_times()
    t0 = time.time()

    iters, elapsed = bicgstab_solver_mpi(
        values, col_idx, row_ptr,
        b, x,
        max_iter, tol, comm
    )

    # замеры после
    u1, s1 = get_ru_times()
    t1 = time.time()

    du = u1 - u0       # user CPU для этого процесса
    ds = s1 - s0       # system CPU
    dreal = t1 - t0    # wall-clock

    total_user = comm.reduce(du, op=MPI.SUM, root=0)
    total_sys  = comm.reduce(ds, op=MPI.SUM, root=0)
    max_real   = comm.reduce(dreal, op=MPI.MAX, root=0)

    if rank == 0:
        print("np   real[s]    iters    user[s]     sys[s]")
        print(f"{size:<3d} {max_real:9.6f} {total_user:10.6f} {total_sys:9.6f}") 


if __name__ == "__main__":
    main()
