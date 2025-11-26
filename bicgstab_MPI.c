#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>

typedef struct {
    int n;
    int nnz;
    double *values;
    int *col_idx;
    int *row_ptr;
} CSRMatrix;

double get_wtime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Локальный SpMV: y_local[i] = (A * x_global) для строк [row_start, row_end)
void spmv_local(CSRMatrix *A, double *x_global,
                double *y_local, int row_start, int row_end)
{
    int i, j;
    double sum;
    for (i = row_start; i < row_end; i++) {
        sum = 0.0;
        for (j = A->row_ptr[i]; j < A->row_ptr[i+1]; j++) {
            sum += A->values[j] * x_global[A->col_idx[j]];
        }
        y_local[i - row_start] = sum;
    }
}

// Локальный скалярный + Allreduce => глобальный dot
double dot_product_mpi(double *x_local, double *y_local, int local_n, MPI_Comm comm)
{
    double local_sum = 0.0;
    for (int i = 0; i < local_n; i++) {
        local_sum += x_local[i] * y_local[i];
    }
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global_sum;
}

// Норма 2 локального вектора + Allreduce
double norm2_mpi(double *x_local, int local_n, MPI_Comm comm)
{
    double local_sum = 0.0;
    for (int i = 0; i < local_n; i++) {
        local_sum += x_local[i] * x_local[i];
    }
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    return sqrt(global_sum);
}

// Распределение строк по процессам (блочное)
void get_local_rows(int n, int rank, int size, int *row_start, int *row_end)
{
    int base = n / size;
    int rem  = n % size; // первые rem процессов получат по (base+1)
    if (rank < rem) {
        *row_start = rank * (base + 1);
        *row_end   = *row_start + (base + 1);
    } else {
        *row_start = rem * (base + 1) + (rank - rem) * base;
        *row_end   = *row_start + base;
    }
}

// BiCGStab с MPI (распределены строки и векторы, x и b глобальные/отовсюду)
int bicgstab_solver_mpi(CSRMatrix *A,
                        double *b_global,  // длина n, доступна везде
                        double *x_global,  // длина n, доступна везде
                        int max_iter, double tol,
                        double *time_elapsed,
                        MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int n = A->n;
    int row_start, row_end;
    get_local_rows(n, rank, size, &row_start, &row_end);
    int local_n = row_end - row_start;

    // Локальные куски векторов размера local_n
    double *r  = (double*)malloc(local_n * sizeof(double));
    double *r_hat = (double*)malloc(local_n * sizeof(double));
    double *p  = (double*)malloc(local_n * sizeof(double));
    double *v  = (double*)malloc(local_n * sizeof(double));
    double *s  = (double*)malloc(local_n * sizeof(double));
    double *t  = (double*)malloc(local_n * sizeof(double));
    double *temp_local = (double*)malloc(local_n * sizeof(double));

    // Вспомогательные для раздачи/сборки векторов по процессам
    int *counts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));
    for (int rnk = 0; rnk < size; rnk++) {
        int rs, re;
        get_local_rows(n, rnk, size, &rs, &re);
        counts[rnk] = re - rs;
        displs[rnk] = rs;
    }

    MPI_Barrier(comm);
    double start_time = get_wtime();

    // Начальное вычисление r = b - A*x
    // 1) каждый процесс считает свой temp_local = (A * x)_local
    spmv_local(A, x_global, temp_local, row_start, row_end);
    // 2) r_local = b_local - temp_local
    for (int i = 0; i < local_n; i++) {
        double b_i = b_global[row_start + i];
        r[i] = b_i - temp_local[i];
        r_hat[i] = r[i];
        p[i] = r[i];
    }

    double rho_old = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        double rho = dot_product_mpi(r_hat, r, local_n, comm);
        if (fabs(rho) < 1e-30) {
            break;
        }

        if (iter > 0) {
            double beta = (rho / rho_old) * (alpha / omega);
            for (int i = 0; i < local_n; i++) {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
        }

        // Собираем p_local со всех процессов в глобальный p_global
        double *p_global = (double*)malloc(n * sizeof(double));
        MPI_Allgatherv(p, local_n, MPI_DOUBLE,
                       p_global, counts, displs, MPI_DOUBLE, comm);

        // v_local = A * p_global
        spmv_local(A, p_global, v, row_start, row_end);

        // r_hat_v = (r_hat, v)
        double r_hat_v = dot_product_mpi(r_hat, v, local_n, comm);
        alpha = rho / r_hat_v;

        // s_local = r_local - alpha * v_local
        for (int i = 0; i < local_n; i++) {
            s[i] = r[i] - alpha * v[i];
        }

        // s_norm (глобальная)
        double s_norm = norm2_mpi(s, local_n, comm);
        if (s_norm < tol) {
            // x_local += alpha * p_local
            for (int i = 0; i < local_n; i++) {
                x_global[row_start + i] += alpha * p[i];
            }
            // Обновляем x_global между процессами
            MPI_Allgatherv(&x_global[row_start], local_n, MPI_DOUBLE,
                           x_global, counts, displs, MPI_DOUBLE, comm);
            free(p_global);
            break;
        }

        // t_local = A * s_global (нужно собрать s_global)
        double *s_global = (double*)malloc(n * sizeof(double));
        MPI_Allgatherv(s, local_n, MPI_DOUBLE,
                       s_global, counts, displs, MPI_DOUBLE, comm);

        spmv_local(A, s_global, t, row_start, row_end);

        double t_s = dot_product_mpi(t, s, local_n, comm);
        double t_t = dot_product_mpi(t, t, local_n, comm);
        omega = t_s / t_t;

        // x_local += alpha * p_local + omega * s_local
        for (int i = 0; i < local_n; i++) {
            x_global[row_start + i] += alpha * p[i] + omega * s[i];
        }
        // Обновляем x_global между процессами
        MPI_Allgatherv(&x_global[row_start], local_n, MPI_DOUBLE,
                       x_global, counts, displs, MPI_DOUBLE, comm);

        // r_local = s_local - omega * t_local
        for (int i = 0; i < local_n; i++) {
            r[i] = s[i] - omega * t[i];
        }

        double r_norm = norm2_mpi(r, local_n, comm);
        if (r_norm < tol) {
            free(p_global);
            free(s_global);
            break;
        }

        rho_old = rho;
        free(p_global);
        free(s_global);
    }

    MPI_Barrier(comm);
    double end_time = get_wtime();
    *time_elapsed = end_time - start_time;

    free(r);
    free(r_hat);
    free(p);
    free(v);
    free(s);
    free(t);
    free(temp_local);
    free(counts);
    free(displs);

    return iter;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int n   = 200000;
    int nnz = 3 * n - 2;

    CSRMatrix A;
    A.n = n;
    A.nnz = nnz;
    A.values = (double*)malloc(nnz * sizeof(double));
    A.col_idx = (int*)malloc(nnz * sizeof(int));
    A.row_ptr = (int*)malloc((n + 1) * sizeof(int));

    // Трёхдиагональная матрица (строим во всех процессах одинаково)
    int idx = 0;
    for (int i = 0; i < n; i++) {
        A.row_ptr[i] = idx;
        if (i > 0) {
            A.values[idx] = 1.0;
            A.col_idx[idx] = i - 1;
            idx++;
        }
        A.values[idx] = -2.0;
        A.col_idx[idx] = i;
        idx++;
        if (i < n - 1) {
            A.values[idx] = 1.0;
            A.col_idx[idx] = i + 1;
            idx++;
        }
    }
    A.row_ptr[n] = idx;

    // Правая часть и начальное приближение (храним глобально в каждом процессе)
    double *b = (double*)malloc(n * sizeof(double));
    double *x = (double*)calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        b[i] = 1.0;
    }

    double elapsed_time;
    int iterations = bicgstab_solver_mpi(&A, b, x, 10000, 1e-10, &elapsed_time, comm);

    if (rank == 0) {
        printf("%d,%f,%d\n", size, elapsed_time, iterations);
    }

    free(A.values);
    free(A.col_idx);
    free(A.row_ptr);
    free(b);
    free(x);

    MPI_Finalize();
    return 0;
}
