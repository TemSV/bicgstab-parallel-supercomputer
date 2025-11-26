#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <omp.h>

typedef struct {
    int n;
    int nnz;
    double *values;
    int *col_idx;
    int *row_ptr;
} CSRMatrix;

/* wall-clock */
double get_wtime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* user + system CPU времени текущего процесса */
void get_ru_times(double *u, double *s) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    *u = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6;
    *s = ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

/* y = A * x */
void spmv_omp(CSRMatrix *A, const double *x, double *y) {
    int n = A->n;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = A->row_ptr[i]; j < A->row_ptr[i+1]; j++) {
            sum += A->values[j] * x[A->col_idx[j]];
        }
        y[i] = sum;
    }
}

/* (x, y) */
double dot_product_omp(const double *x, const double *y, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

/* ||x||_2 */
double norm2_omp(const double *x, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}

/* BiCGStab + OpenMP */
int bicgstab_solver_omp(CSRMatrix *A,
                        const double *b,
                        double *x,
                        int max_iter, double tol,
                        double *time_elapsed)
{
    int n = A->n;

    double *r  = (double*)malloc(n * sizeof(double));
    double *r_hat = (double*)malloc(n * sizeof(double));
    double *p  = (double*)malloc(n * sizeof(double));
    double *v  = (double*)malloc(n * sizeof(double));
    double *s  = (double*)malloc(n * sizeof(double));
    double *t  = (double*)malloc(n * sizeof(double));
    double *Ax = (double*)malloc(n * sizeof(double));

    /* r = b - A*x */
    spmv_omp(A, x, Ax);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        r[i]     = b[i] - Ax[i];
        r_hat[i] = r[i];
        p[i]     = r[i];
    }

    double rho_old = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    double t_start = get_wtime();
    int iter;

    for (iter = 0; iter < max_iter; iter++) {
        double rho = dot_product_omp(r_hat, r, n);
        if (fabs(rho) < 1e-30) break;

        if (iter > 0) {
            double beta = (rho / rho_old) * (alpha / omega);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
        }

        /* v = A * p */
        spmv_omp(A, p, v);

        double r_hat_v = dot_product_omp(r_hat, v, n);
        if (fabs(r_hat_v) < 1e-30) break;
        alpha = rho / r_hat_v;

        /* s = r - alpha * v */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            s[i] = r[i] - alpha * v[i];
        }

        double s_norm = norm2_omp(s, n);
        if (s_norm < tol) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) {
                x[i] += alpha * p[i];
            }
            iter++;
            break;
        }

        /* t = A * s */
        spmv_omp(A, s, t);

        double t_s = dot_product_omp(t, s, n);
        double t_t = dot_product_omp(t, t, n);
        if (t_t < 1e-30) break;
        omega = t_s / t_t;

        /* x += alpha * p + omega * s */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i] + omega * s[i];
        }

        /* r = s - omega * t */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            r[i] = s[i] - omega * t[i];
        }

        double r_norm = norm2_omp(r, n);
        if (r_norm < tol) {
            iter++;
            break;
        }

        rho_old = rho;
    }

    double t_end = get_wtime();
    *time_elapsed = t_end - t_start;

    free(r);
    free(r_hat);
    free(p);
    free(v);
    free(s);
    free(t);
    free(Ax);

    return iter;
}

int main(int argc, char *argv[])
{
    int n   = 200000;
    int nnz = 3 * n - 2;

    CSRMatrix A;
    A.n = n;
    A.nnz = nnz;
    A.values = (double*)malloc(nnz * sizeof(double));
    A.col_idx = (int*)malloc(nnz * sizeof(int));
    A.row_ptr = (int*)malloc((n + 1) * sizeof(int));

    /* трёхдиагональная матрица */
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

    double *b = (double*)malloc(n * sizeof(double));
    double *x = (double*)calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        b[i] = 1.0;
    }

    double u0, s0, u1, s1;
    double t0, t1;

    get_ru_times(&u0, &s0);
    t0 = get_wtime();

    double elapsed;
    int iters = bicgstab_solver_omp(&A, b, x, 10000, 1e-10, &elapsed);

    get_ru_times(&u1, &s1);
    t1 = get_wtime();

    double real = t1 - t0;
    double user = u1 - u0;
    double sys  = s1 - s0;

    int threads = omp_get_max_threads();
    /* threads, real, iters, user, sys */
    printf("%d %.6f %d %.6f %.6f\n", threads, real, iters, user, sys);

    free(A.values);
    free(A.col_idx);
    free(A.row_ptr);
    free(b);
    free(x);

    return 0;
}
