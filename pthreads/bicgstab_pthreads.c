#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <string.h>
#include <sys/time.h>

// Структура разреженной матрицы в формате CSR
typedef struct {
    int n;
    int nnz;
    double *values;
    int *col_idx;
    int *row_ptr;
} CSRMatrix;

// Типы операций для потоков
typedef enum {
    OP_SPMV,
    OP_DOT,
    OP_AXPY,
    OP_EXIT
} OperationType;

// Структура для передачи данных в потоки
typedef struct {
    int thread_id;
    int start_row;
    int end_row;
    
    // Указатели на данные
    CSRMatrix *A;
    double *x;
    double *y;
    double alpha;
    
    // Для редукций
    double partial_result;
    
    // Управление
    OperationType operation;
    int active;
} ThreadData;

// Глобальные переменные синхронизации
pthread_barrier_t barrier_start;
pthread_barrier_t barrier_end;
pthread_mutex_t mutex;
int num_threads;
int threads_should_exit = 0;

// Таймер
double get_wtime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Рабочая функция потока
void* worker_thread(void *arg) {
    ThreadData *data = (ThreadData*)arg;
    int i, j;
    double sum;
    
    while (1) {
        // Ждем сигнала к работе
        pthread_barrier_wait(&barrier_start);
        
        // Проверка на выход
        if (threads_should_exit) break;
        
        // Выполнение операции
        switch (data->operation) {
            case OP_SPMV:
                // y = A * x
                for (i = data->start_row; i < data->end_row; i++) {
                    sum = 0.0;
                    for (j = data->A->row_ptr[i]; j < data->A->row_ptr[i+1]; j++) {
                        sum += data->A->values[j] * data->x[data->A->col_idx[j]];
                    }
                    data->y[i] = sum;
                }
                break;
                
            case OP_DOT:
                // Скалярное произведение
                sum = 0.0;
                for (i = data->start_row; i < data->end_row; i++) {
                    sum += data->x[i] * data->y[i];
                }
                data->partial_result = sum;
                break;
                
            case OP_AXPY:
                // y = y + alpha * x
                for (i = data->start_row; i < data->end_row; i++) {
                    data->y[i] += data->alpha * data->x[i];
                }
                break;
                
            default:
                break;
        }
        
        // Сигнализируем о завершении
        pthread_barrier_wait(&barrier_end);
    }
    
    return NULL;
}

// Функция SpMV с использованием thread pool
void spmv(CSRMatrix *A, double *x, double *y, ThreadData *thread_data) {
    int i;
    
    // Установка параметров
    for (i = 0; i < num_threads; i++) {
        thread_data[i].operation = OP_SPMV;
        thread_data[i].A = A;
        thread_data[i].x = x;
        thread_data[i].y = y;
    }
    
    // Запуск работы
    pthread_barrier_wait(&barrier_start);
    // Ожидание завершения
    pthread_barrier_wait(&barrier_end);
}

// Скалярное произведение
double dot_product(double *x, double *y, ThreadData *thread_data) {
    int i;
    double result = 0.0;
    
    // Установка параметров
    for (i = 0; i < num_threads; i++) {
        thread_data[i].operation = OP_DOT;
        thread_data[i].x = x;
        thread_data[i].y = y;
    }
    
    // Запуск работы
    pthread_barrier_wait(&barrier_start);
    // Ожидание завершения
    pthread_barrier_wait(&barrier_end);
    
    // Редукция результатов
    for (i = 0; i < num_threads; i++) {
        result += thread_data[i].partial_result;
    }
    
    return result;
}

// AXPY операция
void axpy(double alpha, double *x, double *y, ThreadData *thread_data) {
    int i;
    
    // Установка параметров
    for (i = 0; i < num_threads; i++) {
        thread_data[i].operation = OP_AXPY;
        thread_data[i].x = x;
        thread_data[i].y = y;
        thread_data[i].alpha = alpha;
    }
    
    // Запуск работы
    pthread_barrier_wait(&barrier_start);
    // Ожидание завершения
    pthread_barrier_wait(&barrier_end);
}

// Метод BiCGStab
int bicgstab_solver(CSRMatrix *A, double *b, double *x, int max_iter, double tol, double *time_elapsed) {
    int n = A->n;
    int i, iter;
    double rho_old, alpha, omega, rho, beta, r_hat_v, s_norm, t_s, t_t, r_norm;
    
    double start_time = get_wtime();
    
    double *r = malloc(n * sizeof(double));
    double *r_hat = malloc(n * sizeof(double));
    double *p = malloc(n * sizeof(double));
    double *v = malloc(n * sizeof(double));
    double *s = malloc(n * sizeof(double));
    double *t = malloc(n * sizeof(double));
    double *temp = malloc(n * sizeof(double));
    
    // Создание thread pool
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    ThreadData *thread_data = malloc(num_threads * sizeof(ThreadData));
    
    // Инициализация барьеров
    pthread_barrier_init(&barrier_start, NULL, num_threads + 1);
    pthread_barrier_init(&barrier_end, NULL, num_threads + 1);
    pthread_mutex_init(&mutex, NULL);
    
    // Распределение работы
    int rows_per_thread = n / num_threads;
    for (i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == num_threads - 1) ? n : (i + 1) * rows_per_thread;
        thread_data[i].active = 1;
    }
    
    // Создание потоков
    threads_should_exit = 0;
    for (i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, worker_thread, &thread_data[i]);
    }
    
    // Инициализация
    spmv(A, x, temp, thread_data);
    for (i = 0; i < n; i++) {
        r[i] = b[i] - temp[i];
        r_hat[i] = r[i];
        p[i] = r[i];
    }
    
    rho_old = 1.0;
    alpha = 1.0;
    omega = 1.0;
    
    // Основной цикл
    for (iter = 0; iter < max_iter; iter++) {
        rho = dot_product(r_hat, r, thread_data);
        
        if (fabs(rho) < 1e-30) break;
        
        if (iter > 0) {
            beta = (rho / rho_old) * (alpha / omega);
            for (i = 0; i < n; i++) {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
        }
        
        spmv(A, p, v, thread_data);
        r_hat_v = dot_product(r_hat, v, thread_data);
        alpha = rho / r_hat_v;
        
        for (i = 0; i < n; i++) {
            s[i] = r[i] - alpha * v[i];
        }
        
        s_norm = sqrt(dot_product(s, s, thread_data));
        if (s_norm < tol) {
            for (i = 0; i < n; i++) {
                x[i] += alpha * p[i];
            }
            break;
        }
        
        spmv(A, s, t, thread_data);
        t_s = dot_product(t, s, thread_data);
        t_t = dot_product(t, t, thread_data);
        omega = t_s / t_t;
        
        for (i = 0; i < n; i++) {
            x[i] += alpha * p[i] + omega * s[i];
        }
        
        for (i = 0; i < n; i++) {
            r[i] = s[i] - omega * t[i];
        }
        
        r_norm = sqrt(dot_product(r, r, thread_data));
        if (r_norm < tol) break;
        
        rho_old = rho;
    }
    
    // Завершение потоков
    threads_should_exit = 1;
    pthread_barrier_wait(&barrier_start);
    
    for (i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    double end_time = get_wtime();
    *time_elapsed = end_time - start_time;
    
    // Очистка
    pthread_barrier_destroy(&barrier_start);
    pthread_barrier_destroy(&barrier_end);
    pthread_mutex_destroy(&mutex);
    
    free(r); free(r_hat); free(p); free(v); free(s); free(t); free(temp);
    free(threads); free(thread_data);
    
    return iter;
}

int main(int argc, char *argv[]) {
    int n = 200000;
    int nnz = 3 * n - 2;
    int i, idx, iterations;
    double elapsed_time;
    
    num_threads = (argc > 1) ? atoi(argv[1]) : 1;
    
    CSRMatrix A;
    A.n = n;
    A.nnz = nnz;
    A.values = malloc(nnz * sizeof(double));
    A.col_idx = malloc(nnz * sizeof(int));
    A.row_ptr = malloc((n + 1) * sizeof(int));
    
    // Трехдиагональная матрица
    idx = 0;
    for (i = 0; i < n; i++) {
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
    
    // Правая часть
    double *b = malloc(n * sizeof(double));
    double *x = calloc(n, sizeof(double));
    for (i = 0; i < n; i++) {
        b[i] = 1.0;
    }
    
    // Решение
    iterations = bicgstab_solver(&A, b, x, 10000, 1e-10, &elapsed_time);

    //printf("%d,%.6f,%d\n", num_threads, elapsed_time, iterations);
    
    free(A.values); free(A.col_idx); free(A.row_ptr);
    free(b); free(x);
    
    return 0;
}
