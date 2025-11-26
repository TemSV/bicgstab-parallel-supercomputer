# Stabilized Bi-Conjugate Gradient Method

[Читать на русском](README_ru.md)

The repository contains implementations of the BiCGStab method for solving sparse systems of linear algebraic equations (SLAEs) on a supercomputer in several parallelization models: C/pthreads, C/MPI, Python/mpi4py, and C/OpenMP.

## Problem statement

We consider a system of linear equations of the form Ax = b, where A is an n×n coefficient matrix, x is the vector of unknowns, and b is the right-hand side vector.  
Matrix A is considered sparse if most of its elements are zero. Storing a dense matrix requires O(n²) memory and leads to redundant computations on zero elements, which is inefficient for large sparse systems.

## Method choice justification

For sparse SLAEs, both direct and iterative methods are used.  
Direct methods (LU decomposition, Cholesky decomposition) suffer from fill-in: new non-zero elements appear during factorization, sharply increasing memory and time requirements and making them impractical for large highly sparse matrices.  

Iterative methods construct a sequence of approximations based on sparse matrix–vector products and vector operations, which is well suited for large systems and parallel architectures.  
This work uses the BiCGStab method, which is effective for general (including non-symmetric) sparse systems. The main operation is the matrix–vector product with complexity O(nnz), where nnz is the number of non-zero elements, which is significantly cheaper than O(n²) for dense matrices.  

The method is naturally parallelizable: the matrix–vector product is computed row-wise, vector operations are independent, and scalar products require only global reductions.

## BiCGStab algorithm (brief)

The inputs are matrix A, right-hand side vector b, initial guess x₀, and target residual tolerance ε.  
The algorithm iteratively updates the solution xₖ, residual rₖ, and auxiliary vectors (pₖ, vₖ, sₖ, tₖ), using two matrix–vector products per iteration, scalar products, and reductions.  

Iterations continue until the stopping criterion on the residual norm ||rₖ|| < ε is satisfied or a special situation occurs (for example, a very small value of ρₖ).  
The result is an approximate solution x that satisfies the prescribed tolerance.

## Experimental setup

All experiments use the same sparse tridiagonal matrix of size n.  
For each inner row, elements on the main diagonal are equal to 2, and elements on the sub- and superdiagonal are equal to −1.  

The right-hand side vector b is chosen as a vector of ones, and the initial guess x₀ is the zero vector.  
The stopping criterion is the residual norm tolerance ε = 1e−10.

*(Here you can insert summary plots of time/speedup for all implementations.)*

## C implementation with pthreads

The pthreads implementation uses shared memory and creates a thread pool, where the sparse matrix is split by rows among threads for the SpMV operation.  
Scalar products are computed via partial sums in threads with a reduction in the main thread, while vector updates are performed sequentially because they account for a small fraction of the total runtime.  

The experiments show speedup up to 16 threads. After that, further improvement stops and the real time starts to increase due to higher system overheads (synchronization, barriers) and pressure on the memory subsystem.  

*(Here you can insert a time/speedup vs. number of threads plot for the pthreads implementation.)*

## C implementation with MPI

The C/MPI implementation distributes matrix rows and corresponding vector elements across processes.  
Scalar products and norms are synchronized using collective operations, while the main parallelism comes from local SpMV on each process’s rows and parallel computation of local scalar products.  

The best scaling in terms of real execution time is achieved at about 8–16 processes.  
Further increasing the number of processes reduces the real-time gains, while system time grows sharply due to the cost of collective communication and the increased data exchange volume between processes.  

*(Here you can insert a scalability plot for the C/MPI implementation.)*

## Python implementation with mpi4py

The Python implementation with mpi4py follows the same scheme as C/MPI: the matrix and vectors are partitioned by rows among processes, local SpMV and scalar products are computed with NumPy, and global quantities and directions are assembled via collective operations.  

For the same number of processes, this implementation shows significantly larger real time than the C versions due to interpreter overhead and Python-level data structures.  
At the same time, the qualitative dependence of runtime on the number of processes follows the same general trend as in the C/MPI implementation.  

*(Here you can insert a runtime plot for the Python/mpi4py implementation.)*

## C implementation with OpenMP

The OpenMP implementation uses shared memory and parallelizes the main parts of the algorithm via compiler directives.  
SpMV, scalar products, and vector updates are implemented as parallel loops with reductions.  

The C/OpenMP implementation shows the best solution time among all variants at a moderate number of threads (around 8–16), providing significant speedup over the single-threaded run.  
System overheads are lower than in the pthreads and MPI variants. The anomaly at 48 threads is related to convergence behavior when the matrix is partitioned differently.  

*(Here you can insert a time/speedup plot for the OpenMP implementation.)*

## Conclusions

For the chosen problem (an SpMV-oriented BiCGStab method on a sparse tridiagonal matrix), the most efficient implementations are those using a shared-memory model, primarily OpenMP, at a moderate number of threads.  
Scaling to a large number of processes/threads is limited by synchronization costs, collective operations, and characteristics of the memory subsystem.
