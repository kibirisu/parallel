// main.cu
// Batched Jacobi eigensolver (GPU-only) for many small symmetric matrices (adjacency matrices).
// Host-side sieve is applied. Only supports n <= MAX_N (default 32).
// Compile: nvcc -O3 main.cu -o integral_batched -std=c++14
// Run: feed with same input format as showg output (Graph i, order n. followed by n lines of 0/1)

#include <bits/stdc++.h>
#include <cuda_runtime.h>

constexpr int MAX_N = 32;              // max supported matrix order
constexpr int THREADS_PER_BLOCK = 128; // threads per CUDA block (tune)
constexpr int DEFAULT_BATCH = 4096;    // default number of matrices per host->device batch
constexpr double TOL = 1e-6;
constexpr int MAX_ITER = 128;

// ---------- Host helpers ----------
struct HostMat {
    int number;
    int n;
    std::vector<double> data; // column-major n*n
};

inline bool passes_sieve_host(const std::vector<double>& A, int n, int edges) {
    // trace
    double tr = 0.0;
    for (int i = 0; i < n; ++i) tr += A[i*n + i];
    if (fabs(tr) > 1e-9) return false;
    // trace(A^2)
    double tr2 = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            tr2 += A[i*n + j] * A[j*n + i];
    if (fabs(tr2 - 2.0 * edges) > 1e-6) return false;
    if (2 * edges > n * (n - 1) * (n - 1)) return false;
    return true;
}

// ---------- Device kernels ----------
// We'll store each matrix in device global memory packed: d_A[mat_idx * n*n + i*n + j] column-major (col-major => index = col*n + row)
__device__ inline int idx_elem(int mat_idx, int n, int col, int row, int pitch) {
    // pitch = n*n
    return mat_idx * pitch + col * n + row;
}

// For Jacobi rotations we use block-per-matrix. Each block handles one matrix.
// Shared memory layout: sA[col * n + row] (col-major)
__global__ void batched_jacobi_kernel(const double* __restrict__ d_A_all, int n, int pitch, int batch_count, int max_iter, double tol, unsigned char* __restrict__ d_out_flags) {
    extern __shared__ double sA[]; // size n*n per block (dynamic shared)
    int mat_idx = blockIdx.x;
    if (mat_idx >= batch_count) return;

    int tid = threadIdx.x;
    // each block loads matrix into shared memory collaboratively
    int total_elems = n * n;
    for (int k = tid; k < total_elems; k += blockDim.x) {
        int col = k / n;
        int row = k % n;
        sA[col * n + row] = d_A_all[idx_elem(mat_idx, n, col, row, pitch)];
    }
    __syncthreads();

    // Jacobi cyclic sweeps (simple implementation): iterate over all p<q pairs in natural order
    // We'll parallelize operations across threads by letting each thread handle updates to a row/col
    for (int iter = 0; iter < max_iter; ++iter) {
        double max_off = 0.0;
        // for all pairs p<q
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                // fetch a_pp, a_qq, a_pq
                double app = sA[p * n + p];
                double aqq = sA[q * n + q];
                double apq = sA[q * n + p]; // column q, row p -> sA[q*n + p]

                double abs_apq = fabs(apq);
                if (abs_apq > max_off) max_off = abs_apq;

                // compute rotation if needed
                double phi = 0.0;
                if (fabs(apq) > 1e-15) {
                    double tau = (aqq - app) / (2.0 * apq);
                    double t = (tau >= 0.0) ? (1.0 / (tau + sqrt(1.0 + tau * tau))) : (-1.0 / (-tau + sqrt(1.0 + tau * tau)));
                    double c = 1.0 / sqrt(1.0 + t * t);
                    double s = t * c;
                    // apply rotation to rows/cols p and q
                    // threads update elements for rows r = 0..n-1
                    for (int r = tid; r < n; r += blockDim.x) {
                        if (r != p && r != q) {
                            double arp = sA[p * n + r];
                            double arq = sA[q * n + r];
                            // since symmetric, sA[col*n + row], update both symmetric positions
                            double new_arp = c * arp - s * arq;
                            double new_arq = s * arp + c * arq;
                            sA[p * n + r] = new_arp;
                            sA[r * n + p] = new_arp; // symmetric
                            sA[q * n + r] = new_arq;
                            sA[r * n + q] = new_arq;
                        }
                    }
                    if (tid == 0) {
                        double new_app = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                        double new_aqq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                        sA[p * n + p] = new_app;
                        sA[q * n + q] = new_aqq;
                        sA[q * n + p] = 0.0;
                        sA[p * n + q] = 0.0;
                    }
                    __syncthreads();
                }
            } // q
        } // p
        // check convergence: compute max off-diagonal in this sweep (simple reduction by thread 0)
        // We'll compute approximate max by letting thread 0 scan (cheap for small n)
        __syncthreads();
        if (tid == 0) {
            double local_max = 0.0;
            for (int c = 0; c < n; ++c)
                for (int r = 0; r < n; ++r)
                    if (r != c) {
                        double v = fabs(sA[c * n + r]);
                        if (v > local_max) local_max = v;
                    }
            max_off = local_max;
            if (max_off <= tol) {
                // write eigenvalues check result
                bool integral = true;
                for (int i = 0; i < n; ++i) {
                    double eig = sA[i * n + i];
                    double rr = round(eig);
                    if (fabs(eig - rr) > max(1e-8, tol)) { integral = false; break; }
                }
                d_out_flags[mat_idx] = integral ? 1 : 0;
                return;
            }
        }
        __syncthreads();
        if (tid == 0 && iter == max_iter - 1) {
            // last iteration and not converged: still extract diagonals and test
            bool integral = true;
            for (int i = 0; i < n; ++i) {
                double eig = sA[i * n + i];
                double rr = round(eig);
                if (fabs(eig - rr) > max(1e-8, tol)) { integral = false; break; }
            }
            d_out_flags[mat_idx] = integral ? 1 : 0;
            return;
        }
        __syncthreads();
    } // iter
    // fallback
    if (threadIdx.x == 0) d_out_flags[mat_idx] = 0;
}

// Host convenience: check CUDA errors
inline void cuda_check(cudaError_t e, const char* msg=nullptr) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error";
        if (msg) std::cerr << " (" << msg << ")";
        std::cerr << ": " << cudaGetErrorString(e) << "\n";
        exit(1);
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Read graphs in original format
    std::string line;
    int graph_num = 0;
    int order = 0;
    std::vector<std::string> matlines;
    std::vector<HostMat> host_batch;
    host_batch.reserve(DEFAULT_BATCH);

    auto flush_and_process = [&](std::vector<HostMat>& batch) {
        if (batch.empty()) return;
        int batch_count = (int)batch.size();
        int n = batch[0].n;
        int pitch = n * n;
        // pack matrices contiguous (column-major)
        std::vector<double> h_all((size_t)batch_count * pitch);
        for (int m = 0; m < batch_count; ++m) {
            // already column-major
            std::memcpy(&h_all[(size_t)m * pitch], batch[m].data.data(), sizeof(double) * pitch);
        }
        // allocate device
        double* d_all = nullptr;
        unsigned char* d_out = nullptr;
        cuda_check(cudaMalloc(&d_all, sizeof(double) * h_all.size()), "alloc d_all");
        cuda_check(cudaMemcpy(d_all, h_all.data(), sizeof(double) * h_all.size(), cudaMemcpyHostToDevice), "h2d block");
        cuda_check(cudaMalloc(&d_out, batch_count * sizeof(unsigned char)), "alloc d_out");
        cuda_check(cudaMemset(d_out, 0, batch_count * sizeof(unsigned char)), "memset out");

        // launch one block per matrix, shared mem = n*n*sizeof(double)
        int blocks = batch_count;
        int threads = min(THREADS_PER_BLOCK, n); // at least n threads to update rows; but keep >1
        if (threads < 32) threads = 32;
        size_t shared_bytes = sizeof(double) * pitch;
        if (shared_bytes > 48 * 1024) {
            std::cerr << "Error: required shared memory too large for n=" << n << " (bytes=" << shared_bytes << ")\n";
            cudaFree(d_all); cudaFree(d_out);
            return;
        }
        batched_jacobi_kernel<<<blocks, threads, shared_bytes>>>(d_all, n, pitch, batch_count, MAX_ITER, TOL, d_out);
        cuda_check(cudaGetLastError(), "launch jacobi");
        cuda_check(cudaDeviceSynchronize(), "synchronize jacobi");

        // copy back results
        std::vector<unsigned char> h_out(batch_count);
        cuda_check(cudaMemcpy(h_out.data(), d_out, batch_count * sizeof(unsigned char), cudaMemcpyDeviceToHost), "d2h out");

        // print results
        for (int m = 0; m < batch_count; ++m) {
            if (h_out[m]) std::cout << "Graph " << batch[m].number << ": Integral ✅\n";
            else std::cout << "Graph " << batch[m].number << ": Not integral ❌\n";
        }

        cudaFree(d_all);
        cudaFree(d_out);
        batch.clear();
    };

    while (std::getline(std::cin, line)) {
        if (line.rfind("Graph", 0) == 0) {
            if (!matlines.empty()) {
                // build HostMat
                HostMat hm;
                hm.number = graph_num;
                hm.n = order;
                hm.data.assign(order * order, 0.0);
                // fill column-major: element (row i, col j) -> index col*n + row
                for (int i = 0; i < order; ++i) {
                    for (int j = 0; j < order; ++j) {
                        char c = matlines[i][j];
                        double v = (c == '1') ? 1.0 : 0.0;
                        hm.data[j * order + i] = v;
                    }
                }
                int edges = 0;
                for (int i = 0; i < order; ++i)
                    for (int j = i+1; j < order; ++j)
                        if (hm.data[j*order + i] != 0.0) edges++;
                if (hm.n <= MAX_N && passes_sieve_host(hm.data, hm.n, edges)) {
                    host_batch.push_back(std::move(hm));
                } else {
                    // print not integral immediately
                    std::cout << "Graph " << hm.number << ": Not integral ❌\n";
                }
                matlines.clear();
                if ((int)host_batch.size() >= DEFAULT_BATCH) {
                    flush_and_process(host_batch);
                }
            }
            graph_num++;
            std::istringstream iss(line);
            std::string w;
            iss >> w; // Graph
            iss >> w; // num,
            iss >> w; // order
            iss >> order;
        } else if (!line.empty() && (line[0] == '0' || line[0] == '1')) {
            matlines.push_back(line);
        }
    }
    if (!matlines.empty()) {
        HostMat hm;
        hm.number = graph_num;
        hm.n = order;
        hm.data.assign(order * order, 0.0);
        for (int i = 0; i < order; ++i)
            for (int j = 0; j < order; ++j) {
                char c = matlines[i][j];
                double v = (c == '1') ? 1.0 : 0.0;
                hm.data[j * order + i] = v;
            }
        int edges = 0;
        for (int i = 0; i < order; ++i)
            for (int j = i+1; j < order; ++j)
                if (hm.data[j*order + i] != 0.0) edges++;
        if (hm.n <= MAX_N && passes_sieve_host(hm.data, hm.n, edges)) host_batch.push_back(std::move(hm));
        else std::cout << "Graph " << hm.number << ": Not integral ❌\n";
    }
    if (!host_batch.empty()) flush_and_process(host_batch);
    return 0;
}
