#include <iostream>
#include <mutex>
#include <thread>

#include <eigen3/Eigen/Eigen>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

struct GraphData {
    int number;
    int order;
    Eigen::MatrixXd A; // host, row-major conversion when copying to device
};

bool passes_sieve(const Eigen::MatrixXd& A, int edges) {
    int n = A.rows();
    double tr = A.trace();
    if (std::fabs(tr) > 1e-9) return false;
    double tr2 = (A * A).trace();
    if (std::fabs(tr2 - 2.0 * edges) > 1e-6) return false;
    if (2 * edges > n * (n - 1) * (n - 1)) return false;
    return true;
}

bool is_integral_from_evals(const std::vector<double>& evals, double tol = 1e-6) {
    for (double v : evals) {
        double r = std::round(v);
        if (std::fabs(v - r) > tol) return false;
    }
    return true;
}

// GPU/cuSOLVER helper RAII
struct CusolverHandle {
    cusolverDnHandle_t handle;
    CusolverHandle() { cusolverDnCreate(&handle); }
    ~CusolverHandle() { cusolverDnDestroy(handle); }
};

struct CudaStream {
    cudaStream_t s;
    CudaStream() { cudaStreamCreate(&s); }
    ~CudaStream() { cudaStreamDestroy(s); }
};

// Compute eigenvalues for one symmetric matrix on GPU using cusolverDnDsyevd (one stream)
bool gpu_compute_eigenvalues_one(cusolverDnHandle_t handle, cudaStream_t stream,
                                 const double* h_A_colmajor, int n,
                                 std::vector<double>& out_evals) {
    // cuSOLVER expects column-major. Convert host row-major (Eigen is column-major by default,
    // but we will copy in column-major form below).
    // Allocate device matrix (n*n) in column-major
    double* d_A = nullptr;
    cudaError_t cerr = cudaMallocAsync(&d_A, sizeof(double) * n * n, stream);
    if (cerr != cudaSuccess) return false;

    // copy matrix to device (host data already column-major from Eigen)
    cudaMemcpyAsync(d_A, h_A_colmajor, sizeof(double) * n * n, cudaMemcpyHostToDevice, stream);

    // workspace query
    int lwork = 0;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR; // only eigenvalues
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER; // we will pass upper triangle
    // Note: cusolverDnDsyevd uses column-major and can accept full symmetric matrix (only triangle used).
    cusolverStatus_t status = cusolverDnDsyevd_bufferSize(
        handle, jobz, uplo, n, d_A, n, nullptr, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        cudaFreeAsync(d_A, stream);
        return false;
    }

    double* d_work = nullptr;
    cudaMallocAsync(&d_work, sizeof(double) * lwork, stream);
    int* devInfo = nullptr;
    cudaMallocAsync(&devInfo, sizeof(int), stream);

    // allocate output eigenvalues on device
    double* d_W = nullptr;
    cudaMallocAsync(&d_W, sizeof(double) * n, stream);

    // cusolver requires a host workspace for eigenvalues? Using API that writes eigenvalues to host
    // but Dsyevd writes eigenvalues into a host array; however the API variant we'll call is:
    status = cusolverDnDsyevd(
        handle, jobz, uplo, n, d_A, n, d_W, d_work, lwork, devInfo);

    // synchronize this stream before reading results
    cudaError_t cerr2 = cudaStreamSynchronize(stream);
    if (status != CUSOLVER_STATUS_SUCCESS || cerr2 != cudaSuccess) {
        if (d_A) cudaFreeAsync(d_A, stream);
        if (d_work) cudaFreeAsync(d_work, stream);
        if (devInfo) cudaFreeAsync(devInfo, stream);
        if (d_W) cudaFreeAsync(d_W, stream);
        return false;
    }

    int h_devInfo = 0;
    cudaMemcpyAsync(&h_devInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (h_devInfo != 0) {
        if (d_A) cudaFreeAsync(d_A, stream);
        if (d_work) cudaFreeAsync(d_work, stream);
        if (devInfo) cudaFreeAsync(devInfo, stream);
        if (d_W) cudaFreeAsync(d_W, stream);
        return false;
    }

    out_evals.resize(n);
    cudaMemcpyAsync(out_evals.data(), d_W, sizeof(double) * n, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // free
    if (d_A) cudaFreeAsync(d_A, stream);
    if (d_work) cudaFreeAsync(d_work, stream);
    if (devInfo) cudaFreeAsync(devInfo, stream);
    if (d_W) cudaFreeAsync(d_W, stream);

    return true;
}

const int BATCH_SIZE = 256; // liczba kandydatów przesyłanych do GPU jednocześnie
const int MAX_STREAMS = 16;

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string line;
    int graph_num = 0;
    int order = 0;
    std::vector<std::string> matrix_lines;
    std::vector<GraphData> batch;
    batch.reserve(BATCH_SIZE);

    // przygotowanie CUDA/cuSOLVER
    CusolverHandle cusolver;
    std::vector<std::unique_ptr<CudaStream>> streams;
    int n_streams = std::min(MAX_STREAMS, std::max(1, (int)std::thread::hardware_concurrency()));
    for (int i = 0; i < n_streams; ++i) streams.emplace_back(new CudaStream());
    // mutex do synchronicznego wypisywania
    std::mutex out_mtx;

    auto flush_cpu_batch = [&](std::vector<GraphData>& cpu_batch) {
        if (cpu_batch.empty()) return;
        // First: apply sieve on CPU and prepare list of candidates for GPU
        struct Candidate { int idx; int number; int n; Eigen::MatrixXd A; int edges; };
        std::vector<Candidate> candidates;
        candidates.reserve(cpu_batch.size());
        for (int i = 0; i < (int)cpu_batch.size(); ++i) {
            int n = cpu_batch[i].order;
            int edges = (int)(cpu_batch[i].A.sum() / 2.0);
            bool integral = false;
            if (passes_sieve(cpu_batch[i].A, edges)) {
                // push as candidate
                candidates.push_back({i, cpu_batch[i].number, n, cpu_batch[i].A, edges});
            } else {
                std::lock_guard<std::mutex> lk(out_mtx);
                std::cout << "Graph " << cpu_batch[i].number << ": Not integral ❌\n";
            }
        }

        if (candidates.empty()) {
            cpu_batch.clear();
            return;
        }

        // Process candidates in sub-batches to limit concurrent GPU calls
        for (size_t start = 0; start < candidates.size(); start += BATCH_SIZE) {
            size_t end = std::min(candidates.size(), start + BATCH_SIZE);
            size_t chunk = end - start;

            // For each candidate in chunk, create a thread that submits work to one of the CUDA streams.
            // This provides concurrency across many small matrices.
            std::vector<std::thread> workers;
            workers.reserve(chunk);
            for (size_t k = 0; k < chunk; ++k) {
                size_t idx = start + k;
                Candidate &c = candidates[idx];

                workers.emplace_back([&cusolver, &streams, &out_mtx, c]() {
                    // choose stream by simple hash on graph number
                    int sidx = c.number % streams.size();
                    cudaStream_t s = streams[sidx]->s;

                    // Ensure matrix is column-major when sending to cuSOLVER (Eigen is column-major by default)
                    // We'll use pointer to contiguous data in column-major layout:
                    Eigen::MatrixXd Acol = c.A; // copy (A is column-major by default)
                    // Allocate host buffer contiguous (already contiguous)
                    const double* h_A = Acol.data();

                    std::vector<double> evals;
                    bool ok = gpu_compute_eigenvalues_one(cusolver.handle, s, h_A, c.n, evals);

                    bool integral = false;
                    if (ok) integral = is_integral_from_evals(evals);

                    std::lock_guard<std::mutex> lk(out_mtx);
                    if (integral) {
                        std::cout << "Graph " << c.number << ": Integral ✅\n";
                    } else {
                        std::cout << "Graph " << c.number << ": Not integral ❌\n";
                    }
                });
            }

            // join workers
            for (auto &t : workers) t.join();
        }

        cpu_batch.clear();
    };

    // read input (format jak w twoim kodzie)
    while (std::getline(std::cin, line)) {
        if (line.rfind("Graph", 0) == 0) {
            if (!matrix_lines.empty()) {
                Eigen::MatrixXd A(order, order);
                for (int i = 0; i < order; i++)
                    for (int j = 0; j < order; j++)
                        A(i, j) = (matrix_lines[i][j] == '1') ? 1.0 : 0.0;
                batch.push_back({graph_num, order, A});
                matrix_lines.clear();

                if ((int)batch.size() >= BATCH_SIZE) {
                    flush_cpu_batch(batch);
                }
            }
            graph_num++;
            std::istringstream iss(line);
            std::string word;
            iss >> word; // "Graph"
            iss >> word; // numer
            iss >> word; // "order"
            iss >> order;
        } else if (!line.empty() && (line[0] == '0' || line[0] == '1')) {
            matrix_lines.push_back(line);
        }
    }

    if (!matrix_lines.empty()) {
        Eigen::MatrixXd A(order, order);
        for (int i = 0; i < order; i++)
            for (int j = 0; j < order; j++)
                A(i, j) = (matrix_lines[i][j] == '1') ? 1.0 : 0.0;
        batch.push_back({graph_num, order, A});
    }

    flush_cpu_batch(batch);

    // synchronize streams before exit
    for (auto &s : streams) cudaStreamSynchronize(s->s);
    return 0;
}
