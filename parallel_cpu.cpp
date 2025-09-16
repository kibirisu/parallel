#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <omp.h>

struct GraphData {
    int number;
    int order;
    Eigen::MatrixXd A;
};

bool passes_sieve(const Eigen::MatrixXd& A, int edges) {
    int n = A.rows();

    // Warunek 1: trace(A) = 0
    double tr = A.trace();
    if (std::fabs(tr) > 1e-9) return false;

    // Warunek 2: trace(A^2) == 2e
    double tr2 = (A * A).trace();
    if (std::fabs(tr2 - 2.0 * edges) > 1e-6) return false;

    // Warunek 3: suma kwadratów wartości własnych <= n*(n-1)^2
    if (2 * edges > n * (n - 1) * (n - 1)) return false;

    return true; // przechodzi sito → kandydat na całkowity
}

bool is_integral_graph(const Eigen::MatrixXd& A, double tol = 1e-6) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);
    if (solver.info() != Eigen::Success) return false;
    Eigen::VectorXd ev = solver.eigenvalues();
    for (int i = 0; i < ev.size(); i++) {
        double r = std::round(ev[i]);
        if (std::fabs(ev[i] - r) > tol) return false;
    }
    return true;
}

const int BATCH_SIZE = 1000;

void process_batch(std::vector<GraphData> &batch) {
    if (batch.empty()) return;
    std::vector<std::string> results(batch.size());

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)batch.size(); i++) {
        int n = batch[i].order;
        int edges = (int)(batch[i].A.sum() / 2); // suma/2 = liczba krawędzi

        bool integral = false;
        if (passes_sieve(batch[i].A, edges)) {
            integral = is_integral_graph(batch[i].A);
        }

        std::ostringstream oss;
        oss << "Graph " << batch[i].number << ": "
            << (integral ? "Integral ✅" : "Not integral ❌");
        results[i] = oss.str();
    }

    for (auto &r : results) {
        std::cout << r << "\n";
    }
    batch.clear();
}

int main() {
    std::string line;
    int graph_num = 0;
    int order = 0;
    std::vector<std::string> matrix_lines;
    std::vector<GraphData> batch;
    batch.reserve(BATCH_SIZE);

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
                    process_batch(batch);
                }
            }
            graph_num++;
            std::istringstream iss(line);
            std::string word;
            iss >> word; // "Graph"
            iss >> word; // numer z przecinkiem
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

    process_batch(batch);
    return 0;
}
