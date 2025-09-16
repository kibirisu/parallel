#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <omp.h>

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

struct GraphData {
    int number;
    int order;
    Eigen::MatrixXd A;
};

int main() {
    std::string line;
    int graph_num = 0;
    int order = 0;
    std::vector<std::string> matrix_lines;
    std::vector<GraphData> graphs;

    // --- Parsowanie wejścia (sekwencyjne) ---
    while (std::getline(std::cin, line)) {
        if (line.rfind("Graph", 0) == 0) {
            if (!matrix_lines.empty()) {
                Eigen::MatrixXd A(order, order);
                for (int i = 0; i < order; i++)
                    for (int j = 0; j < order; j++)
                        A(i, j) = (matrix_lines[i][j] == '1') ? 1.0 : 0.0;
                graphs.push_back({graph_num, order, A});
                matrix_lines.clear();
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
        graphs.push_back({graph_num, order, A});
    }

    // --- Równoległe sprawdzanie integralności ---
    std::vector<std::string> results(graphs.size());

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)graphs.size(); i++) {
        bool integral = is_integral_graph(graphs[i].A);
        std::ostringstream oss;
        oss << "Graph " << graphs[i].number << ": "
            << (integral ? "Integral ✅" : "Not integral ❌");
        results[i] = oss.str();
    }

    // --- Wypisywanie wyników (sekwencyjnie w kolejności) ---
    for (const auto& res : results) {
        std::cout << res << "\n";
    }

    return 0;
}
