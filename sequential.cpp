#include <iostream>
#include <vector>
#include <eigen3/Eigen/Eigen>

bool is_integral_graph(const Eigen::MatrixXd& A, double tol = 1e-6) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Błąd: Eigen solver failed\n";
        return false;
    }
    Eigen::VectorXd ev = solver.eigenvalues();
    for (int i = 0; i < ev.size(); i++) {
        double r = std::round(ev[i]);
        if (std::fabs(ev[i] - r) > tol) return false;
    }
    return true;
}

int main() {
    std::string line;
    int graph_num = 0;
    int order = 0;
    std::vector<std::string> matrix_lines;

    while (std::getline(std::cin, line)) {
        if (line.rfind("Graph", 0) == 0) {
            // linia "Graph X, order N."
            if (!matrix_lines.empty()) {
                // przetwarzamy poprzedni graf
                Eigen::MatrixXd A(order, order);
                for (int i = 0; i < order; i++) {
                    for (int j = 0; j < order; j++) {
                        A(i, j) = (matrix_lines[i][j] == '1') ? 1.0 : 0.0;
                    }
                }
                bool integral = is_integral_graph(A);
                std::cout << "Graph " << graph_num << ": "
                          << (integral ? "Integral ✅" : "Not integral ❌") << "\n";
                matrix_lines.clear();
            }

            // parsuj numer i rozmiar
            graph_num++;
            std::istringstream iss(line);
            std::string word;
            iss >> word; // "Graph"
            iss >> word; // numer z przecinkiem np. "1,"
            iss >> word; // "order"
            iss >> order;
        } else if (!line.empty() && (line[0] == '0' || line[0] == '1')) {
            matrix_lines.push_back(line);
        }
    }

    // ostatni graf (po EOF)
    if (!matrix_lines.empty()) {
        Eigen::MatrixXd A(order, order);
        for (int i = 0; i < order; i++) {
            for (int j = 0; j < order; j++) {
                A(i, j) = (matrix_lines[i][j] == '1') ? 1.0 : 0.0;
            }
        }
        bool integral = is_integral_graph(A);
        std::cout << "Graph " << graph_num << ": "
                  << (integral ? "Integral ✅" : "Not integral ❌") << "\n";
    }

    return 0;
}
