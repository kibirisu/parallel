#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <eigen3/Eigen/Dense>

// ---- Parser: graph6 -> adjacency (dla n <= 62) ----
Eigen::MatrixXd graph6_to_adjacency(const std::string& line) {
    if (line.empty()) throw std::invalid_argument("empty line");
    unsigned char c0 = static_cast<unsigned char>(line[0]);
    if (c0 == ':') throw std::invalid_argument("sparse6 not supported");
    int n = static_cast<int>(c0) - 63;
    if (n < 0) throw std::invalid_argument("invalid graph6 header");

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    int needed_bits = n*(n-1)/2;
    int pos = 0;
    for (size_t i = 1; i < line.size() && pos < needed_bits; ++i) {
        unsigned char b = static_cast<unsigned char>(line[i]) - 63;
        for (int bit = 5; bit >= 0 && pos < needed_bits; --bit) {
            int k = pos;
            int row = 0;
            while (k >= n - row - 1) { k -= (n - row - 1); ++row; }
            int col = row + 1 + k;
            if ((b >> bit) & 1) {
                A(row, col) = 1.0;
                A(col, row) = 1.0;
            }
            ++pos;
        }
    }
    if (pos < needed_bits) {
        throw std::invalid_argument("not enough bits in graph6 line");
    }
    return A;
}

// ---- Eigen-based checker ----
bool is_integral_graph(const Eigen::MatrixXd& A, double tol = 1e-6) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Eigen solver failed\n";
        return false;
    }
    Eigen::VectorXd ev = solver.eigenvalues();
    for (int i = 0; i < ev.size(); ++i) {
        double r = std::round(ev[i]);
        if (std::fabs(ev[i] - r) > tol) return false;
    }
    return true;
}

// ---- procesuj strumień wejściowy (stdin albo plik) ----
int process_stream(std::istream& in, int verboseN) {
    std::string line;
    long long total = 0;
    long long found = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '>' || line[0] == '#') continue;
        auto pos_space = line.find_first_of(" \t");
        std::string token = (pos_space == std::string::npos) ? line : line.substr(0, pos_space);
        if (token[0] == ':') continue; // skip sparse6

        ++total;
        try {
            Eigen::MatrixXd A = graph6_to_adjacency(token);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);
            if (solver.info() != Eigen::Success) continue;
            Eigen::VectorXd ev = solver.eigenvalues();
            bool integral = true;
            for (int i = 0; i < ev.size(); ++i) {
                double r = std::round(ev[i]);
                if (std::fabs(ev[i] - r) > 1e-6) { integral = false; break; }
            }

            if (integral) {
                std::cout << token << '\n';
                ++found;
            }

            // 🔹 tryb verbose dla pierwszych N grafów
            if (total <= verboseN) {
                std::cerr << "Graph #" << total << "  " << token << "\n";
                std::cerr << "Adjacency matrix:\n" << A << "\n";
                std::cerr << "Eigenvalues: " << ev.transpose() << "\n";
                std::cerr << "Integral? " << (integral ? "YES ✅" : "NO ❌") << "\n\n";
            }

        } catch (std::exception& e) {
            std::cerr << "parse error: " << e.what() << "\n";
            continue;
        }
    }
    std::cerr << "Processed: " << total << " graphs; Found integral: " << found << "\n";
    return 0;
}

int main(int argc, char** argv) {
    int verboseN = 0;
    int argi = 1;

    // parsuj argumenty
    if (argc > 1 && std::string(argv[1]) == "-v") {
        if (argc > 2) verboseN = std::stoi(argv[2]);
        else verboseN = 5; // domyślnie 5
        argi = 3;
    }

    if (argc <= argi || (argc == argi+1 && std::string(argv[argi]) == "-")) {
        return process_stream(std::cin, verboseN);
    } else {
        std::ifstream fin(argv[argi]);
        if (!fin.is_open()) {
            std::cerr << "Nie można otworzyć pliku: " << argv[argi] << '\n';
            return 1;
        }
        return process_stream(fin, verboseN);
    }
}
