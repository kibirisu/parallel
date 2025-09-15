#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>
#include <eigen3/Eigen/Eigen>

using namespace Eigen;

// Funkcja sprawdzająca czy liczby własne są całkowite
bool is_integral(const VectorXd& eigvals, double tol = 1e-6) {
    for (int i = 0; i < eigvals.size(); i++) {
        double rounded = round(eigvals[i]);
        if (abs(eigvals[i] - rounded) > tol) {
            return false;
        }
    }
    return true;
}

// 🔸 Parser graph6 → macierz sąsiedztwa
MatrixXd graph6_to_adjacency(const std::string& line) {
    if (line.empty()) {
        throw std::invalid_argument("Empty graph6 line");
    }

    unsigned char c = line[0];
    int n = (int)c - 63;  // liczba wierzchołków (działa dla n ≤ 62)

    MatrixXd A = MatrixXd::Zero(n, n);

    int needed_bits = n * (n - 1) / 2;
    int pos = 0;
    for (size_t i = 1; i < line.size(); i++) {
        unsigned char byte = line[i] - 63;
        for (int bit = 5; bit >= 0; bit--) {
            if (pos >= needed_bits) break;
            int k = pos;
            int row = 0;
            while (k >= n - row - 1) {
                k -= (n - row - 1);
                row++;
            }
            int col = row + 1 + k;

            if ((byte >> bit) & 1) {
                A(row, col) = 1;
                A(col, row) = 1;
            }
            pos++;
        }
    }
    return A;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Użycie: " << argv[0] << " plik.g6" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Nie można otworzyć pliku: " << filename << std::endl;
        return 1;
    }

    std::vector<std::string> graphs;
    std::string line;
    while (getline(infile, line)) {
        if (!line.empty()) {
            graphs.push_back(line);
        }
    }
    infile.close();

    int integral_count = 0;

    #pragma omp parallel for reduction(+:integral_count)
    for (size_t i = 0; i < graphs.size(); i++) {
        try {
            MatrixXd A = graph6_to_adjacency(graphs[i]);
            SelfAdjointEigenSolver<MatrixXd> solver(A);
            std::cout << "Eigenvalues: " << solver.eigenvalues().transpose() << std::endl;
            if (is_integral(solver.eigenvalues())) {
                #pragma omp critical
                {
                    std::cout << graphs[i] << std::endl; // wypisz integralne grafy w formacie g6
                }
                integral_count++;
            }
        } catch (std::exception& e) {
            #pragma omg critical
            std::cerr << "Błąd parsowania grafu: " << e.what() << std::endl;
        }
    }

    std::cerr << "Znaleziono " << integral_count 
         << " grafów całkowitych / " << graphs.size() << std::endl;

    return 0;
}
