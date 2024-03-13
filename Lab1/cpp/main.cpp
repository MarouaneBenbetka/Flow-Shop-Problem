#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
using namespace std;

class BranchAndBound
{
private:
    std::vector<std::vector<int>> dist_mat;
    int M;
    std::vector<int> ordre_opt;
    std::vector<std::vector<int>> c;

public:
    BranchAndBound(std::vector<std::vector<int>> dist_mat) : dist_mat(dist_mat)
    {
        M = INT_MAX;
        c.resize(dist_mat.size(), std::vector<int>(dist_mat[0].size(), 0));
    }

    void update_c(std::vector<std::vector<int>> &c, std::vector<int> &ordre)
    {
        if (ordre.size() == 1)
        {
            c[0][0] = dist_mat[0][ordre[0] - 1];
            for (int j = 1; j < dist_mat[0].size(); j++)
            {
                c[0][j] = dist_mat[j][ordre[0] - 1] + c[0][j - 1];
            }
        }
        else
        {
            int niveau = ordre.size() - 1;
            c[niveau][0] = c[niveau - 1][0] + dist_mat[0][ordre[niveau] - 1];
            for (int j = 1; j < dist_mat[0].size(); j++)
            {
                c[niveau][j] = std::max(c[niveau - 1][j], c[niveau][j - 1]) + dist_mat[j][ordre[niveau] - 1];
            }
        }
    }

    int get_c_max(std::vector<std::vector<int>> &c, int niveau)
    {
        return c[niveau][dist_mat[0].size() - 1];
    }

    void printOptimalSolution()
    {
        std::cout << "Optimal Order: ";
        for (size_t i = 0; i < ordre_opt.size(); i++)
        {
            std::cout << ordre_opt[i];
            if (i < ordre_opt.size() - 1)
            {
                std::cout << " -> ";
            }
        }
        std::cout << std::endl;
        std::cout << "Optimal Cost: " << M << std::endl;
    }

    int evaluate(std::vector<std::vector<int>> &C, std::vector<int> &ordre, std::vector<int> &tache_restante)
    {
        int lb_max = INT_MIN;
        int niveau = ordre.size() - 1;
        for (int j = 0; j < dist_mat[0].size(); j++)
        {
            int lb = C[niveau][j];
            int min_temps_exec_derniere_tache = INT_MAX;
            for (int tache : tache_restante)
            {
                lb += dist_mat[j][tache - 1];
                int temps_exec_derniere_tache = 0;
                for (int j_prime = j + 1; j_prime < dist_mat[0].size(); j_prime++)
                {
                    temps_exec_derniere_tache += dist_mat[j_prime][tache - 1];
                }
                min_temps_exec_derniere_tache = std::min(min_temps_exec_derniere_tache, temps_exec_derniere_tache);
            }
            lb += min_temps_exec_derniere_tache;

            if (lb > lb_max)
            {
                lb_max = lb;
            }
        }
        return lb_max;
    }

    void iterate(std::vector<int> ordre, std::vector<int> tache_restante, std::vector<std::vector<int>> &C)
    {
        for (int i : tache_restante)
        {
            // init
            std::vector<int> new_ordre = ordre;
            std::vector<int> new_tache_restante = tache_restante;
            // update
            new_ordre.push_back(i);
            new_tache_restante.erase(std::remove(new_tache_restante.begin(), new_tache_restante.end(), i), new_tache_restante.end());
            std::vector<std::vector<int>> C_new = C;
            update_c(C_new, new_ordre);
            //
            if (new_tache_restante.empty())
            {
                int c_max = get_c_max(C_new, new_ordre.size() - 1);
                if (c_max < M)
                {
                    M = c_max;
                    ordre_opt = new_ordre;
                    c = C_new;
                }
            }
            else
            {
                int evaluation = evaluate(C_new, new_ordre, new_tache_restante);
                // elagage
                if (evaluation < M)
                {
                    iterate(new_ordre, new_tache_restante, C_new);
                }
            }
        }
    }

    void run()
    {
        std::vector<int> ordre;
        std::vector<int> tache_rest(dist_mat[0].size());
        for (size_t i = 0; i < tache_rest.size(); i++)
        {
            tache_rest[i] = i + 1;
        }
        iterate(ordre, tache_rest, c);
    }
};

int main()
{
    std::vector<std::vector<int>> dist_mat = {
        {0, 2, 9, 10},
        {1, 0, 6, 4},
        {15, 7, 0, 8},
        {6, 3, 12, 0}
    };

    std::vector<std::vector<int>> dist_mat22 = {
        {54, 83, 15, 71, 77, 36, 53, 38, 27, 87, 76, 91, 14, 29, 12, 77, 32, 87, 68, 94},
        {79, 3, 11, 99, 56, 70, 99, 60, 5, 56, 3, 61, 73, 75, 47, 14, 21, 86, 5, 77},
        {16, 89, 49, 15, 89, 45, 60, 23, 57, 64, 7, 1, 63, 41, 63, 47, 26, 75, 77, 40},
        {66, 58, 31, 68, 78, 91, 13, 59, 49, 85, 85, 9, 39, 41, 56, 40, 54, 77, 51, 31},
        {58, 56, 20, 85, 53, 35, 53, 41, 69, 13, 86, 72, 8, 49, 47, 87, 58, 18, 68, 28}

    };

    BranchAndBound solver(dist_mat22);
    solver.run();
    solver.printOptimalSolution();

    return 0;
}
