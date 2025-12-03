#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
#include <complex>
#include <armadillo>
#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <highfive/highfive.hpp>
#include <iomanip>
#include <iostream>
#include <istream>
#include <nlohmann/json.hpp>
#include <string>
#include <limits>
#include <sstream>
using namespace std;
using namespace arma;
using namespace HighFive;


const double EV_TO_AU = 1 / 27.211324570273;
const double AU_back_EV = 27.211324570273;        
const double PI = arma::datum::pi;


namespace fs = std::filesystem;
using json = nlohmann::json;


struct AtomParam {
    double Ip_plus_Ap_over_2;   //s orbital
    double Ip_plus_Ap_p_over_2; // porbital
    double beta;                
    int Zstar;
};

struct Primitive {
    double alpha;   
    double d; //coeff for s
};

using BasisSet = std::vector<Primitive>;

std::map<std::string, BasisSet> sto3g_s = {
    {"H", {{3.42525091,0.15432897},
           {0.62391373,0.53532814},
           {0.16885540,0.44463454}}},
    {"C", {{2.94124940,-0.09996723},
           {0.68348310, 0.39951283},
           {0.22228990, 0.70011547}}},
    {"N", {{3.78045590,-0.09996723},
           {0.87849660, 0.39951283},
           {0.28571440, 0.70011547}}},
    {"O", {{5.03315130,-0.09996723},
           {1.16959610, 0.39951283},
           {0.38038900, 0.70011547}}},
    {"F", {{6.46480320,-0.09996723},
           {1.50228120, 0.39951283},
           {0.48858850, 0.70011547}}}
};

// just no h
std::map<std::string, BasisSet> sto3g_p = {
    {"C", {{2.94124940,0.15591627},
           {0.68348310,0.60768372},
           {0.22228990,0.39195739}}},
    {"N", {{3.78045590,0.15591627},
           {0.87849660,0.60768372},
           {0.28571440,0.39195739}}},
    {"O", {{5.03315130,0.15591627},
           {1.16959610,0.60768372},
           {0.38038900,0.39195739}}},
    {"F", {{6.46480320,0.15591627},
           {1.50228120,0.60768372},
           {0.48858850,0.39195739}}}
};

std::vector<std::string> element_symbols;
std::vector<vec>  atom_positions;

std::map<std::string, AtomParam> cndo_parameters = {
    {"H", {7.176,0,-9.0 ,1}},
    {"C", {14.051,5.572 ,-21,4}},
    {"N", {19.316,7.275,-25,5}},
    {"O", {25.390,9.111,-31,6}},
    {"F", {32.272,11.080,-39,7}}
};

// Eq. 3.8
double calculate_T(double V_squared, double distance_squared) {
    return V_squared * distance_squared;
}


// Eq. 3.10
double calculate_V_squared(double sigma_A, double sigma_B) {
    return 1 / (sigma_A + sigma_B);
}

//3.11
double calculate_sigma(double alpha_k, double alpha_kp) {
    return 1 /(alpha_k + alpha_kp);
}

// 3.12
double calculate_UA(double sigma_A) {
     
    return pow(datum::pi * sigma_A, 1.5);
}

// 3.15
double calculate_primitive_nonzero(double UA, double UB, double sqrt_T, double distance)
{
    return UA *UB * erf(sqrt_T) / distance;
}

//3.16
double calculate_primitive_zero(double UA, double UB, double V_squared) {

    double V = sqrt(V_squared);
    return UA *UB * 2 * V / sqrt(PI);

}


double calculate_gamma(int atom_A, int atom_B)
{
    const vec& position_A = atom_positions[atom_A];
    const vec& position_B = atom_positions[atom_B];

    double distance_squared = dot(position_A - position_B, position_A - position_B);
    double distance  = sqrt(distance_squared);

    //using s oribtal
    const BasisSet& basis_A = sto3g_s[element_symbols[atom_A]];
    const BasisSet& basis_B = sto3g_s[element_symbols[atom_B]];

    double total = 0;

    for (const auto& prim_k  : basis_A) {    

        double norm_k = pow(2 * prim_k.alpha / PI, 0.75);

        for (const auto& prim_kp : basis_A) {  

            double norm_kp = pow(2 * prim_kp.alpha /PI, 0.75);
            for (const auto& prim_l  : basis_B) {  

                double norm_l = pow(2* prim_l.alpha /PI, 0.75);

                for (const auto& prim_lp : basis_B) { 
                    double norm_lp = pow(2* prim_lp.alpha /PI, 0.75);

                    double coeff_k = prim_k.d;
                    double coeff_kp = prim_kp.d;
                    double coeff_l = prim_l.d;
                    double coeff_lp = prim_lp.d;
                    double coeff_product =coeff_k * norm_k * coeff_kp *norm_kp * coeff_l *norm_l * coeff_lp *norm_lp;

                    double sigma_A = calculate_sigma(prim_k.alpha, prim_kp.alpha);
                    double sigma_B = calculate_sigma(prim_l.alpha, prim_lp.alpha);

                    double UA = calculate_UA(sigma_A);
                    double UB = calculate_UA(sigma_B);

                    double V_squared = calculate_V_squared(sigma_A, sigma_B);

                    double T = calculate_T(V_squared, distance_squared);
                    double primitive;

                    if (distance_squared < 1e-11) {  //is is to prevent explosion due to div by zero
                        primitive = calculate_primitive_zero(UA, UB, V_squared);
                    } 


                    else {
                        double sqrt_T = sqrt(T);
                        primitive = calculate_primitive_nonzero(UA, UB, sqrt_T, distance);
                    }

                    total += (coeff_product * primitive);
                }
            }
        }
    }

    return total *AU_back_EV; //convert to eV
}

// straight from my hmwk 3
struct Basis_components {
    arma::vec center; 
    int l;
    int m;
    int n;     
    vector<double> exponents; 
    vector<double> coeffs; 
    vector<double> norm_coeff;
    string sym;  
};

//closely from hmwk 3

void read_xyz(const fs::path& path) {
    ifstream f(path.string());
    if (!f) {
        cerr << "couldnt open XYZ file: " << path << endl;
        exit(1);
    }

    int num_atoms;
    if (!(f >> num_atoms)) {

        cerr << "couldnt read number of atoms" << endl;
        exit(1);
    }

    f.ignore(numeric_limits<streamsize>::max(), '\n'); 

    element_symbols.clear();
    atom_positions.clear();

    for (int i = 0; i < num_atoms; i++) {

        int atomic_number;
        double x_coord, y_coord, z_coord;

        if (!(f >> atomic_number >> x_coord >> y_coord >> z_coord)) {

            cerr << "cant read atom " << i+1 << " from XYZ file" << endl;
            exit(1);
        }
    
        f.ignore(numeric_limits<streamsize>::max(), '\n');

        string element;

        if (atomic_number == 1) element = "H";
        else if (atomic_number == 6) element = "C";
        else if (atomic_number == 7) element = "N";
        else if (atomic_number == 8) element = "O";
        else if (atomic_number == 9) element = "F";
        else {

            cerr << "No data for this atom:  " << atomic_number << endl;
            exit(1);
        }

        element_symbols.push_back(element);
        atom_positions.push_back({x_coord, y_coord, z_coord});
    }
}

//basically taken from my hmwk 3 implementation , copy pasted with additions
vector<Basis_components> build_basis() {
    vector<Basis_components> basis_functions;

    for (int i = 0; i < element_symbols.size(); i++) {
        string element = element_symbols[i];
        vec center = atom_positions[i];
        if (element == "H") { 
            
            Basis_components bf;
            bf.center = center;
            bf.l = 0; 
            bf.m = 0;
            bf.n = 0;
            bf.sym = element;

            for (const auto& prim : sto3g_s["H"]) {
                bf.exponents.push_back(prim.alpha);
                bf.coeffs.push_back(prim.d);
            }

            basis_functions.push_back(bf);
        } 
        
        else { 
            // s orbital
            Basis_components bf_s;
            bf_s.center = center;
            bf_s.l = 0; 
            bf_s.m = 0; 
            bf_s.n = 0;
            bf_s.sym = element;

            for (const auto& prim : sto3g_s[element]) {
                bf_s.exponents.push_back(prim.alpha);
                bf_s.coeffs.push_back(prim.d);
            }

            basis_functions.push_back(bf_s);

            // p orbitals
            for (int comp = 0; comp < 3; comp++) {

                Basis_components bf_p;
                bf_p.center = center;
                bf_p.l = 0; 
                bf_p.m = 0; 
                bf_p.n = 0;
                bf_p.sym = element;

                if (comp == 0) { 
                    bf_p.l = 1; 
                } 
                else if (comp == 1) { 
                    bf_p.m = 1; 
                } 
                else { 
                    bf_p.n = 1; 
                } 

                for (const auto& prim : sto3g_p[element]) {
                    bf_p.exponents.push_back(prim.alpha);
                    bf_p.coeffs.push_back(prim.d);
                }
                basis_functions.push_back(bf_p);
            }
        }
    }

    return basis_functions;
}


//vector to know atom of each atomic oribital
vector<int> ao_to_atom;

void build_ao_to_atom(const vector<string>& symbols) {

    ao_to_atom.clear();

    for (int A = 0; A < symbols.size(); A++) {
        int num_orbitals;

        if (symbols[A] == "H") {
            num_orbitals = 1;
        } 
        else {
            num_orbitals = 4;
        }

        for (int i = 0; i < num_orbitals; i++) {
            ao_to_atom.push_back(A);
        }
    }
}



mat build_fock(const mat& total_density, const mat& spin_density, const mat& overlap_matrix, const vector<Basis_components>& basis_functions) {
    
    int num_basis = overlap_matrix.n_rows;
    mat fock(num_basis, num_basis, fill::zeros);

    vector<double> pAA(element_symbols.size(), 0.0);

    //pAA for each atom 

    int mu_global = 0;
    for (int A = 0; A < element_symbols.size(); A++) {
        int num_orbitals;

        if (element_symbols[A] == "H") {
            num_orbitals = 1;
        } 

        else {
            num_orbitals = 4;
        }

        for (int i = 0; i < num_orbitals; i++) {
            pAA[A] += total_density(mu_global, mu_global);
            mu_global += 1;
        }
    }

    for (int mu = 0; mu < num_basis; mu++) {

        int A = ao_to_atom[mu];

        const auto& param = cndo_parameters[element_symbols[A]];
        const auto& bf = basis_functions[mu];
        int lmn_sum =bf.l + bf.m +bf.n;
        double IplusA;

        if (lmn_sum == 0) {
            IplusA = param.Ip_plus_Ap_over_2;
        } 

        else {
            IplusA = param.Ip_plus_Ap_p_over_2;
        }

        double beta_A = param.beta;
        int Z = param.Zstar;

        for (int nu = 0; nu < num_basis; nu++) {
            int B = ao_to_atom[nu];
            double beta_B = cndo_parameters[element_symbols[B]].beta;

            if (mu == nu) {
                //eqation 1.4
                double term1 = - IplusA;
                double term2 = (pAA[A] - Z) * calculate_gamma(A, A);
                double term3 = - (spin_density(mu, mu) - 0.5) * calculate_gamma(A, A);
                double term4 = 0;

                for (int C = 0; C < element_symbols.size(); C++) {

                    if (C != A) {
                        term4 += (pAA[C] - cndo_parameters[element_symbols[C]].Zstar) *calculate_gamma(A, C);
                    }
                }

                fock(mu, nu) = term1 + term2 + term3 +term4;
            } 

            else {

                //eq. 1.5
                double beta_term = 0.5 * (beta_A + beta_B) * overlap_matrix(mu, nu);
                double coul_term = - spin_density(mu, nu) * calculate_gamma(A, B);
                fock(mu, nu) = beta_term + coul_term;
            }
        }
    }
    return fock;
}



mat make_core_hamiltonian(const mat& overlap_matrix, const std::vector<Basis_components>& basis_functions)
{
    const int num_basis = static_cast<int>(overlap_matrix.n_rows);

    mat core_ham(num_basis, num_basis, fill::zeros);
    std::vector<double> gamma_AA(element_symbols.size());

    for (int A = 0; A < static_cast<int>(element_symbols.size()); A++) {

        gamma_AA[A] = calculate_gamma(A, A);         
    }

    //loop over every atomic orbital
    for (int mu = 0; mu < num_basis; mu++){

        const int atom_A = ao_to_atom[mu];                     
        const std::string& sym_A = element_symbols[atom_A];
        const AtomParam& param_A = cndo_parameters.at(sym_A);
        const Basis_components& bf_mu = basis_functions[mu];

        //choose val based on oribtal type
        double half_IplusA;
        const int lmn = bf_mu.l + bf_mu.m +bf_mu.n;

        if (lmn == 0) {                     
            //s orbital
            half_IplusA = param_A.Ip_plus_Ap_over_2;
        } 

        else {                            
            // p orbital 
            half_IplusA = param_A.Ip_plus_Ap_p_over_2;
        }

        const int Z_A = param_A.Zstar;

        double diag = -half_IplusA;      

        diag -= (Z_A - 0.5) *gamma_AA[atom_A];

        
        for (int C = 0; C < static_cast<int>(element_symbols.size()); C++) {

            if (C == atom_A){

                continue;
            }

            const int Z_C = cndo_parameters.at(element_symbols[C]).Zstar;
            diag -= Z_C * calculate_gamma(atom_A, C);
        }

        core_ham(mu, mu) = diag;

        // offdiagonal elements eq. 2.7

        for (int nu = 0; nu < num_basis; nu++) {


            if (mu == nu) {

                continue;
            }             
            const int atom_B = ao_to_atom[nu];
            const double beta_A = param_A.beta;            
            const double beta_B = cndo_parameters.at(element_symbols[atom_B]).beta;

            core_ham(mu, nu) = 0.5 *(beta_A + beta_B) * overlap_matrix(mu, nu);
        }
    }

    return core_ham;
}


int main(int argc, char *argv[]) {

  // Check args
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " path/to/config" << std::endl;
    return EXIT_FAILURE;
  }

  // parse the config file
  fs::path config_file_path(argv[1]);
  if (!fs::exists(config_file_path)) {
    std::cerr << "Path: " << config_file_path << " does not exist" << std::endl;
    return EXIT_FAILURE;
  }

  std::ifstream config_file(config_file_path);
  json config = json::parse(config_file);

  fs::path atoms_file_path = config["atoms_file_path"];
  fs::path output_file_path = config["output_file_path"];
  int num_alpha = config["num_alpha_electrons"];
  int num_beta = config["num_beta_electrons"];




  //NOTE HERE!!

  // Adding a bunch of my lambda funcs from hmwk 3 (just copy pasting them thats why theyre still lambdas not methods)
  // on prev hmwks you asked why I didnt just use methods
  //Honestly thought we could only code within the lines that said  "your code here" which were within main
  //and homeworks really build on eachother so i just kept copypasting useful lambdas into main to reuse
  //and it worked so I stuck with it


  auto factorial = [](int number) -> double {
        if (number < 0) {
            return 0;
        }
        if (number == 0) {
            return 1;
        }
        if (number == 1) {
            return 1;
        }
        double result = 1;
        for (int count = 2; count <= number; count++) {
            result = result* count;
        }
        return result;
    };

    auto two_factorial = [](int number) -> double {
        if (number < 0) {
            if (number % 2 == -1) {
                return 1;
            }
            else {
                return 0;
            }
        }
        if (number == 0) {
            return 1;
        }
        if (number == 1) {
            return 1;
        }
        double result = 1;
        if (number % 2 == 0) {
            for (int count = 2; count <= number; count += 2) {
                result = result* count;
            }
        }
        else {
            for (int count = 1; count <= number; count += 2) {
                result = result* count;
            }
        }
        return result;
    };


    auto binomial = [&factorial](int total_items, int items_to_choose) -> double {
        if (items_to_choose < 0) {
            return 0;
        }
        if (items_to_choose > total_items) {
            return 0;
        }
        double numerator = factorial(total_items);
        double denominator = factorial(items_to_choose) * factorial(total_items - items_to_choose);
        if (denominator == 0) {
            return 0;
        }
        return numerator / denominator;
    };

    auto calc_overlap = [&binomial, &two_factorial](double x1, double y1, double z1, double exponent1, int power_x1, int power_y1, int power_z1, double x2, double y2, double z2, double exponent2, int power_x2, int power_y2, int power_z2) -> double {
        //this func will compute the gaussian product center along a given axis
        auto get_reduced_position = [&x1, &x2, &y1, &y2, &z1, &z2, &exponent1, &exponent2](char dimension) -> double {
            double coordinate_sum = 0;
            double denominator = exponent1 +exponent2;
            if (dimension == 'x') {
                coordinate_sum = exponent1*x1 + exponent2*x2;
            }
            else if (dimension == 'y') {
                coordinate_sum = exponent1*y1 + exponent2*y2;
            }
            else {
                coordinate_sum = exponent1*z1 + exponent2*z2;
            }
            if (denominator == 0) {
                return 0;
            }
            return coordinate_sum /denominator;
        };

        //this ones self explanatory - it will calcualte the exp term along a given axis
        auto calculate_exponential_term = [&x1, &x2, &y1, &y2, &z1, &z2, &exponent1, &exponent2](char dimension) -> double {
            double difference = 0;
            double denominator = exponent1 +exponent2;
            if (denominator == 0.0 || exponent1 <= 0.0 || exponent2 <= 0.0) {
                return 0.0; 
            }
            if (dimension == 'x') {
                difference = x1 - x2;
            }
            else if (dimension == 'y') {
                difference = y1 - y2;
            }
            else {
                difference = z1 - z2;
            }
            if (denominator == 0) {
                return 0;
            }
            double term = -(exponent1*exponent2 * difference*difference) /denominator;
            return exp(term);
        };

        //sum for a given axis term calculator
        auto calculate_double_sum_term = [&binomial, &two_factorial, &x1, &x2, &y1, &y2, &z1, &z2, &exponent1, &exponent2] (int power_a, int power_b, int index_i, int index_j, char dimension, double reduced_position) -> double {
            double exponent_sum = exponent1 + exponent2;

            double binomial_coefficient_a = binomial(power_a, index_i);
            double binomial_coefficient_b = binomial(power_b, index_j);
            double double_factorial = two_factorial(index_i + index_j - 1);
            double polynomial_a = 0;
            double polynomial_b = 0;

            if (dimension == 'x') {
                polynomial_a = pow(reduced_position - x1, power_a - index_i);
                polynomial_b = pow(reduced_position - x2, power_b - index_j);
            }

            else if (dimension == 'y') {
                polynomial_a = pow(reduced_position - y1, power_a - index_i);
                polynomial_b = pow(reduced_position - y2, power_b - index_j);
            }

            else {
                polynomial_a = pow(reduced_position - z1, power_a - index_i);
                polynomial_b = pow(reduced_position - z2, power_b - index_j);
            }
            double denominator = pow(2 * exponent_sum, (index_i + index_j) / 2);

            double result = (binomial_coefficient_a *binomial_coefficient_b *double_factorial * polynomial_a * polynomial_b) / denominator;
            return result;
        };

        // full double sum over i and j for given axis
        auto calculate_double_sum = [&calculate_double_sum_term, &get_reduced_position](int power_a, int power_b, char dimension) -> double {
            double total_sum = 0;

            for (int index_i = 0; index_i <= power_a; index_i++) {

                for (int index_j = 0; index_j <= power_b; index_j++) {

                    if ((index_i + index_j) % 2 != 0) {
                        continue;
                    }

                    double reduced_position = get_reduced_position(dimension);
                    total_sum += calculate_double_sum_term(power_a, power_b, index_i, index_j, dimension, reduced_position);
                }
            }
            return total_sum;
        };

        //overlap term for a given axis
        auto calculate_overlap_term = [&calculate_exponential_term, &calculate_double_sum, &exponent1, &exponent2](char dimension, int power_a, int power_b) -> double {
            double denominator = exponent1 +exponent2;

            double radical_term = sqrt(PI /denominator);

            double exponential_term = calculate_exponential_term(dimension);
            double sum_term = calculate_double_sum(power_a, power_b, dimension);
            double final_result = exponential_term *radical_term *sum_term;

            return final_result;
        };

        //full 3d overlap calc from 3 1d overlaps
        double overlap_x = calculate_overlap_term('x', power_x1, power_x2);
        double overlap_y = calculate_overlap_term('y', power_y1, power_y2);
        double overlap_z = calculate_overlap_term('z', power_z1, power_z2);

        double result = overlap_x *overlap_y *overlap_z;

        return result;
    };


    auto normalize_basis = [&calc_overlap](vector<Basis_components>& basis_functions) {

        for (auto& bf : basis_functions) {
            bf.norm_coeff.clear();

            for (int k = 0; k < static_cast<int>(bf.exponents.size()); k++) {

                double x = bf.center(0);
                double y = bf.center(1);
                double z = bf.center(2);
                double exp = bf.exponents[k];

                int lx = bf.l;
                int ly = bf.m;
                int lz = bf.n;
                double overlap_aa = calc_overlap(x, y, z, exp, lx, ly, lz, x, y, z, exp, lx, ly, lz);

                double norm = 1/ sqrt(overlap_aa);

                bf.norm_coeff.push_back(norm);

            }
        }
    };

    //build_S
    auto build_S = [&calc_overlap](const vector<Basis_components>& basis_functions) -> mat {

        int num_basis = basis_functions.size();
        mat overlap_matrix(num_basis, num_basis, fill::zeros);

        for (int mu = 0; mu < num_basis; mu++) {

            for (int nu = 0; nu < num_basis; nu++) {

                double sum = 0;
                for (int k = 0; k < 3; k++) { 

                    for (int l = 0; l < 3; l++) { 
                        double x1 = basis_functions[mu].center(0);
                        double y1 = basis_functions[mu].center(1);
                        double z1 = basis_functions[mu].center(2);

                        double x2 = basis_functions[nu].center(0);
                        double y2 = basis_functions[nu].center(1);
                        double z2 = basis_functions[nu].center(2);


                        double exp1 = basis_functions[mu].exponents[k];
                        double exp2 = basis_functions[nu].exponents[l];

                        int power_x1 = basis_functions[mu].l;
                        int power_y1 = basis_functions[mu].m;
                        int power_z1 = basis_functions[mu].n;
                        int power_x2 = basis_functions[nu].l;
                        int power_y2 = basis_functions[nu].m;
                        int power_z2 = basis_functions[nu].n;
                        double coeff1 = basis_functions[mu].coeffs[k];
                        double norm1 = basis_functions[mu].norm_coeff[k];

                        double coeff2 = basis_functions[nu].coeffs[l];
                        double norm2 = basis_functions[nu].norm_coeff[l];

                        double overlap_kl = calc_overlap(x1, y1, z1, exp1, power_x1, power_y1, power_z1, x2, y2, z2, exp2, power_x2, power_y2, power_z2);
                        sum += coeff1 *coeff2 * norm1 * norm2 *overlap_kl;
                    }
                }
                overlap_matrix(mu, nu) = sum;
            }
        }
        return overlap_matrix;
    };





    // Main execution block here and below
    read_xyz(atoms_file_path);

    cout << "Symbols: " << element_symbols[0] << " " << element_symbols[1] << endl;
    cout << "p: " << num_alpha << " q: " << num_beta << " total: " << (num_alpha + num_beta) << endl;
    // build basis_functions
    vector<Basis_components> basis_functions = build_basis();

    normalize_basis(basis_functions);

    //building s 
    mat overlap_matrix = build_S(basis_functions);

    // build ao_to_atom
    build_ao_to_atom(element_symbols);
    

    int num_basis = basis_functions.size();
    mat gamma_matrix(element_symbols.size(), element_symbols.size());

    for (int a = 0; a < element_symbols.size(); a++) {

        for (int b = 0; b < element_symbols.size(); b++) {

            gamma_matrix(a, b) = calculate_gamma(a, b);
        
        }
    }

    cout << "gamma" << endl;
    cout << gamma_matrix << endl;

    mat Palpha(num_basis, num_basis, fill::zeros);
    mat Pbeta(num_basis, num_basis, fill::zeros);
    mat Ptot = Palpha + Pbeta;

    mat Falpha = build_fock(Ptot, Palpha, overlap_matrix, basis_functions);
    mat Fbeta  = build_fock(Ptot, Pbeta , overlap_matrix, basis_functions);

    cout << fixed << setprecision(6);
    mat core_ham = make_core_hamiltonian(overlap_matrix, basis_functions);
    const double zero_cutoff = 1e-10;

    for (int i = 0; i < num_basis; i++) {

        for (int j = 0; j < num_basis; j++) {

            double s_abs = std::abs(overlap_matrix(i, j));

            if (s_abs < zero_cutoff) {
                overlap_matrix(i, j) = 0;
            }

            double h_abs = std::abs(core_ham(i, j));

            if (h_abs < zero_cutoff) {
                core_ham(i, j) = 0;
            }
        }
    }

    cout << "Overlap" << endl;
    cout << overlap_matrix << endl;

    cout << "H_core" << endl;
    cout << core_ham << endl;


    const int max_iterations = 100;
    const double convergence_threshold = 1e-6;
    int iteration_count = 0;
    bool is_converged = false;


    mat Falpha_initial(num_basis, num_basis, fill::zeros);
    mat Fbeta_initial(num_basis, num_basis, fill::zeros);
  

    while ((!is_converged) && (iteration_count < max_iterations)) {

        //build the fock matrices

        mat Falpha = build_fock(Ptot, Palpha, overlap_matrix, basis_functions);
        mat Fbeta  = build_fock(Ptot, Pbeta , overlap_matrix, basis_functions);

        // just setting this for the autograder/pytest
         if (iteration_count == 0) {

            Falpha_initial = Falpha;
            Fbeta_initial  = Fbeta;

        }

        cout << "Iteration: " << iteration_count << endl;
        cout << "Fa" << endl;
        cout << Falpha << endl;

        cout << "Fb" << endl;
        cout << Fbeta << endl;

        //diagonalize Fock matrices falpha and fbeta
        arma::vec eigenvalues_alpha;
        arma::mat eigenvectors_alpha;
        arma::eig_sym(eigenvalues_alpha, eigenvectors_alpha, Falpha);

        arma::vec eigenvalues_beta;
        arma::mat eigenvectors_beta;
        arma::eig_sym(eigenvalues_beta, eigenvectors_beta, Fbeta);

        arma::uvec sorted_indices_alpha = arma::sort_index(eigenvalues_alpha);
        arma::uvec sorted_indices_beta  = arma::sort_index(eigenvalues_beta);

        arma::mat coefficients_alpha = eigenvectors_alpha.cols(sorted_indices_alpha);
        arma::mat coefficients_beta  = eigenvectors_beta.cols(sorted_indices_beta);

        cout << "after solving eigen equation: " << iteration_count << endl;

        cout << "Ca" << endl;
        cout << coefficients_alpha << endl;

        cout << "Cb" << endl;
        cout << coefficients_beta << endl;

        // new density matrices
        mat Palpha_new(num_basis, num_basis, fill::zeros);
        mat Pbeta_new (num_basis, num_basis, fill::zeros);

        for (int i = 0; i < num_alpha; i++) {

            arma::colvec orbital = coefficients_alpha.col(i);
            Palpha_new += orbital *orbital.t();
        }

        for (int i = 0; i < num_beta; i++) {

            arma::colvec orbital = coefficients_beta.col(i);
            Pbeta_new += orbital * orbital.t();

        }

        cout << " p = " << num_alpha << " q = " << num_beta << endl;
        cout << "Pa_new" << endl;
        cout << Palpha_new << endl;
        cout << "Pb_new" << endl;
        cout << Pbeta_new << endl;

        mat Ptot_new = Palpha_new + Pbeta_new;

        cout << "P_t" << endl
        ;
        for (int mu = 0; mu < num_basis; mu++) {

            cout << fixed << setprecision(4) << Ptot_new(mu, mu) << endl;
        }

        //this will check convergence
        double delta_alpha = arma::max(arma::abs(arma::vectorise(Palpha_new - Palpha)));
        double delta_beta  = arma::max(arma::abs(arma::vectorise(Pbeta_new - Pbeta)));
        double convergence_value = std::max(delta_alpha, delta_beta);

        if (convergence_value < convergence_threshold) {
            is_converged = true;
        }

        Palpha = Palpha_new;
        Pbeta= Pbeta_new;
        Ptot = Ptot_new;

        iteration_count++;
    }

   




    mat Falpha_final = build_fock(Ptot, Palpha, overlap_matrix, basis_functions);
    mat Fbeta_final  = build_fock(Ptot, Pbeta , overlap_matrix, basis_functions);

    vec final_Ea, final_Eb;
    mat final_Ca, final_Cb;

    eig_sym(final_Ea, final_Ca, Falpha_final);
    eig_sym(final_Eb, final_Cb, Fbeta_final);

    uvec ida = sort_index(final_Ea);
    uvec idb = sort_index(final_Eb);

    final_Ea = final_Ea(ida);
    final_Eb = final_Eb(idb);
    final_Ca = final_Ca.cols(ida);
    final_Cb = final_Cb.cols(idb);

    cout << fixed << setprecision(4);
    cout << "Ea" << endl;
    cout << final_Ea << endl;
    cout << "Eb" << endl;
    cout << final_Eb << endl;
    cout << "Ca" << endl;
    cout << final_Ca << endl;
    cout << "Cb" << endl;
    cout << final_Cb << endl;


   cout << "\n Final MO energies (eV) \n";

    cout << "Alpha (Ea):\n";
    cout << final_Ea.t();  //transpose so they print as a row

    cout << "Beta  (Eb):\n";
    cout << final_Eb.t() << endl;


    //HOMO/LUMO and band gap (

    if (num_alpha <= 0 || num_beta <= 0 ||
        num_alpha >= num_basis || num_beta >= num_basis) {
        cerr << "Cannot define HOMO/LUMO: check num_alpha/num_beta vs num_basis." << endl;
    } 
    else {
        int homo_alpha_idx = num_alpha - 1;
        int lumo_alpha_idx = num_alpha;
        int homo_beta_idx = num_beta - 1;
        int lumo_beta_idx = num_beta;

        double homo_alpha_E = final_Ea(homo_alpha_idx);
        double lumo_alpha_E = final_Ea(lumo_alpha_idx);
        double homo_beta_E = final_Eb(homo_beta_idx);
        double lumo_beta_E = final_Eb(lumo_beta_idx);

        double homo_E = std::max(homo_alpha_E, homo_beta_E);
        double lumo_E = std::min(lumo_alpha_E, lumo_beta_E);
        double band_gap_eV = lumo_E - homo_E;

        cout << "\nHOMO/LUMO energies (eV) \n";
        cout << "HOMO_alpha (index " << homo_alpha_idx << "): " << homo_alpha_E << " eV\n";
        cout << "LUMO_alpha (index " << lumo_alpha_idx << "): " << lumo_alpha_E << " eV\n";
        cout << "HOMO_beta  (index " << homo_beta_idx << "): " << homo_beta_E << " eV\n";
        cout << "LUMO_beta  (index " << lumo_beta_idx << "): " << lumo_beta_E << " eV\n";
        cout << "Overall HOMO: " << homo_E << " eV\n";
        cout << "Overall LUMO: " << lumo_E << " eV\n";
        cout << "HOMO-LUMO gap: " << band_gap_eV << " eV\n\n";
    }
    // === end HOMO/LUMO block ===

    // elec energy
    double electronic_energy_eV = 0;
    for (int mu = 0; mu < num_basis; mu++) {

        for (int nu = 0; nu < num_basis; nu++) {

            double h  = core_ham(mu, nu);
            double Fa = Falpha_final(mu, nu);
            double Fb = Fbeta_final(mu, nu);

            electronic_energy_eV += 0.5* (Palpha(mu, nu)*(h + Fa) +Pbeta (mu, nu) *(h + Fb));
        }
    }


    //repulsion energy calc
    double nuclear_energy_au = 0;

    for (int A = 0; A < element_symbols.size(); A++) {

        for (int B = 0; B < A; B++) {

            double ZA = cndo_parameters[element_symbols[A]].Zstar;
            double ZB = cndo_parameters[element_symbols[B]].Zstar;

            vec RA = atom_positions[A];
            vec RB = atom_positions[B];

            double distance_AB = norm(RA - RB);   
            nuclear_energy_au += ZA *ZB /distance_AB;
        }
    }

    double nuclear_energy_eV = nuclear_energy_au * AU_back_EV;

    double total_energy_eV = electronic_energy_eV + nuclear_energy_eV;


    cout << "Nuclear Repulsion Energy is " << nuclear_energy_eV << " eV." << endl;
    cout << "Electron Energy is " << electronic_energy_eV << " eV." << endl;
    cout << "The molecule in file " << atoms_file_path << " has energy " << total_energy_eV << endl;

    // check that output dir exists
    if (!fs::exists(output_file_path.parent_path())){
        fs::create_directories(output_file_path.parent_path());
    }
 
    // delete the file if it does exist (so that no old answers stay there by accident)
    if (fs::exists(output_file_path)){
        fs::remove(output_file_path);
    }

    HighFive::File file(output_file_path.string(), HighFive::File::Create);

    //helper fucs to put everything in the hdf5's in the correct formats
    auto write_matrix = [&](const std::string &name, const mat &M) {
        int nr = M.n_rows;
        int nc = M.n_cols;

        std::vector<std::vector<double> > data(nr, std::vector<double>(nc));

        for (int i = 0; i < nr; i++) {
            for (int j = 0; j < nc; j++) {
                data[i][j] = M(i, j);
            }
        }

        file.createDataSet(name, data);
    };
   
    auto write_vector = [&](const std::string &name, const vec &v) {
        int n = v.n_elem;

        std::vector<std::vector<double> > data(n, std::vector<double>(1));

        for (int i = 0; i < n; i++) {
            data[i][0] = v(i);
        }

        file.createDataSet(name, data);
    };

    auto write_scalar = [&](const std::string &name, double value) {

        file.createDataSet(name, value);
    };



    write_matrix("S", overlap_matrix);
    write_matrix("gamma", gamma_matrix);
    write_matrix("H_core", core_ham);
    write_matrix("Fa_initial",Falpha_initial);
    write_matrix("Fb_initial",Fbeta_initial);
    write_vector("Ea", final_Ea);
    write_vector("Eb", final_Eb);
    write_scalar("electronic_energy", electronic_energy_eV);
    write_scalar("nuclear_energy", nuclear_energy_eV);
    write_scalar("total_energy", total_energy_eV);

    return EXIT_SUCCESS;
}