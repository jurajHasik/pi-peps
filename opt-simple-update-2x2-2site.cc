#include "ctm-cluster-io.h"
#include "simple-update.h"
#include <chrono>

using namespace itensor;

int main( int argc, char *argv[] ) {
    // ########################################################################
    // Handle command-line arguments

    //OP2S_TYPE arg_op2s_type;
    //F_MPO3S arg_f_mpo3s;
    std::string arg_outClusterFile;
    int arg_auxBondDim, arg_nIter;
    double arg_tau, arg_J, arg_h;
    
    if( argc >= 7) { 
        // minimal: [executable name], arg_outClusterFile, arg_auxBondDim,
        //          arg_nIter
        arg_outClusterFile = argv[1];
        arg_auxBondDim     = std::stoi(argv[2]);
        arg_nIter          = std::stoi(argv[3]);
        arg_tau            = std::stod(argv[4]);
        arg_J              = std::stod(argv[5]);
        arg_h              = std::stod(argv[6]);
        //arg_op2s_type   = toOP2S_TYPE(std::string(argv[2]));
        //arg_f_mpo3s     = toF_MPO3S(std::string(argv[3]));
    } else {
        std::cout <<"Invalid amount of Agrs (< 7)"<< std::endl;
        exit(EXIT_FAILURE);
    }

    // Define cluster
    Cluster cls = Cluster();
    cls.sizeN = 2;
    cls.sizeM = 2;
    cls.auxBondDim = arg_auxBondDim;
    cls.physDim    = 2;

    cls.siteIds = std::vector< std::string >(4);
    cls.siteIds = { "A", "B", "C", "D" };

    cls.cToS  = {
        {std::make_pair(0,0),"A"},
        {std::make_pair(1,0),"B"},
        {std::make_pair(0,1),"C"},
        {std::make_pair(1,1),"D"}
    };

    Index aIA(TAG_I_AUX, cls.auxBondDim, AUXLINK);
    Index aIB(TAG_I_AUX, cls.auxBondDim, AUXLINK);
    Index pIA(TAG_I_PHYS, cls.physDim, PHYS);
    Index pIB(TAG_I_PHYS, cls.physDim, PHYS);

    ITensor A(aIA, prime(aIA,1), prime(aIA,2), prime(aIA,3), pIA);
    ITensor B(aIB, prime(aIB,1), prime(aIB,2), prime(aIB,3), pIB);

    // Randomize
    randomize(A);
    randomize(B);

    cls.sites = {{"A", A}, {"B", B}, {"C",B}, {"D",A}};

    std::cout << cls; //DBG

    // Define 2-site operators
    // NN-Heisenberg in external mag-field
    MPO_2site nnh;
    // coordination, tau, J, h
    nnh = getMPO2s_NNH(4, arg_tau, arg_J, arg_h);

    // Prepare Sites (from cluster) + weights to begin optimization 
    // procedure
    // We assume we have cluster of the form   AB
    //                                         BA
    // Adding the weights results in the following tensor network
    //
    //      l3     l4               The site aux indices are as follows
    //       |      | 
    // --l1--A--l2--B--l1--                    auxI1 
    //       |      |                           | 
    //      l4     l3                   auxI0-- A --auxI2
    //       |      |                           | 
    // --l2--B--l1--A--l2--                    auxI3
    //       |      | 
    //      l3     l4

    ITensor l1(prime(aIB,2), aIA);
    ITensor l2(prime(aIA,2), aIB);
    ITensor l3(prime(aIB,3), prime(aIA,1)); 
    ITensor l4(prime(aIA,3), prime(aIB,1));

    // Initially set all weights to 1
    for (int i=1; i<=aIA.m(); i++) {
        l1.set(prime(aIB,2)(i), aIA(i), 1.0);
        l2.set(prime(aIA,2)(i), aIB(i), 1.0);
        l3.set(prime(aIB,3)(i), prime(aIA,1)(i), 1.0);
        l4.set(prime(aIA,3)(i), prime(aIB,1)(i), 1.0);
    }
    Print(l1);
    Print(l2);
    Print(l3);
    Print(l4);

    // Define set of inverse diag weights
    auto l1I = l1;
    auto l2I = l2;
    auto l3I = l3;
    auto l4I = l4;    

    std::chrono::steady_clock::time_point t_iso_begin, t_iso_end;
    t_iso_begin = std::chrono::steady_clock::now();
    for (int nStep=1; nStep<=arg_nIter; nStep++) {
        

        // Apply 2-site op along bond A--l2--B
        A = A*l1*l3*l4;
        B = B*l4*l1*l3;
        applyH_T1_L_T2(nnh, A, B, l2);
        A = A*l1I*l3I*l4I;
        B = B*l4I*l1I*l3I;

        for (int i=1; i<=aIA.m(); i++ ) {
            l2I.set(prime(aIA,2)(i), aIB(i), 1.0/l2.real(prime(aIA,2)(i), aIB(i)));
        }
        // PrintData(l2I);
        // std::cout << "##### APPLIED nnh along A--l2--B #####" << std::endl;
        // Print(A);
        // Print(l2);
        // Print(B);

        // Apply 2-site op along bond B--l1--A
        A = A*l3*l2*l4;
        B = B*l2*l4*l3;
        applyH_T1_L_T2(nnh, B, A, l1);
        A = A*l3I*l2I*l4I;
        B = B*l2I*l4I*l3I;

        for (int i=1; i<=aIA.m(); i++ ) {
            l1I.set(prime(aIB,2)(i), aIA(i), 1.0/l1.real(prime(aIB,2)(i), aIA(i)));
        }
        // PrintData(l1I);
        // std::cout << "##### APPLIED nnh along B--l1--A #####" << std::endl;
        // Print(B);
        // Print(l1);
        // Print(A);

        // Apply 2-site op along bond A--l4--B
        A = A*l1*l3*l2;
        B = B*l1*l3*l2;
        applyH_T1_L_T2(nnh, A, B, l4);
        A = A*l1I*l3I*l2I;
        B = B*l1I*l3I*l2I;

        for (int i=1; i<=aIA.m(); i++ ) {
            l4I.set(prime(aIA,3)(i), prime(aIB,1)(i), 
                1.0/l4.real(prime(aIA,3)(i), prime(aIB,1)(i)));
        }
        // PrintData(l4I);
        // std::cout << "##### APPLIED nnh along A--l4--B #####" << std::endl;
        // Print(A);
        // Print(l4);
        // Print(B);

        // Apply 2-site op along bond B--l3--A
        A = A*l2*l4*l1;
        B = B*l2*l4*l1;
        applyH_T1_L_T2(nnh, B, A, l3);
        A = A*l2I*l4I*l1I;
        B = B*l2I*l4I*l1I;

        for (int i=1; i<=aIA.m(); i++ ) {
            l3I.set(prime(aIB,3)(i), prime(aIA,1)(i), 
                1.0/l3.real(prime(aIB,3)(i), prime(aIA,1)(i)));
        }
        // PrintData(l3I);
        // std::cout << "##### APPLIED nnh along B--l3--A #####" << std::endl;
        // Print(B);
        // Print(l3);
        // Print(A);
    
        if ( nStep % 1000 == 0 ) { 
            t_iso_end = std::chrono::steady_clock::now();
            std::cout <<"STEP "<< nStep <<" T= "<< std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0 
            << std::endl;
            PrintData(l1);
            PrintData(l2);
            PrintData(l3);
            PrintData(l4);
            t_iso_begin = std::chrono::steady_clock::now();
        }
    }

    // Set new sites to cluster
    auto sqrtT = [](double r) { return sqrt(r); };

    l1.apply(sqrtT);
    l2.apply(sqrtT);
    l3.apply(sqrtT);
    l4.apply(sqrtT);

    A = A * l1 * l2 * l3 * l4;
    A = swapPrime(A, 0, 2);
    A = swapPrime(A, 1, 3);

    B = B * l1 * l2 * l3 * l4;
    B = swapPrime(B, 0, 2);
    B = swapPrime(B, 1, 3);

    // Build new cluster
    cls.sites = {{"A", A}, {"B", B}, {"C",B}, {"D",A}};
    std::cout << cls; //DBG

    writeCluster(arg_outClusterFile, cls);
}