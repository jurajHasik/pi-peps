#include "ctm-cluster-io.h"
#include "simple-update.h"
#include <chrono>

using namespace itensor;

int main( int argc, char *argv[] ) {
    // ########################################################################
    // Handle command-line arguments

    OP2S_TYPE arg_op2s_type;
    F_MPO3S arg_f_mpo3s;
    std::string arg_clusterFile, arg_ioEnvTag; 
    
    if( argc >= 4) { 
        // minimal: [executable name], arg_clusterFile, arg_auxEnvDim,
        //          arg_ctmIter
        arg_clusterFile = argv[1];
        arg_op2s_type   = toOP2S_TYPE(std::string(argv[2]));
        arg_f_mpo3s     = toF_MPO3S(std::string(argv[3]));
    } else {
        std::cout <<"Invalid amount of Agrs (< 4)"<< std::endl;
        exit(EXIT_FAILURE);
    }

    // read in 2 x 2 cluster data 
    Cluster cluster = readCluster(arg_clusterFile);
    std::cout << cluster; //DBG

    if ( (cluster.sizeN != 2) || (cluster.sizeM != 2) ) {
        std::cout <<"Input cluster is not 2x2"<< std::endl;
        exit(EXIT_FAILURE);
    }
    if ( cluster.siteIds.size() != 2 ) {
        std::cout <<"Input cluster does not contain 2 non-equivalent"
            <<" sites"<< std::endl;
        exit(EXIT_FAILURE);
    }

    // Define 2-site operators
    // NN-Heisenberg in external mag-field
    MPO_2site nnh;
    nnh = getMPO2s_NNH(4, 1.0, 1.0, 0.0);

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
    ITensor A = cluster.sites.at(cluster.cToS.at(std::make_pair(0,0)));
    ITensor B = cluster.sites.at(cluster.cToS.at(std::make_pair(1,0)));

    Index aIA = noprime(findtype( A, AUXLINK) );
    Index aIB = noprime(findtype( B, AUXLINK) );
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

    // Apply 2-site op along bond A--l2--B
    A = A*l1*l3*l4;
    B = B*l4*l1*l3;
    applyH_T1_L_T2(nnh, A, B, l2);
    A = A*l1I*l3I*l4I;
    B = B*l4I*l1I*l3I;

    for (int i=0; i<=aIA.m(); i++ ) {
        l2I.set(prime(aIA,2)(i), aIB(i), 1.0/l2.real(prime(aIA,2)(i), aIB(i)));
    }
    PrintData(l2I);

     // Apply 2-site op along bond B--l1--A
    A = A*l3*l2*l4;
    B = B*l2*l4*l3;
    applyH_T1_L_T2(nnh, B, A, l1);
    A = A*l3I*l2I*l4I;
    B = B*l2I*l4I*l3I;

    for (int i=0; i<=aIA.m(); i++ ) {
        l1I.set(prime(aIB,2)(i), aIA(i), 1.0/l1.real(prime(aIB,2)(i), aIA(i)));
    }
    PrintData(l1I);


    writeCluster("test_H123.in", cluster);
}