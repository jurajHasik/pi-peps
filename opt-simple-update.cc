#include "ctm-cluster-io.h"
#include "simple-update.h"
#include <chrono>

using namespace itensor;

int main( int argc, char *argv[] ) {
    // ########################################################################
    // Handle command-line arguments

    int arg_auxEnvDim, arg_ctmIter;
    std::string arg_clusterFile, arg_ioEnvTag; 
    
    if( argc >= 4) { 
        // minimal: [executable name], arg_clusterFile, arg_auxEnvDim,
        //          arg_ctmIter
        arg_clusterFile = argv[1];
        arg_auxEnvDim   = stoi(std::string(argv[2]));
        arg_ctmIter     = stoi(std::string(argv[3]));
    } else {
        std::cout <<"Invalid amount of Agrs (< 4)"<< std::endl;
        exit(EXIT_FAILURE);
    }

    // read in n x m cluster data 
    Cluster cluster = readCluster(arg_clusterFile);
    std::cout << cluster; //DBG

    MPO_3site mpo3s_Id = getMPO3s_Id(cluster.physDim);

    /*
     * Applying H_123 to sites ABD
     * 
     *      a1          a1
     *      |            |
     * a0--|A|--a2--a0--|B|--a2
     *      |            | 
     *      a3           a3
     *                   | 
     *                   a1
     *                   |
     *              a0--|D|--a2
     *                   | 
     *                   a3
     * 
     * we pass in as pairs (of indices) connecting the sites (A_a2,B_a0)
     * and (B_a3,C_a1) 
     * 
     */
    applyH_123(2.0*mpo3s_Id,
        cluster.sites["A"], cluster.sites["B"], cluster.sites["D"],
        std::make_pair( 
            noprime( findtype(cluster.sites["A"].inds(), AUXLINK)).prime(2),
            noprime( findtype(cluster.sites["B"].inds(), AUXLINK)).prime(0) ),
        std::make_pair( 
            noprime( findtype(cluster.sites["B"].inds(), AUXLINK)).prime(3),
            noprime( findtype(cluster.sites["D"].inds(), AUXLINK)).prime(1)));

    writeCluster("test_H123.in", cluster);
}