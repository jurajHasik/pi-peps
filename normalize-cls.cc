#include "ctm-cluster-io.h"

using namespace itensor;

int main( int argc, char *argv[] ) {
    std::cout.precision( std::numeric_limits< double >::max_digits10 );
    
    // ########################################################################
    // Handle command-line arguments

    std::string arg_inClusterFile, arg_outClusterFile, arg_normType; 
    
    if( argc >= 4) { 
        // minimal: [executable name], arg_clusterFile, arg_auxEnvDim,
        //          arg_ctmIter
        arg_inClusterFile  = argv[1];
        arg_outClusterFile = argv[2];
        arg_normType       = argv[3];
    } else {
        std::cout <<"Invalid amount of Agrs (< 4)"<< std::endl;
        exit(EXIT_FAILURE);
    }

    // read in n x m cluster data 
    Cluster cluster = readCluster(arg_inClusterFile);
    std::cout << cluster; //DBG

    if (arg_normType == "BLE") {
        double m;
        auto maxElem = [&m](double d) {
            m = (std::abs(d) > m) ? std::abs(d) : m;
        };

        for (auto & tentry : cluster.sites) {
            m = 0.0;
            tentry.second.visit(maxElem);
            tentry.second = tentry.second / m;
            std::cout<<"On-site tensor "<< tentry.first <<" max Elem: "<< m << std::endl;
        }
    } else {
        std::cout <<"Unsupported normalization: "<< arg_normType << std::endl;
        exit(EXIT_FAILURE);
    }

    writeCluster(arg_outClusterFile, cluster);

    return 0;
}
