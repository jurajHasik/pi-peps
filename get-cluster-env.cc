#include "ctm-cluster-io.h"
#include "ctm-cluster.h"
#include "cluster-ev-builder.h"

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

    // Prepare blank CtmEnv
    CtmEnv ctmEnv;

    // Depending on the number of args call different environment 
    // initialization function
    switch (argc) {
        case 5: {
            // [executable name], arg_clusterFile, arg_auxEnvDim, arg_ctmIter,
            // env_init
            CtmEnv::init_env_type init_Env = toINIT_ENV(std::string(argv[4]));
            if ( init_Env == CtmEnv::INIT_ENV_const1 ) {
                ctmEnv = CtmEnv("TEST_ENV_2x2", arg_auxEnvDim, cluster);
                ctmEnv.initMockEnv();
            } else {
                std::cout<< argv[4] <<" requires its additional args" 
                    << std::endl;
                exit(EXIT_FAILURE);
            }
            break;
        }
        case 6: {
            // [executable name], arg_clusterFile, arg_auxEnvDim, arg_ctmIter,
            // env_init, boolean
            CtmEnv::init_env_type init_Env = toINIT_ENV(std::string(argv[4]));
            if ( init_Env == CtmEnv::INIT_ENV_rnd ) {
                
                bool isComplex;
                if (std::string(argv[5]) == "0") 
                    isComplex = false;
                else if (std::string(argv[5]) == "1")
                    isComplex = true;
                else {
                    std::cout<< "5th Arg must be 0 or 1" << std::endl;
                    exit(EXIT_FAILURE);
                }

                ctmEnv = CtmEnv("TEST_ENV_2x2", arg_auxEnvDim, cluster);
                ctmEnv.initRndEnv(isComplex);
            } else {
                std::cout<< argv[4] <<" requires its additional args"
                    << std::endl;
                exit(EXIT_FAILURE);
            }
            break;
        } 
        case 7: {
            // [executable name], arg_clusterFile, arg_auxEnvDim, arg_ctmIter,
            // env_init, env_file_fmt, in_files_prefix
            CtmEnv::init_env_type init_Env = toINIT_ENV(std::string(argv[4]));
            io_env_fmt_type ioEnvFmt = toIO_ENV_FMT(std::string(argv[5]));
            std::string in_files_prefix = std::string(argv[6]);
            if (init_Env == CtmEnv::INIT_ENV_file) {
                ctmEnv = CtmEnv("TEST_ENV_2x2", 
                    readEnv(ioEnvFmt, in_files_prefix, cluster),
                    cluster);
            } else {
                std::cout<< argv[4] << " requires its additional args" << "\n";
                exit(EXIT_FAILURE);
            }
            break;
        }
        default: { 
            // argc should at most 3 for correct execution
            std::cout <<"Invalid amount of Agrs (>6)"<< std::endl;
            exit(EXIT_FAILURE);
            break;
        }
    }

    ctmEnv.printSVDspec(); //DBG

    CtmEnv::isometry_type iso_type = CtmEnv::ISOMETRY_T3;

    for (int iter=1; iter<=arg_ctmIter; iter++ ) {
        
        ctmEnv.insURow_DBG(iso_type);
        ctmEnv.insDRow_DBG(iso_type);
        //ctmEnv.normalizePTN();

        ctmEnv.insLCol_DBG(iso_type);
        ctmEnv.insRCol_DBG(iso_type);
        //ctmEnv.normalizePTN();

        // Normalize
        ctmEnv.normalizeBLE();
        //ctmEnv.normalizePTN();

        if ( iter % 50 == 0 ) {
            ctmEnv.computeSVDspec();
            ctmEnv.printSVDspec(); //DBG
        }
    }

    //writeEnv(IO_ENV_FMT_txt, "TEST", ctmEnv.getCtmData());

    EVBuilder ev("TEST_ENV_2x2", cluster, ctmEnv);

    auto mpo_id = ev.getTOT_DBG(EVBuilder::MPO_Id, 
        cluster.sites.at(cluster.siteIds[0]), 0);

    auto mpo_sz = ev.getTOT_DBG(EVBuilder::MPO_S_Z, 
        cluster.sites.at(cluster.siteIds[0]), 0);

    auto mpo_sz2 = ev.getTOT_DBG(EVBuilder::MPO_S_Z2, 
        cluster.sites.at(cluster.siteIds[0]), 0);

    std::cout <<"ID: "<< ev.eV_1sO(mpo_id, std::make_pair(0,0)) << std::endl;
    std::cout <<"SZ: "<< ev.eV_1sO(mpo_sz, std::make_pair(0,0)) << std::endl;
    std::cout <<"SZ2: "<< ev.eV_1sO(mpo_sz2, std::make_pair(0,0)) << std::endl;

    auto op2s_id = ev.get2STOT_DBG(EVBuilder::OP2S_Id,
        cluster.sites.at(cluster.siteIds[0]),
        cluster.sites.at(cluster.siteIds[0]));

    // Prepare rotated on-site tensor
    auto RA = cluster.sites.at(cluster.siteIds[0]);
    auto physI = findtype(RA.inds(),Site);
    auto R = ITensor(physI, prime(physI,1));
    for ( int i=1; i<=physI.m(); i++ ) {
        R.set( physI(physI.m()+1-i), prime(physI,1)(i), pow(-1,i-1) );
    }
    RA *= R;
    RA.prime(Site, -1);
    PrintData(RA);

    auto op2s_ss = ev.get2STOT_DBG(EVBuilder::OP2S_SS,
        cluster.sites.at(cluster.siteIds[0]),
        RA);

    std::cout <<"ID: "<< ev.eV_2sO(op2s_id,
        std::make_pair(0,1), std::make_pair(0,2)) << std::endl;

    std::cout <<"SS: "<< ev.eV_2sO(op2s_ss,
        std::make_pair(0,0), std::make_pair(0,1)) << std::endl;

    return 0;
}
