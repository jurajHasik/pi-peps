#include "ctm-cluster-io.h"
#include "ctm-cluster-env_v2.h"
#include "cluster-ev-builder.h"
#include <chrono>

using namespace itensor;

int main( int argc, char *argv[] ) {
    std::cout.precision( std::numeric_limits< double >::max_digits10 );
    
    // ########################################################################
    // Handle command-line arguments

    int arg_auxEnvDim, arg_ctmIter;
    std::string arg_clusterFile, arg_ioEnvTag, arg_isoType, arg_norm; 
    
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

    CtmEnv::isometry_type iso_type;
    CtmEnv::normalization_type norm_type;

    // Depending on the number of args call different environment 
    // initialization function
    switch (argc) {
        case 7: {
            // [executable name], arg_clusterFile, arg_auxEnvDim, arg_ctmIter,
            // env_init
            CtmEnv::init_env_type init_Env = toINIT_ENV(std::string(argv[4]));
            if ( init_Env == CtmEnv::INIT_ENV_const1 ) {
                ctmEnv = CtmEnv("TEST_ENV_2x2", arg_auxEnvDim, cluster);
                ctmEnv.initMockEnv();
            } else if ( init_Env == CtmEnv::INIT_ENV_ctmrg ) {
                ctmEnv = CtmEnv("TEST_ENV_2x2", arg_auxEnvDim, cluster);
                ctmEnv.initCtmrgEnv();
            } else {
                std::cout<< argv[4] <<" requires its additional args" 
                    << std::endl;
                exit(EXIT_FAILURE);
            }
            iso_type  = toISOMETRY(argv[5]);
            norm_type = toNORMALIZATION(argv[6]);
            break;
        }
        case 8: {
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
            iso_type  = toISOMETRY(argv[6]);
            norm_type = toNORMALIZATION(argv[7]);
            break;
        } 
        case 9: {
            // [executable name], arg_clusterFile, arg_auxEnvDim, arg_ctmIter,
            // env_init, env_file_fmt, in_files_prefix
            CtmEnv::init_env_type init_Env = toINIT_ENV(std::string(argv[4]));
            io_env_fmt_type ioEnvFmt = toIO_ENV_FMT(std::string(argv[5]));
            std::string in_files_prefix = std::string(argv[6]);
            if (init_Env == CtmEnv::INIT_ENV_file) {
                ctmEnv = CtmEnv("TEST_ENV_2x2", 
                    readEnv_V2(ioEnvFmt, in_files_prefix, cluster),
                    cluster);
            } else {
                std::cout<< argv[4] << " requires its additional args" << "\n";
                exit(EXIT_FAILURE);
            }
            iso_type  = toISOMETRY(argv[7]);
            norm_type = toNORMALIZATION(argv[8]);
            break;
        }
        default: {
            // argc should at most 3 for correct execution
            std::cout <<"Invalid amount of Agrs (>6)"<< std::endl;
            exit(EXIT_FAILURE);
            break;
        }
    }

    std::cout << ctmEnv;
    ctmEnv.printSVDspec(); //DBG

    Print(ctmEnv.build_corner('1',0,0));
    Print(ctmEnv.build_corner('2',0,0));
    Print(ctmEnv.build_corner('3',0,0));
    Print(ctmEnv.build_corner('4',0,0));

    Print(ctmEnv.build_2x2_RDM('L',0,0));
    Print(ctmEnv.build_2x2_RDM('U',0,0));
    Print(ctmEnv.build_2x2_RDM('R',0,0));
    Print(ctmEnv.build_2x2_RDM('D',0,0));

    // holds timing for *_DBG mostves
    std::vector<double> accT(8,0.0);
    // holding energies
    std::vector<double> e_nnH;
    std::vector<double> e_nnH_AC;
    std::vector<double> e_nnH_BD;
    std::vector<double> e_nnH_CD;

    // Build expectation value builder
    std::cout << ctmEnv.getCtmData_DBG();
    EVBuilder ev("TEST_ENV_2x2", cluster, ctmEnv.getCtmData_DBG());
    ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());

    // Prepare rotated on-site tensor
    auto RA = cluster.sites.at("A");
    auto physI = findtype(RA.inds(),PHYS);
    auto R = ITensor(physI, prime(physI,1));
    for ( int i=1; i<=physI.m(); i++ ) {
        R.set( physI(physI.m()+1-i), prime(physI,1)(i), pow(-1,i-1) );
    }
    RA *= R;
    RA.prime(PHYS, -1);
    PrintData(RA);
    // auto op2s_ss = ev.get2STOT_DBG(EVBuilder::OP2S_SS,
    auto op2s_ss = ev.get2STOT_DBG(EVBuilder::OP2S_SS,
        cluster.sites.at(cluster.cToS.at(std::make_pair(0,0))), //cluster.sites.at("A"), 
        //RA);
        cluster.sites.at(cluster.cToS.at(std::make_pair(1,0)))); //cluster.sites.at("B"));

    auto op2s_ssAC = ev.get2STOT(EVBuilder::OP2S_SS,
        cluster.sites.at(cluster.cToS.at(std::make_pair(0,0))), //cluster.sites.at("A"),
        cluster.sites.at(cluster.cToS.at(std::make_pair(0,1)))); //cluster.sites.at("C"));

    auto op2s_ssBD = ev.get2STOT(EVBuilder::OP2S_SS,
        cluster.sites.at(cluster.cToS.at(std::make_pair(1,0))), //cluster.sites.at("B"),
        cluster.sites.at(cluster.cToS.at(std::make_pair(1,1)))); //cluster.sites.at("D"));

    auto op2s_ssCD = ev.get2STOT(EVBuilder::OP2S_SS,
        cluster.sites.at(cluster.cToS.at(std::make_pair(0,1))), //cluster.sites.at("C"),
        cluster.sites.at(cluster.cToS.at(std::make_pair(1,1)))); //cluster.sites.at("D"));

    // second set of 2site ops
    auto op2s_ssBA = ev.get2STOT_DBG(EVBuilder::OP2S_SS,
        cluster.sites.at(cluster.cToS.at(std::make_pair(1,0))), //cluster.sites.at("B"), 
        cluster.sites.at(cluster.cToS.at(std::make_pair(0,0)))); //cluster.sites.at("A"));

    auto op2s_ssCA = ev.get2STOT(EVBuilder::OP2S_SS,
        cluster.sites.at(cluster.cToS.at(std::make_pair(0,1))), //cluster.sites.at("C"),
        cluster.sites.at(cluster.cToS.at(std::make_pair(0,0)))); // cluster.sites.at("A"));

    auto op2s_ssDB = ev.get2STOT(EVBuilder::OP2S_SS,
        cluster.sites.at(cluster.cToS.at(std::make_pair(1,1))), //cluster.sites.at("D"),
        cluster.sites.at(cluster.cToS.at(std::make_pair(1,0)))); // cluster.sites.at("B"));

    auto op2s_ssDC = ev.get2STOT(EVBuilder::OP2S_SS,
        cluster.sites.at(cluster.cToS.at(std::make_pair(1,1))), //cluster.sites.at("D"),
        cluster.sites.at(cluster.cToS.at(std::make_pair(0,1)))); // cluster.sites.at("C"));

    // NNN 2site ops
    auto op2s_ssAD = ev.get2STOT_DBG(EVBuilder::OP2S_SS,
        cluster.sites.at(cluster.cToS.at(std::make_pair(0,0))), //cluster.sites.at("A"), 
        cluster.sites.at(cluster.cToS.at(std::make_pair(1,1)))); //cluster.sites.at("D"));

    auto op2s_ssBC = ev.get2STOT(EVBuilder::OP2S_SS,
        cluster.sites.at(cluster.cToS.at(std::make_pair(1,0))), //cluster.sites.at("B"),
        cluster.sites.at(cluster.cToS.at(std::make_pair(0,1)))); // cluster.sites.at("C"));
    // NNN 2site ops
    auto op2s_ssDA = ev.get2STOT_DBG(EVBuilder::OP2S_SS,
        cluster.sites.at(cluster.cToS.at(std::make_pair(1,1))), //cluster.sites.at("A"), 
        cluster.sites.at(cluster.cToS.at(std::make_pair(0,0)))); //cluster.sites.at("D"));

    auto op2s_ssCB = ev.get2STOT(EVBuilder::OP2S_SS,
        cluster.sites.at(cluster.cToS.at(std::make_pair(0,1))), //cluster.sites.at("B"),
        cluster.sites.at(cluster.cToS.at(std::make_pair(1,0)))); // cluster.sites.at("C"));


    // energy with initial environment
    e_nnH.push_back( ev.eV_2sO_Rectangle(op2s_ss,
        std::make_pair(0,0), std::make_pair(1,0)) );
    e_nnH_AC.push_back( ev.eV_2sO_Rectangle(op2s_ssAC,
        std::make_pair(0,0), std::make_pair(0,1)) );
    e_nnH_BD.push_back( ev.eV_2sO_Rectangle(op2s_ssBD,
        std::make_pair(1,0), std::make_pair(1,1)) );
    e_nnH_CD.push_back( ev.eV_2sO_Rectangle(op2s_ssCD,
        std::make_pair(0,1), std::make_pair(1,1)) );

    // ##### randomization of directional CTM moves ###################
    // Seed with a real random value, if available
    // std::random_device rd;  // Will be used to obtain a seed for the 
    //                         // random number engine
    // std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded 
    //                         // with rd()
    // std::uniform_int_distribution<> dis(1, 4);
    // ##### randomization of directional CTM moves ###################

    // Start timing iteration loop
    std::chrono::steady_clock::time_point t_begin = 
        std::chrono::steady_clock::now();

    for (int iter=1; iter<=arg_ctmIter; iter++ ) {

        ctmEnv.insLCol_DBG(iso_type, norm_type, accT, true);
        ctmEnv.insRCol_DBG(iso_type, norm_type, accT, true);
        ctmEnv.insURow_DBG(iso_type, norm_type, accT, true);
        ctmEnv.insDRow_DBG(iso_type, norm_type, accT, true);

        std::cout << "STEP " << iter << std::endl;

        if ( iter % 1 == 0 ) {
            ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());
            e_nnH.push_back( ev.eV_2sO_Rectangle(op2s_ss,
                std::make_pair(0,0), std::make_pair(1,0)) );
            e_nnH_AC.push_back( ev.eV_2sO_Rectangle(op2s_ssAC,
                std::make_pair(0,0), std::make_pair(0,1)) );
            e_nnH_BD.push_back( ev.eV_2sO_Rectangle(op2s_ssBD,
                std::make_pair(1,0), std::make_pair(1,1)) );
            e_nnH_CD.push_back( ev.eV_2sO_Rectangle(op2s_ssCD,
                std::make_pair(0,1), std::make_pair(1,1)) );
        }
    }

    // End timing iteration loop
    std::chrono::steady_clock::time_point t_end =
        std::chrono::steady_clock::now();

    std::cout <<"time [sec] = "<< std::chrono::duration_cast
        <std::chrono::microseconds>(t_end - t_begin).count()/1000000.0 
        << std::endl;
    std::cout <<"accT [mSec]: "<< accT[0] <<" "<< accT[1] <<" "<< accT[2]
        <<" "<< accT[3] << std::endl;
    std::cout <<"isoZ [mSec]: "<< accT[4] <<" "<< accT[5] <<" "<< accT[6]
        <<" "<< accT[7] << std::endl;

    std::cout << ctmEnv;

    //writeEnv(IO_ENV_FMT_txt, "TEST", ctmEnv.getCtmData());
    ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());

    auto mpo_id  = ev.getTOT_DBG(EVBuilder::MPO_Id, "A", 0);
    auto mpo_sz2 = ev.getTOT_DBG(EVBuilder::MPO_S_Z2, "A", 0);

    auto mpo_sz  = ev.getTOT_DBG(EVBuilder::MPO_S_Z, "A", 0);
    auto mpo_szB = ev.getTOT_DBG(EVBuilder::MPO_S_Z, "B", 0);
    auto mpo_szC = ev.getTOT_DBG(EVBuilder::MPO_S_Z, "C", 0);
    auto mpo_szD = ev.getTOT_DBG(EVBuilder::MPO_S_Z, "D", 0);

    auto mpo_sp  = ev.getTOT_DBG(EVBuilder::MPO_S_P, "A", 0);
    auto mpo_sm  = ev.getTOT_DBG(EVBuilder::MPO_S_M, "A", 0);

    auto mpo_spB = ev.getTOT_DBG(EVBuilder::MPO_S_P, "B", 0);
    auto mpo_smB = ev.getTOT_DBG(EVBuilder::MPO_S_M, "B", 0);

    std::cout <<"ID: "<< ev.eV_1sO_1sENV(mpo_id, std::make_pair(0,0)) << std::endl;
    std::cout <<"SZ2: "<< ev.eV_1sO_1sENV(mpo_sz2, std::make_pair(0,0)) << std::endl;

    std::cout <<"SZ: "<< ev.eV_1sO_1sENV(mpo_sz, std::make_pair(0,0)) << std::endl;
    std::cout <<"SZ: "<< ev.eV_1sO_1sENV(mpo_szB, std::make_pair(1,0)) << std::endl;
    std::cout <<"SZ: "<< ev.eV_1sO_1sENV(mpo_szC, std::make_pair(0,1)) << std::endl;
    std::cout <<"SZ: "<< ev.eV_1sO_1sENV(mpo_szD, std::make_pair(1,1)) << std::endl;

    std::cout <<"SP A: "<< ev.eV_1sO_1sENV(mpo_sp, std::make_pair(0,0)) << std::endl;
    std::cout <<"SP B: "<< ev.eV_1sO_1sENV(mpo_spB, std::make_pair(1,0)) << std::endl;
    std::cout <<"SM A: "<< ev.eV_1sO_1sENV(mpo_sm, std::make_pair(0,0)) << std::endl;
    std::cout <<"SM B: "<< ev.eV_1sO_1sENV(mpo_smB, std::make_pair(1,0)) << std::endl;

    // auto op2s_id = ev.get2STOT_DBG(EVBuilder::OP2S_Id,
    // std::cout <<"Constructing 2-site MPO on sites: "<< cluster.siteIds[0]
    //     <<" and "<< cluster.siteIds[1] << std::endl;
    // auto op2s_id = ev.get2STOT(EVBuilder::OP2S_Id,
    //     cluster.sites.at(cluster.siteIds[0]),
    //     cluster.sites.at(cluster.siteIds[1]));

    // std::cout <<"ID: "<< ev.eV_2sO(op2s_id,
    //     std::make_pair(0,0), std::make_pair(0,3)) << std::endl;

    std::cout <<"ITER: "<<" E:"<< std::endl;
    for ( std::size_t i=0; i<e_nnH.size(); i++ ) {
        std::cout << i <<" "<< e_nnH[i] 
            <<" "<< e_nnH_AC[i]
            <<" "<< e_nnH_BD[i]
            <<" "<< e_nnH_CD[i]
            << std::endl;
    }

    std::cout << "BA: " << ev.eV_2sO_Rectangle(op2s_ssBA,
        std::make_pair(1,0), std::make_pair(2,0)) << 
    " CA: "<< ev.eV_2sO_Rectangle(op2s_ssCA,
        std::make_pair(0,1), std::make_pair(0,2)) <<
    " DB: "<< ev.eV_2sO_Rectangle(op2s_ssDB,
        std::make_pair(1,1), std::make_pair(1,2)) <<
    " DC: "<< ev.eV_2sO_Rectangle(op2s_ssDC,
        std::make_pair(1,1), std::make_pair(2,1)) << std::endl;

    std::cout << "AD: " << ev.eV_2sO_Rectangle(op2s_ssAD,
        std::make_pair(0,0), std::make_pair(1,1)) << 
    " DA: "<< ev.eV_2sO_Rectangle(op2s_ssDA,
        std::make_pair(1,1), std::make_pair(2,2)) <<
    " BC: "<< ev.eV_2sO_Rectangle(op2s_ssBC,
        std::make_pair(1,0), std::make_pair(2,1)) <<
    " CB: "<< ev.eV_2sO_Rectangle(op2s_ssCB,
        std::make_pair(0,1), std::make_pair(1,2)) << std::endl;

    
    /*ev.expVal_1sO1sO_H( 
        EVBuilder::MPO_Id, EVBuilder::MPO_Id,
        std::make_pair(0, 0), 20);

    ev.expVal_1sO1sO_H( 
        EVBuilder::MPO_S_Z, EVBuilder::MPO_S_Z,
        std::make_pair(0, 0), 20);

    ev.expVal_1sO1sO_H( 
        EVBuilder::MPO_S_P, EVBuilder::MPO_S_M,
        std::make_pair(0, 0), 20);

    ev.expVal_1sO1sO_H( 
        EVBuilder::MPO_S_M, EVBuilder::MPO_S_P,
        std::make_pair(0, 0), 20);*/

    return 0;
}
