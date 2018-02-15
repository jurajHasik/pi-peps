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
    Args args = {"dbg",true,"dbgLvl",0};
    switch (argc) {
        case 7: {
            // [executable name], arg_clusterFile, arg_auxEnvDim, arg_ctmIter,
            // env_init
            CtmEnv::init_env_type init_Env = toINIT_ENV(std::string(argv[4]));
            if ( init_Env == CtmEnv::INIT_ENV_const1 ) {
                ctmEnv = CtmEnv("TEST_ENV_2x2", arg_auxEnvDim, cluster, args);
                ctmEnv.initMockEnv();
            } else if ( init_Env == CtmEnv::INIT_ENV_ctmrg ) {
                ctmEnv = CtmEnv("TEST_ENV_2x2", arg_auxEnvDim, cluster, args);
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

                ctmEnv = CtmEnv("TEST_ENV_2x2", arg_auxEnvDim, cluster, args);
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

    //DEBUG
    //std::cout << ctmEnv;
    //ctmEnv.printSVDspec();
    // Print(ctmEnv.build_corner('1',0,0));
    // Print(ctmEnv.build_corner('2',0,0));
    // Print(ctmEnv.build_corner('3',0,0));
    // Print(ctmEnv.build_corner('4',0,0));
    // Print(ctmEnv.build_2x2_RDM('L',0,0));
    // Print(ctmEnv.build_2x2_RDM('U',0,0));
    // Print(ctmEnv.build_2x2_RDM('R',0,0));
    // Print(ctmEnv.build_2x2_RDM('D',0,0));

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
    // auto RA = cluster.sites.at("A");
    // auto physI = findtype(RA.inds(),PHYS);
    // auto R = ITensor(physI, prime(physI,1));
    // for ( int i=1; i<=physI.m(); i++ ) {
    //     R.set( physI(physI.m()+1-i), prime(physI,1)(i), pow(-1,i-1) );
    // }
    // RA *= R;
    // RA.prime(PHYS, -1);
    // PrintData(RA);

    // energy with initial environment
    e_nnH.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,0), std::make_pair(1,0)) );
    e_nnH_AC.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,0), std::make_pair(0,1)) );
    e_nnH_BD.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,0), std::make_pair(1,1)) );
    e_nnH_CD.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
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
    std::chrono::steady_clock::time_point t_begin_int, t_end_int; 
    t_begin_int = std::chrono::steady_clock::now();

    for (int iter=1; iter<=arg_ctmIter; iter++ ) {

        ctmEnv.insLCol_DBG(iso_type, norm_type, accT);
        ctmEnv.insRCol_DBG(iso_type, norm_type, accT);
        ctmEnv.insURow_DBG(iso_type, norm_type, accT);
        ctmEnv.insDRow_DBG(iso_type, norm_type, accT);

        if ( iter % 1 == 0 ) {
            ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());
            e_nnH.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
                std::make_pair(0,0), std::make_pair(1,0)) );
            e_nnH_AC.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
                std::make_pair(0,0), std::make_pair(0,1)) );
            e_nnH_BD.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
                std::make_pair(1,0), std::make_pair(1,1)) );
            e_nnH_CD.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
                std::make_pair(0,1), std::make_pair(1,1)) );
        
            t_end_int = std::chrono::steady_clock::now();
            std::cout << "STEP " << iter <<" T: "<< std::chrono::duration_cast
            <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
                <<" [sec]"<< std::endl;
            t_begin_int = std::chrono::steady_clock::now();
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

    //writeEnv(IO_ENV_FMT_txt, "TEST", ctmEnv.getCtmData());
    ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());

    std::cout <<"ITER: "<<" E:"<< std::endl;
    for ( std::size_t i=0; i<e_nnH.size(); i++ ) {
        std::cout << i <<" "<< e_nnH[i] 
            <<" "<< e_nnH_AC[i]
            <<" "<< e_nnH_BD[i]
            <<" "<< e_nnH_CD[i]
            << std::endl;
    }

    std::vector<double> evNN;
    std::vector<double> evNNN;

    evNN.push_back(e_nnH[e_nnH.size()-1]);
    evNN.push_back(e_nnH_AC[e_nnH.size()-1]);
    evNN.push_back(e_nnH_BD[e_nnH.size()-1]);
    evNN.push_back(e_nnH_CD[e_nnH.size()-1]);

    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,0), std::make_pair(2,0))); //BA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(0,1), std::make_pair(0,2))); //CA
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,1), std::make_pair(1,2))); //DB
    evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
        std::make_pair(1,1), std::make_pair(2,1))); //DC

    std::cout <<"BA: "<< evNN[4] <<" CA: "<< evNN[5] <<" DB: "<< evNN[6] 
        <<" DC: "<< evNN[7] << std::endl;

    evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(0,0)) );
    evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(1,1)) );
    evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(1,0)) );
    evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(2,1)) );

    evNNN.push_back( ev.eval2x2DiagN1N1(EVBuilder::OP2S_SS, std::make_pair(0,0)) );
    evNNN.push_back( ev.eval2x2DiagN1N1(EVBuilder::OP2S_SS, std::make_pair(1,1)) );
    evNNN.push_back( ev.eval2x2DiagN1N1(EVBuilder::OP2S_SS, std::make_pair(1,0)) );
    evNNN.push_back( ev.eval2x2DiagN1N1(EVBuilder::OP2S_SS, std::make_pair(0,1)) );

    // evNNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
    //     std::make_pair(0,0), std::make_pair(1,1))); //A->(1,1)->D
    // evNNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
    //     std::make_pair(1,1), std::make_pair(2,2))); //A->(-1,-1)->D
    // evNNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
    //     std::make_pair(4,0), std::make_pair(3,1))); //A->(1,-1)->D
    // evNNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
    //     std::make_pair(3,1), std::make_pair(2,2))); //A->(-1,1)->D 

    // evNNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
    //     std::make_pair(1,0), std::make_pair(2,1))); //BC
    // evNNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
    //      std::make_pair(0,1), std::make_pair(1,2))); //CB
    // evNNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
    //     std::make_pair(1,0), std::make_pair(0,1))); //BC
    // evNNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
    //     std::make_pair(2,1), std::make_pair(1,2))); //CB

    std::cout <<"AD: "<< evNNN[0] <<" DA: "<< evNNN[1] <<" AD: "<< evNNN[2]
        <<" DA: "<< evNNN[3] << std::endl;
    std::cout <<"BC: "<< evNNN[4] <<" CB: "<< evNNN[5] <<" BC: "<< evNNN[6]
        <<" CB: "<< evNNN[7] << std::endl;

    std::cout <<"Norm site=[0,0] : "<< ev.getNorm_Rectangle(false, std::make_pair(0,0), 
        std::make_pair(0,0)) << std::endl;
    std::cout <<"Norm Region=([0,0],[1,1]) : "<< ev.getNorm_Rectangle(false, 
        std::make_pair(0,0), std::make_pair(1,1)) << std::endl;

    std::cout <<"ID: "<< ev.eV_1sO_1sENV(EVBuilder::MPO_Id, std::make_pair(0,0)) << std::endl;
    std::cout <<"SZ2: "<< ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z2, std::make_pair(0,0)) << std::endl;

    std::vector<double> sA_zpm;
    std::vector<double> sB_zpm;
    std::vector<double> sC_zpm;
    std::vector<double> sD_zpm;

    sA_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(0,0)));
    sA_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(0,0)));
    sA_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(0,0)));
    std::cout<<"S_A: "<< sA_zpm[0] <<", "<< sA_zpm[1] <<", "<< sA_zpm[2] << std::endl;

    sB_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(1,0)));
    sB_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(1,0)));
    sB_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(1,0)));
    std::cout<<"S_B: "<< sB_zpm[0] <<", "<< sB_zpm[1] <<", "<< sB_zpm[2] << std::endl;

    sC_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(0,1)));
    sC_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(0,1)));
    sC_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(0,1)));
    std::cout<<"S_C: "<< sC_zpm[0] <<", "<< sC_zpm[1] <<", "<< sC_zpm[2] << std::endl;

    sD_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(1,1)));
    sD_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(1,1)));
    sD_zpm.push_back(ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(1,1)));
    std::cout<<"S_D: "<< sD_zpm[0] <<", "<< sD_zpm[1] <<", "<< sD_zpm[2] << std::endl;

    std::sort(evNN.begin(), evNN.end());
    std::sort(evNNN.begin(), evNNN.end());

    double evNNavg  = 0.0;
    double evNNNavg = 0.0;
    for (const auto& r : evNN) evNNavg += r;
    for (const auto& r : evNNN) evNNNavg += r;

    std::cout <<"E_NN: "<< evNNavg/8.0 <<" E_NN_split: "<< abs(evNN[0]-evNN[7]) 
        << " E_NNN: "<< evNNNavg/8.0<<" E_NNN_split: "<< abs(evNNN[0]-evNNN[7])
        << std::endl;

    std::cout << ctmEnv;

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
