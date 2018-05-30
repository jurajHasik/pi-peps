#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include "json.hpp"
#include "models.h"
#include "simple-update_v2.h"
#include "ctm-cluster-env_v2.h"
#include "cluster-ev-builder.h"
#include "ctm-cluster-io.h"
#include "ctm-cluster.h"

using namespace itensor;

int main( int argc, char *argv[] ) {

    // ***** INITIALIZE SIMPLE UPDATE ALGORITHM *******************************
	std::string arg_simFile = std::string(argv[1]);
	std::ifstream simfile(arg_simFile, std::ios::in);

	nlohmann::json jsonCls;
	simfile >> jsonCls;

    // write simulation parameters to log file
    std::cout << jsonCls.dump(4) << std::endl;

	//read cluster infile OR initialize by one of the predefined
	//options FILE, RND, RND_AB, AFM, RVB, ...
	std::string initBy(jsonCls["initBy"].get<std::string>());

	int physDim, auxBondDim;
	std::string inClusterFile;
	inClusterFile = jsonCls["inClusterFile"].get<std::string>();
	physDim       = jsonCls["physDim"].get<int>();
	auxBondDim    = jsonCls["auxBondDim"].get<int>();

	// read cluster outfile
	std::string outClusterFile(jsonCls["outClusterFile"].get<std::string>());

	// read Hamiltonian and Trotter decomposition
    auto json_model_params(jsonCls["model"]);
	
    std::string sitesInit(jsonCls.value("sitesInit","FILE"));
    std::string suWeightsInit(jsonCls.value("suWeightsInit","DELTA"));

    int arg_suIter  = jsonCls["suIter"].get<int>();
    int arg_obsFreq = jsonCls["obsFreq"].get<int>();
    bool arg_suDbg  = jsonCls["suDbg"].get<bool>();
    int arg_suDbgLevel = jsonCls["suDbgLevel"].get<int>();

    // read CTMRG parameters
    auto json_ctmrg_params(jsonCls["ctmrg"]);
    int auxEnvDim = json_ctmrg_params["auxEnvDim"].get<int>();
    std::string arg_ioEnvTag(json_ctmrg_params["ioEnvTag"].get<std::string>());
    CtmEnv::init_env_type arg_initEnvType(toINIT_ENV(json_ctmrg_params["initEnvType"].get<std::string>()));
    bool envIsComplex = json_ctmrg_params["envIsComplex"].get<bool>();
    CtmEnv::isometry_type iso_type(toISOMETRY(json_ctmrg_params["isoType"].get<std::string>()));
    double arg_isoPseudoInvCutoff = json_ctmrg_params["isoPseudoInvCutoff"].get<double>();
    CtmEnv::normalization_type norm_type(toNORMALIZATION(json_ctmrg_params["normType"].get<std::string>()));
    std::string env_SVD_METHOD(json_ctmrg_params["env_SVD_METHOD"].get<std::string>()); 
    int arg_maxEnvIter = json_ctmrg_params["maxEnvIter"].get<int>();
    int arg_maxInitEnvIter = json_ctmrg_params["initMaxEnvIter"].get<int>();
    double arg_envEps  = json_ctmrg_params["envEpsilon"].get<double>();
    bool arg_reinitEnv = json_ctmrg_params["reinitEnv"].get<bool>();
    bool arg_envDbg    = json_ctmrg_params["dbg"].get<bool>();
    int arg_envDbgLvl  = json_ctmrg_params["dbgLvl"].get<int>();
    // end reading CTMRG parameters

    // ***** INITIALIZE SIMPLE UPDATE ALGORITHM DONE **************************

	// ***** INITIALIZE CLUSTER ***********************************************
	Cluster cls( readCluster(inClusterFile) );
        
    // choose initial wavefunction
    if (sitesInit == "FILE") {
        // Do nothing - the on-site tensor have been initialized from
        // inClusterFile
    } else {
        setSites(cls, sitesInit);
    }
    setWeights(cls, suWeightsInit);
    std::cout << cls;
    // ***** INITIALIZE CLUSTER DONE ******************************************

    // ***** INITIALIZE MODEL *************************************************
    // DEFINE GATE SEQUENCE
    std::vector< MPO_3site > gateMPO;
    std::vector< MPO_3site * > ptr_gateMPO;
    std::vector< std::vector<std::string> > gates;
    std::vector< std::vector<int> > gate_auxInds;

    // Generate gates for given model by Trotter decomposition
    getModel(json_model_params, gateMPO, ptr_gateMPO, gates, gate_auxInds);

    // For symmetric Trotter decomposition
    int init_gate_size = gates.size();
    for (int i=0; i<init_gate_size; i++) {
        ptr_gateMPO.push_back(ptr_gateMPO[init_gate_size-1-i]);
        gates.push_back(gates[init_gate_size-1-i]);
        gate_auxInds.push_back(gate_auxInds[init_gate_size-1-i]);
    }
    // ***** INITIALIZE MODEL DONE ********************************************

    // ***** INITIALIZE ENVIRONMENT *******************************************
    Cluster evCls = cls;
    absorbWeightsToSites(evCls);
    CtmEnv ctmEnv(arg_ioEnvTag, auxEnvDim, evCls, 
        {"isoPseudoInvCutoff",arg_isoPseudoInvCutoff,
         "SVD_METHOD",env_SVD_METHOD,
         "dbg",arg_envDbg,
         "dbgLevel",arg_envDbgLvl}
        );
    switch (arg_initEnvType) {
        case CtmEnv::INIT_ENV_const1: {
            ctmEnv.initMockEnv();
            break;
        }
        case CtmEnv::INIT_ENV_ctmrg: {
            ctmEnv.initCtmrgEnv();
            break;
        }
        case CtmEnv::INIT_ENV_rnd: {
            ctmEnv.initRndEnv(envIsComplex);
            break;
        }
    }
    
    // INITIALIZE EXPECTATION VALUE BUILDER
    EVBuilder ev(arg_ioEnvTag, evCls, ctmEnv.getCtmData_DBG());
    ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());
    // ***** INITIALIZE ENVIRONMENT DONE **************************************

    // ########################################################################
    // # SETUP OPTIMIZATION LOOP                                              #
    // ########################################################################

    std::ofstream out_file_energy(outClusterFile+".energy.dat", std::ios::out);
    std::ofstream out_file_diag(outClusterFile+".diag.dat", std::ios::out);
    out_file_energy.precision( std::numeric_limits< double >::max_digits10 );
    out_file_diag.precision( std::numeric_limits< double >::max_digits10 );

    // Diagnostic data
    std::vector<int> diag_ctmIter;
    std::vector< Args > diagData_fu;
    Args diag_fu;

    // hold energies
    std::vector<double> e_curr(4,0.0), e_prev(4,0.0);
    std::vector<double> e_nnH;
    std::vector<double> e_nnH_AC;
    std::vector<double> e_nnH_BD;
    std::vector<double> e_nnH_CD;
    std::vector<double> evNN;
    std::vector<double> evNNN;
    std::vector<double> ev_sA(3,0.0);
    std::vector<double> ev_sB(3,0.0);
    std::vector<double> ev_sC(3,0.0);
    std::vector<double> ev_sD(3,0.0);

    std::vector<double> accT(8,0.0); // holds timings for CTM moves
    std::chrono::steady_clock::time_point t_begin_int, t_end_int;

    Args suArgs = {
        "suDbg",arg_suDbg,
        "suDbgLevel",arg_suDbgLevel
    };

    // ENTER OPTIMIZATION LOOP
    for (int suI = 1; suI <= arg_suIter; suI++) {
        std::cout <<"Simple Update - STEP "<< suI << std::endl;

        // PERFORM SIMPLE UPDATE
        std::cout << "GATE: " << (suI-1)%gates.size() << std::endl;

        diag_fu = simpleUpdate(*(ptr_gateMPO[(suI-1)%gates.size()]), cls,
            gates[(suI-1)%gates.size()], gate_auxInds[(suI-1)%gates.size()], suArgs);

        diagData_fu.push_back(diag_fu);

        // out_file_diag << fuI <<" "<< diag_ctmIter.back() <<" "<< diag_fu.getInt("alsSweep",0)
        //     <<" "<< diag_fu.getString("siteMaxElem")
        //     <<" "<< diag_fu.getReal("finalDist0",0.0)
        //     <<" "<< diag_fu.getReal("finalDist1",0.0);
        // out_file_diag <<" "<< diag_fu.getReal("ratioNonSymLE",0.0)
        //     <<" "<< diag_fu.getReal("ratioNonSymFN",0.0);
        // out_file_diag <<" "<< diag_fu.getReal("minGapDisc",0.0) 
        //     <<" "<< diag_fu.getReal("minEvKept",0.0);
        // out_file_diag  <<std::endl;

        // writeCluster(outClusterFile, cls);
        
        if (suI % arg_obsFreq == 0) {
            evCls = cls;
            absorbWeightsToSites(evCls);
            ctmEnv.updateCluster(evCls);
            ev.setCluster(evCls);
            
            // ENTER ENVIRONMENT LOOP
            for (int envI=1; envI<=arg_maxEnvIter; envI++ ) {
                t_begin_int = std::chrono::steady_clock::now();

                // ctmEnv.insLCol_DBG(iso_type, norm_type, accT);
                // ctmEnv.insRCol_DBG(iso_type, norm_type, accT);
                // ctmEnv.insURow_DBG(iso_type, norm_type, accT);
                // ctmEnv.insDRow_DBG(iso_type, norm_type, accT);

                ctmEnv.insLCol_DBG(iso_type, norm_type, accT);
                ctmEnv.insURow_DBG(iso_type, norm_type, accT);
                ctmEnv.insRCol_DBG(iso_type, norm_type, accT);
                ctmEnv.insDRow_DBG(iso_type, norm_type, accT);

                t_end_int = std::chrono::steady_clock::now();
                std::cout << "CTM STEP " << envI <<" T: "<< std::chrono::duration_cast
                        <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
                        <<" [sec] "; 

                if ( (arg_maxEnvIter > 1) && (envI % 1 == 0) ) {
                    t_begin_int = std::chrono::steady_clock::now();
                    
                    ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());

                    // e_curr[0]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,0), std::make_pair(1,0));
                    // e_curr[1]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,0), std::make_pair(0,1));
                    // e_curr[2]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(1,0), std::make_pair(1,1));
                    // e_curr[3]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,1), std::make_pair(1,1));

                    e_curr[0]=ev.eval2Smpo(EVBuilder::OP2S_SZSZ, std::make_pair(0,0), std::make_pair(1,0));
                    e_curr[1]=ev.eval2Smpo(EVBuilder::OP2S_SZSZ, std::make_pair(0,0), std::make_pair(0,1));
                    e_curr[2]=ev.eval2Smpo(EVBuilder::OP2S_SZSZ, std::make_pair(1,0), std::make_pair(1,1));
                    e_curr[3]=ev.eval2Smpo(EVBuilder::OP2S_SZSZ, std::make_pair(0,1), std::make_pair(1,1));

                    t_end_int = std::chrono::steady_clock::now();

                    std::cout<<" || E in T: "<< std::chrono::duration_cast
                        <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
                        <<" [sec] E: "<< e_curr[0] <<" "<< e_curr[1] <<" "<< e_curr[2] <<" "
                        << e_curr[3]; 

                    // if the difference between energies along NN links is lower then arg_envEps
                    // consider the environment converged
                    if ((std::abs(e_prev[0]-e_curr[0]) < arg_envEps) &&
                        (std::abs(e_prev[1]-e_curr[1]) < arg_envEps) &&
                        (std::abs(e_prev[2]-e_curr[2]) < arg_envEps) &&
                        (std::abs(e_prev[3]-e_curr[3]) < arg_envEps) ) {

                        diag_ctmIter.push_back(envI);
                        std::cout<< " ENV CONVERGED ";
                        break;
                    }

                    if ( arg_envDbg && (envI==arg_maxEnvIter) )  {
                        diag_ctmIter.push_back(envI);
                        // diagnose spectra
                        ITensor tL(ctmEnv.C_LU[0].inds().front()),sv,tR;
                        auto spec = svd(ctmEnv.C_LU[0],tL,sv,tR);
                        Print(spec);

                        tL = ITensor(ctmEnv.C_RU[0].inds().front());
                        spec = svd(ctmEnv.C_RU[0],tL,sv,tR);
                        Print(spec);

                        tL = ITensor(ctmEnv.C_RD[0].inds().front());
                        spec = svd(ctmEnv.C_RD[0],tL,sv,tR);
                        Print(spec);

                        tL = ITensor(ctmEnv.C_LD[0].inds().front());
                        spec = svd(ctmEnv.C_LD[0],tL,sv,tR);
                        Print(spec);
                    }
                    e_prev = e_curr;
                }
                std::cout << std::endl;
            }

            std::cout <<"accT [mSec]: "<< accT[0] <<" "<< accT[1] <<" "<< accT[2]
                <<" "<< accT[3] << std::endl;
            std::cout <<"isoZ [mSec]: "<< accT[4] <<" "<< accT[5] <<" "<< accT[6]
                <<" "<< accT[7] << std::endl;

            
            ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());

            t_begin_int = std::chrono::steady_clock::now();

            // compute energies NN links
            // e_nnH.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
            //     std::make_pair(0,0), std::make_pair(1,0)) );
            // e_nnH_AC.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
            //     std::make_pair(0,0), std::make_pair(0,1)) );
            // e_nnH_BD.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
            //     std::make_pair(1,0), std::make_pair(1,1)) );
            // e_nnH_CD.push_back( ev.eval2Smpo(EVBuilder::OP2S_SS,
            //     std::make_pair(0,1), std::make_pair(1,1)) );

            // evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
            //     std::make_pair(1,0), std::make_pair(2,0))); //BA
            // evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
            //     std::make_pair(0,1), std::make_pair(0,2))); //CA
            // evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
            //     std::make_pair(1,1), std::make_pair(1,2))); //DB
            // evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SS,
            //     std::make_pair(1,1), std::make_pair(2,1))); //DC

            e_nnH.push_back( ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
                std::make_pair(0,0), std::make_pair(1,0)) );
            e_nnH_AC.push_back( ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
                std::make_pair(0,0), std::make_pair(0,1)) );
            e_nnH_BD.push_back( ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
                std::make_pair(1,0), std::make_pair(1,1)) );
            e_nnH_CD.push_back( ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
                std::make_pair(0,1), std::make_pair(1,1)) );

            evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
                std::make_pair(1,0), std::make_pair(2,0))); //BA
            evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
                std::make_pair(0,1), std::make_pair(0,2))); //CA
            evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
                std::make_pair(1,1), std::make_pair(1,2))); //DB
            evNN.push_back(ev.eval2Smpo(EVBuilder::OP2S_SZSZ,
                std::make_pair(1,1), std::make_pair(2,1))); //DC

            t_end_int = std::chrono::steady_clock::now();
            std::cout << "NN <S.S> computed in T: "<< std::chrono::duration_cast
                    <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
                    <<" [sec] "<< std::endl;

            t_begin_int = std::chrono::steady_clock::now();

            // compute energies NNN links
            evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(0,0)) );
            evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(1,1)) );
            evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(1,0)) );
            evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(2,1)) );

            evNNN.push_back( ev.eval2x2DiagN11(EVBuilder::OP2S_SS, std::make_pair(0,0)) );
            evNNN.push_back( ev.eval2x2DiagN11(EVBuilder::OP2S_SS, std::make_pair(1,1)) );
            evNNN.push_back( ev.eval2x2DiagN11(EVBuilder::OP2S_SS, std::make_pair(1,0)) );
            evNNN.push_back( ev.eval2x2DiagN11(EVBuilder::OP2S_SS, std::make_pair(0,1)) );
        
            t_end_int = std::chrono::steady_clock::now();
            std::cout << "NNN <S.S> computed in T: "<< std::chrono::duration_cast
                    <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
                    <<" [sec] "<< std::endl;

            t_begin_int = std::chrono::steady_clock::now();

            ev_sA[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(0,0));
            ev_sA[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(0,0));
            ev_sA[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(0,0));

            ev_sB[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(1,0));
            ev_sB[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(1,0));
            ev_sB[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(1,0));

            ev_sC[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(0,1));
            ev_sC[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(0,1));
            ev_sC[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(0,1));

            ev_sD[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, std::make_pair(1,1));
            ev_sD[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, std::make_pair(1,1));
            ev_sD[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, std::make_pair(1,1));

            t_end_int = std::chrono::steady_clock::now();
            std::cout << "<S> computed in T: "<< std::chrono::duration_cast
                    <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
                    <<" [sec] "<< std::endl;

            // write observables
            double avgE_8links = 0.;
            out_file_energy << suI-1 <<" "<< e_nnH.back() 
                <<" "<< e_nnH_AC.back()
                <<" "<< e_nnH_BD.back()
                <<" "<< e_nnH_CD.back();
            for ( unsigned int j=evNN.size()-4; j<evNN.size(); j++ ) {
                avgE_8links += evNN[j];
                out_file_energy<<" "<< evNN[j];
            }
            avgE_8links = (avgE_8links + e_nnH.back() + e_nnH_AC.back() 
                + e_nnH_BD.back() + e_nnH_CD.back())/8.0;
            out_file_energy <<" "<< avgE_8links;
            
            double evNNN_avg = 0.;
            for(int evnnni=evNNN.size()-8; evnnni < evNNN.size(); evnnni++)   
                evNNN_avg += evNNN[evnnni];
            evNNN_avg = evNNN_avg / 8.0;

            out_file_energy <<" "<< evNNN_avg;
            //out_file_energy <<" "<< avgE_8links + arg_J2*evNNN_avg;
            
            // write magnetization
            double evMag_avg = 0.;
            // evMag_avg = 0.25*(
            //     sqrt(ev_sA[0]*ev_sA[0] + ev_sA[1]*ev_sA[1] )//+ ev_sA[2]*ev_sA[2])
            //     + sqrt(ev_sB[0]*ev_sB[0] + ev_sB[1]*ev_sB[1] )//+ ev_sB[2]*ev_sB[2])
            //     + sqrt(ev_sC[0]*ev_sC[0] + ev_sC[1]*ev_sC[1] )//+ ev_sC[2]*ev_sC[2])
            //     + sqrt(ev_sD[0]*ev_sD[0] + ev_sD[1]*ev_sD[1] )//+ ev_sD[2]*ev_sD[2])
            //     );
            evMag_avg = 0.25*(
                sqrt(ev_sA[0]*ev_sA[0])//+ ev_sA[2]*ev_sA[2])
                + sqrt(ev_sB[0]*ev_sB[0])//+ ev_sB[2]*ev_sB[2])
                + sqrt(ev_sC[0]*ev_sC[0])//+ ev_sC[2]*ev_sC[2])
                + sqrt(ev_sD[0]*ev_sD[0])//+ ev_sD[2]*ev_sD[2])
                );
            out_file_energy <<" "<< evMag_avg;
            evMag_avg = 0.25*(
                sqrt(ev_sA[1]*ev_sA[1])//+ ev_sA[2]*ev_sA[2])
                + sqrt(ev_sB[1]*ev_sB[1])//+ ev_sB[2]*ev_sB[2])
                + sqrt(ev_sC[1]*ev_sC[1])//+ ev_sC[2]*ev_sC[2])
                + sqrt(ev_sD[1]*ev_sD[1])//+ ev_sD[2]*ev_sD[2])
                );
            out_file_energy <<" "<< evMag_avg;

            out_file_energy << std::endl;

            writeCluster(outClusterFile, evCls);
        }

    }

    // SIMPLE UPDATE FINISHED
    std::cout <<"SIMPLE UPDATE DONE"<< std::endl;
}