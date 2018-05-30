#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include "json.hpp"
#include "ctm-cluster-io.h"
#include "ctm-cluster-env_v2.h"
#include "cluster-ev-builder.h"
#include "models.h"
#include "full-update.h"

using namespace itensor;

int main( int argc, char *argv[] ) {

    // ***** INITIALIZE FULL UPDATE ALGORITHM *********************************
	std::string arg_initFile = std::string(argv[1]);
	std::ifstream infile(arg_initFile, std::ios::in);

	nlohmann::json jsonCls;
	infile >> jsonCls;

    // write simulation parameters to log file
    std::cout << jsonCls.dump(4) << std::endl;

	//read cluster infile OR initialize by one of the predefined
	//options FILE, RND, RND_AB, AFM, RVB, ...
	std::string initBy(jsonCls["initBy"].get<std::string>());

	int physDim, auxBondDim;
	std::string inClusterFile;
	if (initBy=="FILE") inClusterFile = jsonCls["inClusterFile"].get<std::string>();
	physDim = jsonCls["physDim"].get<int>();
	auxBondDim = jsonCls["auxBondDim"].get<int>();

	// read cluster outfile
	std::string outClusterFile(jsonCls["outClusterFile"].get<std::string>());

	// read Hamiltonian and Trotter decomposition
    auto json_model_params(jsonCls["model"]);

	// full update parameters
    int arg_fuIter  = jsonCls["fuIter"].get<int>();
    int arg_obsFreq = jsonCls["obsFreq"].get<int>();
    std::string arg_fuIsoInit = jsonCls["fuIsoInit"].get<std::string>();
    double arg_fuIsoInitNoiseLevel = jsonCls["fuIsoInitNoiseLevel"].get<double>();
    int arg_maxAltLstSqrIter = jsonCls["maxAltLstSqrIter"].get<int>();
    bool symmetrizeProtoEnv = jsonCls["symmetrizeProtoEnv"].get<bool>();
    bool posDefProtoEnv = jsonCls["positiveDefiniteProtoEnv"].get<bool>();
    double pseudoInvCutoff = jsonCls["pseudoInvCutoff"].get<double>();
    double pseudoInvMaxLogGap = jsonCls["pseudoInvMaxLogGap"].get<double>();
    double isoEpsilon = jsonCls["isoEpsilon"].get<double>();
    std::string arg_otNormType = jsonCls["otNormType"].get<std::string>();
    bool arg_fuDbg = jsonCls["fuDbg"].get<bool>();
    int arg_fuDbgLevel = jsonCls["fuDbgLevel"].get<int>();
	
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
    // ***** INITIALIZE FULL UPDATE ALGORITHM DONE ****************************

	// INITIALIZE CLUSTER
	Cluster cls;
	if (initBy=="FILE") {
		Index aIA, aIB, pIA, pIB, aIC, aID, pIC, pID;
		ITensor A, B, C, D;

		cls = readCluster(inClusterFile);
		if (cls.auxBondDim > auxBondDim) std::cout <<"Warning: auxBondDim of the"
			<<" input cluster is higher then the desired one!" << std::endl;
		cls.auxBondDim = auxBondDim;
		if (cls.physDim != physDim) std::cout <<"Warning: physDim of the"
			<<" input cluster and simulation parameters are not in agreement!"
			<< std::endl;

		A = cls.sites[cls.cToS.at(std::make_pair(0,0))];
        B = cls.sites[cls.cToS.at(std::make_pair(1,0))];
        C = cls.sites[cls.cToS.at(std::make_pair(0,1))];
        D = cls.sites[cls.cToS.at(std::make_pair(1,1))];

        pIA = noprime(findtype(A.inds(), PHYS));
        pIB = noprime(findtype(B.inds(), PHYS));
        pIC = noprime(findtype(C.inds(), PHYS));
        pID = noprime(findtype(D.inds(), PHYS));
        aIA = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aIB = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aIC = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aID = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        
        // relabel original indices to take up desired bond dimensions
        auto taIA = noprime(findtype(A.inds(), AUXLINK));
        auto taIB = noprime(findtype(B.inds(), AUXLINK));
        auto taIC = noprime(findtype(C.inds(), AUXLINK));
        auto taID = noprime(findtype(D.inds(), AUXLINK));
        
        auto D_I = delta(taIA,aIA);
        A = A*D_I*prime(D_I,1)*prime(D_I,2)*prime(D_I,3);
        D_I = delta(taIB,aIB);
        B = B*D_I*prime(D_I,1)*prime(D_I,2)*prime(D_I,3);
        D_I = delta(taIC,aIC);
        C = C*D_I*prime(D_I,1)*prime(D_I,2)*prime(D_I,3);
        D_I = delta(taID,aID);
        D = D*D_I*prime(D_I,1)*prime(D_I,2)*prime(D_I,3);

        cls.sites = {{"A", A}, {"B", B}, {"C",C}, {"D",D}};
	} else {
        Index aIA, aIB, pIA, pIB, aIC, aID, pIC, pID;
		ITensor A, B, C, D;

        // ----- DEFINE BLANK CLUSTER ----------------------------------
        cls = Cluster();
        cls.sizeN = 2;
        cls.sizeM = 2;
        cls.auxBondDim = auxBondDim;
        cls.physDim    = physDim;

        cls.siteIds = std::vector< std::string >(4);
        cls.siteIds = { "A", "B", "C", "D" };

        cls.cToS  = {
            {std::make_pair(0,0),"A"},
            {std::make_pair(1,0),"B"},
            {std::make_pair(0,1),"C"},
            {std::make_pair(1,1),"D"}
        };

        aIA = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aIB = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aIC = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aID = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        pIA = Index(TAG_I_PHYS, cls.physDim, PHYS);
        pIB = Index(TAG_I_PHYS, cls.physDim, PHYS);
        pIC = Index(TAG_I_PHYS, cls.physDim, PHYS);
        pID = Index(TAG_I_PHYS, cls.physDim, PHYS);

        A = ITensor(aIA, prime(aIA,1), prime(aIA,2), prime(aIA,3), pIA);
        B = ITensor(aIB, prime(aIB,1), prime(aIB,2), prime(aIB,3), pIB);
        C = ITensor(aIC, prime(aIC,1), prime(aIC,2), prime(aIC,3), pIC);
        D = ITensor(aID, prime(aID,1), prime(aID,2), prime(aID,3), pID);

        if (initBy == "RND_AB") {
            std::cout <<"Initializing by RANDOM TENSORS A,B,C=B,D=A"<< std::endl;
            randomize(A);
            randomize(B);

            auto shift05 = [](double r){ return r-0.5; };
            A.apply(shift05);
            B.apply(shift05);

            C = B * delta(pIB, pIC);
            D = A * delta(pIA, pID);
            for (int i=0; i<=3; ++i) {
                C = C * delta(prime(aIB,i), prime(aIC,i));
                D = D * delta(prime(aIA,i), prime(aID,i));
            }
        } else if(initBy == "RANDOM") {
            std::cout <<"Initializing by RANDOM TENSORS"<< std::endl;
            // Randomize

            randomize(A);
            randomize(B);
            randomize(C);
            randomize(D);

            // auto shift05 = [](Real r){ return r-0.5; };
            // A.apply(shift05);
            // B.apply(shift05);
            // C.apply(shift05);
            // D.apply(shift05);

        } else if (initBy == "AFM") {
            std::cout <<"Initializing by AFM order A=down, B=up"<< std::endl;
            // Spin DOWN on site A, spin   UP on site B
            // Spin UP   on site C, spin DOWN on site D
            A.set(aIA(1), prime(aIA,1)(1), prime(aIA,2)(1), prime(aIA,3)(1),
                pIA(2), 1.0);
            B.set(aIB(1), prime(aIB,1)(1), prime(aIB,2)(1), prime(aIB,3)(1),
                pIB(1), 1.0);
            C.set(aIC(1), prime(aIC,1)(1), prime(aIC,2)(1), prime(aIC,3)(1),
                pIC(1), 1.0);
            D.set(aID(1), prime(aID,1)(1), prime(aID,2)(1), prime(aID,3)(1),
                pID(2), 1.0);
        } else if (initBy == "XPRST") {
            std::cout <<"Initializing by PRODUCT STATE along X"<< std::endl;
            // Spin DOWN on site A, spin   UP on site B
            // Spin UP   on site C, spin DOWN on site D
            A.set(aIA(1), prime(aIA,1)(1), prime(aIA,2)(1), prime(aIA,3)(1),
                pIA(1), 1.0/std::sqrt(2.0));
            A.set(aIA(1), prime(aIA,1)(1), prime(aIA,2)(1), prime(aIA,3)(1),
                pIA(2), 1.0/std::sqrt(2.0));
            B.set(aIB(1), prime(aIB,1)(1), prime(aIB,2)(1), prime(aIB,3)(1),
                pIB(1), 1.0/std::sqrt(2.0));
            B.set(aIB(1), prime(aIB,1)(1), prime(aIB,2)(1), prime(aIB,3)(1),
                pIB(2), 1.0/std::sqrt(2.0));
            C.set(aIC(1), prime(aIC,1)(1), prime(aIC,2)(1), prime(aIC,3)(1),
                pIC(1), 1.0/std::sqrt(2.0));
            C.set(aIC(1), prime(aIC,1)(1), prime(aIC,2)(1), prime(aIC,3)(1),
                pIC(2), 1.0/std::sqrt(2.0));
            D.set(aID(1), prime(aID,1)(1), prime(aID,2)(1), prime(aID,3)(1),
                pID(1), 1.0/std::sqrt(2.0));
            D.set(aID(1), prime(aID,1)(1), prime(aID,2)(1), prime(aID,3)(1),
                pID(2), 1.0/std::sqrt(2.0));
        } else {
            std::cout <<"Unsupported cluster initialization: "<< initBy << std::endl;
        }

        cls.sites = {{"A", A}, {"B", B}, {"C",C}, {"D",D}};
        // ----- END DEFINE CLUSTER ------------------------------------
    }

    // INITIALIZE ENVIRONMENT
    CtmEnv ctmEnv(arg_ioEnvTag, auxEnvDim, cls, 
        {"isoPseudoInvCutoff",arg_isoPseudoInvCutoff,
         "SVD_METHOD",env_SVD_METHOD,
         "dbg",arg_envDbg,"dbgLevel",arg_envDbgLvl});
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
        // case CtmEnv::INIT_ENV_file: {
        //     io_env_fmt_type ioEnvFmt = toIO_ENV_FMT(std::string(argv[5]));
        //     std::string in_files_prefix = std::string(argv[6]);
            
        //     ctmEnv = CtmEnv("TEST_ENV_2x2", 
        //         readEnv_V2(ioEnvFmt, in_files_prefix, cluster),
        //         cluster);
        //     break;
        // }
    }
    
    // INITIALIZE EXPECTATION VALUE BUILDER
    EVBuilder ev(arg_ioEnvTag, cls, ctmEnv.getCtmData_DBG());
    ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());

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

    // ##################################################################
    // # SETUP OPTIMIZATION LOOP                                        #
    // ##################################################################

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

    // STORE ISOMETRIES
    std::vector< std::vector< ITensor > > iso_store(
        gates.size(), {ITensor(), ITensor(), ITensor(), ITensor()} );

    Args fuArgs = {
        "maxAltLstSqrIter",arg_maxAltLstSqrIter,
        "fuDbg",arg_fuDbg,
        "fuDbgLevel",arg_fuDbgLevel,
        "fuIsoInit",arg_fuIsoInit,
        "fuIsoInitNoiseLevel",arg_fuIsoInitNoiseLevel,
        "symmetrizeProtoEnv",symmetrizeProtoEnv,
        "positiveDefiniteProtoEnv",posDefProtoEnv,
        "pseudoInvCutoff",pseudoInvCutoff,
        "pseudoInvMaxLogGap",pseudoInvMaxLogGap,
        "isoEpsilon",isoEpsilon,
        "otNormType",arg_otNormType
    };
    // Diagnostic data
    std::vector<int> diag_ctmIter;
    std::vector< Args > diagData_fu;
    Args diag_fu;

    std::ofstream out_file_energy(outClusterFile+".energy.dat", std::ios::out);
    std::ofstream out_file_diag(outClusterFile+".diag.dat", std::ios::out);
    out_file_energy.precision( std::numeric_limits< double >::max_digits10 );
    out_file_diag.precision( std::numeric_limits< double >::max_digits10 );

    // ***** COMPUTING INITIAL ENVIRONMENT ************************************
    std::cout <<"COMPUTING INITIAL ENVIRONMENT "<< std::endl;
    // t_begin_int = std::chrono::steady_clock::now();
        
    // ENTER ENVIRONMENT LOOP
    for (int envI=1; envI<=arg_maxInitEnvIter; envI++ ) {

        ctmEnv.insLCol_DBG(iso_type, norm_type, accT);
        ctmEnv.insRCol_DBG(iso_type, norm_type, accT);
        ctmEnv.insURow_DBG(iso_type, norm_type, accT);
        ctmEnv.insDRow_DBG(iso_type, norm_type, accT);

        if ( envI % 1 == 0 ) {
            ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());
        
            t_end_int = std::chrono::steady_clock::now();
            std::cout << "CTM STEP " << envI <<" T: "<< std::chrono::duration_cast
                <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
                <<" [sec] E: "<< e_curr[0] <<" "<< e_curr[1] <<" "<< e_curr[2] <<" "
                << e_curr[3] << std::endl;
            
            e_curr[0]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,0), std::make_pair(1,0));
            e_curr[1]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,0), std::make_pair(0,1));
            e_curr[2]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(1,0), std::make_pair(1,1));
            e_curr[3]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,1), std::make_pair(1,1));

            if ((std::abs(e_prev[0]-e_curr[0]) < arg_envEps) &&
                (std::abs(e_prev[1]-e_curr[1]) < arg_envEps) &&
                (std::abs(e_prev[2]-e_curr[2]) < arg_envEps) &&
                (std::abs(e_prev[3]-e_curr[3]) < arg_envEps) ) {

                diag_ctmIter.push_back(envI);
                std::cout<< "INIT ENV CONVERGED" << std::endl;
                break;
            }

            if (envI==arg_maxInitEnvIter) {
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
            t_begin_int = std::chrono::steady_clock::now();
        }
    }
    // ***** COMPUTING INITIAL ENVIRONMENT DONE *******************************

    ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());

    // Compute initial properties
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

    // compute energies NNN links
    evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(0,0)) );
    evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(1,1)) );
    evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(1,0)) );
    evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(2,1)) );

    evNNN.push_back( ev.eval2x2DiagN11(EVBuilder::OP2S_SS, std::make_pair(0,0)) );
    evNNN.push_back( ev.eval2x2DiagN11(EVBuilder::OP2S_SS, std::make_pair(1,1)) );
    evNNN.push_back( ev.eval2x2DiagN11(EVBuilder::OP2S_SS, std::make_pair(1,0)) );
    evNNN.push_back( ev.eval2x2DiagN11(EVBuilder::OP2S_SS, std::make_pair(0,1)) );

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

    // write energy
    double avgE_8links = 0.;
    out_file_energy << -1 <<" "<< e_nnH.back() 
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
    //out_file_energy << std::endl;

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

    std::cout <<"FU_ITER: "<<" E:"<< std::endl;
    for ( unsigned int i=0; i<e_nnH.size(); i++ ) {
        std::cout << i <<" "<< e_nnH[i] 
            <<" "<< e_nnH_AC[i]
            <<" "<< e_nnH_BD[i]
            <<" "<< e_nnH_CD[i];
            for ( unsigned int j=i*4; j<(i+1)*4; j++ ) {
                std::cout<<" "<< evNN[j];
            }
        std::cout<< std::endl;
    }
    for(int evnnni=evNNN.size()-8; evnnni < evNNN.size(); evnnni++)   
        std::cout << evNNN[evnnni] <<" ";
    std::cout<< std::endl;


    // ENTER OPTIMIZATION LOOP
    for (int fuI = 1; fuI <= arg_fuIter; fuI++) {
    	std::cout <<"Full Update - STEP "<< fuI << std::endl;

        // PERFORM FULL UPDATE
        std::cout << "GATE: " << (fuI-1)%gates.size() << std::endl;
            
        diag_fu = fullUpdate_2site_v2(*(ptr_gateMPO[(fuI-1)%gates.size()]), cls, ctmEnv, 
            gates[(fuI-1)%gates.size()], gate_auxInds[(fuI-1)%gates.size()], 
            iso_store[(fuI-1)%gates.size()], fuArgs);

        diagData_fu.push_back(diag_fu);

        out_file_diag << fuI <<" "<< diag_ctmIter.back() <<" "<< diag_fu.getInt("alsSweep",0)
            <<" "<< diag_fu.getString("siteMaxElem")
            <<" "<< diag_fu.getReal("finalDist0",0.0)
            <<" "<< diag_fu.getReal("finalDist1",0.0);
        out_file_diag <<" "<< diag_fu.getReal("ratioNonSymLE",0.0)
            <<" "<< diag_fu.getReal("ratioNonSymFN",0.0);
        out_file_diag <<" "<< diag_fu.getReal("minGapDisc",0.0) 
            <<" "<< diag_fu.getReal("minEvKept",0.0);
        out_file_diag  <<std::endl;

        ctmEnv.updateCluster(cls);
        ev.setCluster(cls);
        writeCluster(outClusterFile, cls);
        
        // SETUP ENVIRONMENT LOOP
        accT = std::vector<double>(8,0.0);
        // reset environment
        if (arg_reinitEnv) 
            switch (arg_initEnvType) {
                case CtmEnv::INIT_ENV_const1: {
                    ctmEnv.initMockEnv();
                    break;
                }
                case CtmEnv::INIT_ENV_ctmrg: {
                    ctmEnv.initCtmrgEnv();
                    //ctmEnv.symmetrizeEnv(arg_fuDbg);
                    break;
                }
                case CtmEnv::INIT_ENV_rnd: {
                    ctmEnv.initRndEnv(envIsComplex);
                    ctmEnv.symmetrizeEnv();
                    break;
                } 
            }
		
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

        if (fuI % arg_obsFreq == 0) {
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
        }

        // write energy
        double avgE_8links = 0.;
        out_file_energy << fuI-1 <<" "<< e_nnH.back() 
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
    }

    // FULL UPDATE FINISHED - COMPUTING FINAL ENVIRONMENT
    std::cout <<"FULL UPDATE DONE - COMPUTING FINAL ENVIRONMENT "<< std::endl;
    // t_begin_int = std::chrono::steady_clock::now();
    
    // reset environment
    if (arg_reinitEnv) 
        switch (arg_initEnvType) {
            case CtmEnv::INIT_ENV_const1: {
                ctmEnv.initMockEnv();
                break;
            }
            case CtmEnv::INIT_ENV_ctmrg: {
                ctmEnv.initCtmrgEnv();
                //ctmEnv.symmetrizeEnv(arg_fuDbg);
                break;
            }
            case CtmEnv::INIT_ENV_rnd: {
                ctmEnv.initRndEnv(envIsComplex);
                ctmEnv.symmetrizeEnv();
                break;
            } 
        }

    // ENTER ENVIRONMENT LOOP
    for (int envI=1; envI<=arg_maxInitEnvIter; envI++ ) {

        ctmEnv.insLCol_DBG(iso_type, norm_type, accT);
        ctmEnv.insRCol_DBG(iso_type, norm_type, accT);
        ctmEnv.insURow_DBG(iso_type, norm_type, accT);
        ctmEnv.insDRow_DBG(iso_type, norm_type, accT);

        if ( envI % 1 == 0 ) {
            ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());
        
            t_end_int = std::chrono::steady_clock::now();
            std::cout << "CTM STEP " << envI <<" T: "<< std::chrono::duration_cast
                <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
                <<" [sec] E: "<< e_curr[0] <<" "<< e_curr[1] <<" "<< e_curr[2] <<" "
                << e_curr[3] << std::endl;
            
            e_curr[0]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,0), std::make_pair(1,0));
            e_curr[1]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,0), std::make_pair(0,1));
            e_curr[2]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(1,0), std::make_pair(1,1));
            e_curr[3]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,1), std::make_pair(1,1));

            if ((std::abs(e_prev[0]-e_curr[0]) < arg_envEps) &&
                (std::abs(e_prev[1]-e_curr[1]) < arg_envEps) &&
                (std::abs(e_prev[2]-e_curr[2]) < arg_envEps) &&
                (std::abs(e_prev[3]-e_curr[3]) < arg_envEps) ) {

                std::cout<< "ENV CONVERGED" << std::endl;
                break;
            }

            e_prev = e_curr;
            t_begin_int = std::chrono::steady_clock::now();
        }
    }

    std::cout <<"accT [mSec]: "<< accT[0] <<" "<< accT[1] <<" "<< accT[2]
        <<" "<< accT[3] << std::endl;
    std::cout <<"isoZ [mSec]: "<< accT[4] <<" "<< accT[5] <<" "<< accT[6]
        <<" "<< accT[7] << std::endl;

    ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());

	// Compute final properties
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

    // compute energies NNN links
    evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(0,0)) );
    evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(1,1)) );
    evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(1,0)) );
    evNNN.push_back( ev.eval2x2Diag11(EVBuilder::OP2S_SS, std::make_pair(2,1)) );

    evNNN.push_back( ev.eval2x2DiagN11(EVBuilder::OP2S_SS, std::make_pair(0,0)) );
    evNNN.push_back( ev.eval2x2DiagN11(EVBuilder::OP2S_SS, std::make_pair(1,1)) );
    evNNN.push_back( ev.eval2x2DiagN11(EVBuilder::OP2S_SS, std::make_pair(1,0)) );
    evNNN.push_back( ev.eval2x2DiagN11(EVBuilder::OP2S_SS, std::make_pair(0,1)) );

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

    // write energy
    avgE_8links = 0.;
    out_file_energy << arg_fuIter <<" "<< e_nnH.back() 
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
    
    evNNN_avg = 0.;
    for(int evnnni=evNNN.size()-8; evnnni < evNNN.size(); evnnni++)   
        evNNN_avg += evNNN[evnnni];
    evNNN_avg = evNNN_avg / 8.0;

    out_file_energy <<" "<< evNNN_avg;
    //out_file_energy <<" "<< avgE_8links + arg_J2*evNNN_avg;
    
    // write magnetization
    evMag_avg = 0.25*(
        sqrt(sA_zpm[0]*sA_zpm[0]) //+ sA_zpm[1]*sA_zpm[1] )//+ sA_zpm[2]*sA_zpm[2])
        + sqrt(sB_zpm[0]*sB_zpm[0]) //+ sB_zpm[1]*sB_zpm[1] )//+ sB_zpm[2]*sB_zpm[2])
        + sqrt(sC_zpm[0]*sC_zpm[0]) //+ sC_zpm[1]*sC_zpm[1] )//+ sC_zpm[2]*sC_zpm[2])
        + sqrt(sD_zpm[0]*sD_zpm[0]) //+ sD_zpm[1]*sD_zpm[1] )//+ sD_zpm[2]*sD_zpm[2])
        );
    out_file_energy <<" "<< evMag_avg;

    evMag_avg = 0.25*(
        sqrt(sA_zpm[1]*sA_zpm[1] )//+ sA_zpm[2]*sA_zpm[2])
        + sqrt(sB_zpm[1]*sB_zpm[1] )//+ sB_zpm[2]*sB_zpm[2])
        + sqrt(sC_zpm[1]*sC_zpm[1] )//+ sC_zpm[2]*sC_zpm[2])
        + sqrt(sD_zpm[1]*sD_zpm[1] )//+ sD_zpm[2]*sD_zpm[2])
        );
    out_file_energy <<" "<< evMag_avg;

    out_file_energy << std::endl;

    std::cout <<"FU_ITER: "<<" E:"<< std::endl;
    for ( unsigned int i=0; i<e_nnH.size(); i++ ) {
        std::cout << i <<" "<< e_nnH[i] 
            <<" "<< e_nnH_AC[i]
            <<" "<< e_nnH_BD[i]
            <<" "<< e_nnH_CD[i];
            for ( unsigned int j=i*4; j<(i+1)*4; j++ ) {
                std::cout<<" "<< evNN[j];
            }
        std::cout<< std::endl;
    }
    for(int evnnni=evNNN.size()-8; evnnni < evNNN.size(); evnnni++)   
        std::cout << evNNN[evnnni] <<" ";
    std::cout<< std::endl;  

    std::cout <<"ID: " << ev.eV_1sO_1sENV(EVBuilder::MPO_Id, std::make_pair(0,0)) << std::endl;
    std::cout <<"SZ2: "<< ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z2, std::make_pair(0,0)) << std::endl;

    // Store final new cluster
    writeCluster(outClusterFile, cls);

    // write out diagnostic data of full-update
    for (int i=0; i<arg_fuIter; i++) {
        std::cout<< i <<" "<< diag_ctmIter[i] <<" "<< diagData_fu[i].getInt("alsSweep",0)
            <<" "<< diagData_fu[i].getString("siteMaxElem")
            <<" "<< diagData_fu[i].getReal("finalDist0",0.0)
            <<" "<< diagData_fu[i].getReal("finalDist1",0.0);
        if (arg_fuDbg && (arg_fuDbgLevel >=1))
            std::cout<<" "<< diagData_fu[i].getReal("ratioNonSymLE",0.0)
            <<" "<< diagData_fu[i].getReal("ratioNonSymFN",0.0);
        std::cout<<std::endl;
    }

    ctmEnv.computeSVDspec();
    ctmEnv.printSVDspec();
}