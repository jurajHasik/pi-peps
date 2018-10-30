#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
// #include "omp.h"
// #include "mkl.h"
#include "json.hpp"
#include "ctm-cluster-io.h"
#include "ctm-cluster-env_v2.h"
#include "cluster-ev-builder.h"
//#include "simple-update_v2.h"
//#include "full-update.h"
//#include "full-update-TEST.h"
#include "mpo.h"
#include "models.h"
#include "engine.h"

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
	double initStateNoise = jsonCls.value("initStateNoise",1.0e-16);
    physDim = jsonCls["physDim"].get<int>();
	auxBondDim = jsonCls["auxBondDim"].get<int>();

	// read cluster outfile
	std::string outClusterFile(jsonCls["outClusterFile"].get<std::string>());

	// read Hamiltonian and Trotter decomposition
    auto json_model_params(jsonCls["model"]);
    std::string arg_modelType = json_model_params.value("modelType","UNSUPPORTED"); 
    bool symmTrotter  = json_model_params.value("symmTrotter",true);
    bool randomizeSeq = json_model_params.value("randomizeSeq",false);

	// full update parameters
    int arg_fuIter  = jsonCls["fuIter"].get<int>();
    int arg_obsFreq = jsonCls["obsFreq"].get<int>();
    bool arg_fuDbg = jsonCls["fuDbg"].get<bool>();
    int arg_fuDbgLevel = jsonCls["fuDbgLevel"].get<int>();
    std::string arg_otNormType = jsonCls["otNormType"].get<std::string>();


    int arg_maxAltLstSqrIter = jsonCls["maxAltLstSqrIter"].get<int>();
    double isoEpsilon = jsonCls["isoEpsilon"].get<double>();
    std::string arg_fuIsoInit = jsonCls["fuIsoInit"].get<std::string>();
    double arg_fuIsoInitNoiseLevel = jsonCls["fuIsoInitNoiseLevel"].get<double>();
    bool symmetrizeProtoEnv = jsonCls["symmetrizeProtoEnv"].get<bool>();
    bool posDefProtoEnv = jsonCls["positiveDefiniteProtoEnv"].get<bool>();
    // Iterative ALS procedure
    double epsdistf = jsonCls.value("epsdistf",1.0e-8);
    auto json_als_params(jsonCls["als"]);
    std::string solver = json_als_params.value("solver","UNSUPPORTED");
    bool solver_dbg    = json_als_params.value("dbg",false);
    double epsregularisation = json_als_params.value("epsregularisation",1.0e-8);

    // BiCG params
    double cg_convergence_check = json_als_params.value("cg_convergence_check",1);
    double cg_gradientNorm_eps  = json_als_params.value("cg_gradientNorm_eps",1.0e-8);
    int cg_max_iter             = json_als_params.value("cg_max_iter",0);

    // direct linear solver params
    std::string als_ds_method   = json_als_params.value("method","LU");

    // pseudoinverse solver params
    double pseudoInvCutoff       = json_als_params.value("pseudoInvCutoff",1.0e-8);
    double pseudoInvCutoffInsert = json_als_params.value("pseudoInvCutoffInsert",0.0);

    // legacy
    //double pseudoInvMaxLogGap = jsonCls["pseudoInvMaxLogGap"].get<double>();
    bool dynamicEps = jsonCls.value("dynamicEps",false);
    
	
    // gauge Fixing by simple update with identity operators
    bool arg_su_gauge_fix = jsonCls.value("suGaugeFix",false);
    int arg_su_gauge_fix_freq = jsonCls.value("suGaugeFixFreq",arg_obsFreq);
    auto json_gauge_fix_params(jsonCls["gaugeFix"]);
    std::string arg_suWeightsInit = json_gauge_fix_params.value("suWeightsInit","DELTA");
    int arg_suIter = json_gauge_fix_params.value("suIter",128);
    bool arg_gf_dbg = json_gauge_fix_params.value("suDbg",false);
    int arg_gf_dbgLvl = json_gauge_fix_params.value("suDbgLevel",0);
    

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
    auto rsvd_power   = jsonCls.value("rsvd_power",2);
    auto rsvd_reortho = jsonCls.value("rsvd_reortho",1);
	int arg_maxEnvIter = json_ctmrg_params["maxEnvIter"].get<int>();
    int arg_maxInitEnvIter = json_ctmrg_params["initMaxEnvIter"].get<int>();
    int arg_obsMaxIter = jsonCls.value("obsMaxIter",arg_maxInitEnvIter);
    double arg_envEps  = json_ctmrg_params["envEpsilon"].get<double>();
    bool arg_reinitEnv = json_ctmrg_params["reinitEnv"].get<bool>();
    bool arg_reinitObsEnv = json_ctmrg_params.value("reinitObsEnv",false);
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

        // TEST - add small elements instead of 0
        double eps = initStateNoise;
        auto addEpsilon = [&eps](Real r) { return (std::abs(r) > eps) ? r : r + eps; };
        A.apply(addEpsilon);
        B.apply(addEpsilon);
        C.apply(addEpsilon);
        D.apply(addEpsilon);

        cls.aux  = {aIA, aIB, aIC, aID};
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
        cls.SI = { {"A",0}, {"B",1}, {"C",2}, {"D",3} };

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
        } else if (initBy == "ZPRST") {
            std::cout <<"Initializing by PRODUCT STATE along Z +1/2"<< std::endl;
            // Spin UP on all sites
            A.set(aIA(1), prime(aIA,1)(1), prime(aIA,2)(1), prime(aIA,3)(1),
                pIA(1), 1.0);
            B.set(aIB(1), prime(aIB,1)(1), prime(aIB,2)(1), prime(aIB,3)(1),
                pIB(1), 1.0);
            C.set(aIC(1), prime(aIC,1)(1), prime(aIC,2)(1), prime(aIC,3)(1),
                pIC(1), 1.0);
            D.set(aID(1), prime(aID,1)(1), prime(aID,2)(1), prime(aID,3)(1),
                pID(1), 1.0);
        } else {
            std::cout <<"Unsupported cluster initialization: "<< initBy << std::endl;
        }

        cls.aux  = {aIA, aIB, aIC, aID};
        cls.phys = {pIA, pIB, pIC, pID};

        cls.sites = {{"A", A}, {"B", B}, {"C",C}, {"D",D}};

        // Define siteToWeights
        cls.siteToWeights["A"] = {
            {{"A","B"},{2,0},"L1"},
            {{"A","B"},{0,2},"L2"},
            {{"A","C"},{1,3},"L3"},
            {{"A","C"},{3,1},"L4"}
        };
        cls.siteToWeights["B"] = {
            {{"B","A"},{2,0},"L2"},
            {{"B","A"},{0,2},"L1"},
            {{"B","D"},{1,3},"L5"},
            {{"B","D"},{3,1},"L6"}
        };
        cls.siteToWeights["C"] = {
            {{"C","D"},{2,0},"L7"},
            {{"C","D"},{0,2},"L8"},
            {{"C","A"},{1,3},"L4"},
            {{"C","A"},{3,1},"L3"}
        };
        cls.siteToWeights["D"] = {
            {{"D","B"},{3,1},"L5"},
            {{"D","B"},{1,3},"L6"},
            {{"D","C"},{2,0},"L8"},
            {{"D","C"},{0,2},"L7"}
        };
        // ----- END DEFINE CLUSTER ------------------------------------
    }

    // INITIALIZE ENVIRONMENT
    CtmEnv ctmEnv(arg_ioEnvTag, auxEnvDim, cls, 
        {"isoPseudoInvCutoff",arg_isoPseudoInvCutoff,
         "SVD_METHOD",env_SVD_METHOD,
         "rsvd_power",rsvd_power,
         "rsvd_reortho",rsvd_reortho,
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
        case CtmEnv::INIT_ENV_obc: {
            ctmEnv.initOBCEnv();
            break;
        }
        case CtmEnv::INIT_ENV_pwr: {
            ctmEnv.initPWREnv();
            break;
        }
        case CtmEnv::INIT_ENV_rnd: {
           	ctmEnv.initRndEnv(envIsComplex);
            break;
        } 
        default: {
            std::cout<<"Unsupported INIT_ENV" << std::endl;
            exit(EXIT_FAILURE);   
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

    // hold ctm observables used in the convergence criterion
    std::vector<double> e_curr(4,0.0), e_prev(4,0.0);

    std::vector<double> accT(8,0.0); // holds timings for CTM moves
    std::chrono::steady_clock::time_point t_begin_int, t_end_int;

    // ##################################################################
    // # SETUP OPTIMIZATION LOOP                                        #
    // ##################################################################

    // DEFINE MODEL AND GATE SEQUENCE
    std::unique_ptr<Model>  ptr_model;
    std::unique_ptr<Engine> ptr_engine;     // full update engine
    std::unique_ptr<Engine> ptr_gfe;        // gauge fixing engine

    // // randomisation
    // std::vector<int> rndInds;
    // std::vector< OpNS * >              tmp_ptr_gateMPO;
    // std::vector< std::vector<std::string> > tmp_gates;
    // std::vector< std::vector<int> >         tmp_gate_auxInds;

    ptr_model =  getModel(json_model_params);
    ptr_engine = buildEngine(json_model_params);
    
    if (ptr_engine) { 
        std::cout<<"Valid Engine constructed"<<std::endl;
    } else {
        std::cout<<"Engine pointer is NULL"<<std::endl;
        exit(EXIT_FAILURE);
    }
    if (arg_su_gauge_fix) {
        ptr_gfe = buildEngine(json_gauge_fix_params);
        if (ptr_gfe) {
            std::cout<<"Valid GaugeFix Engine constructed"<<std::endl;
        } else {
            std::cout<<"GaugeFix Engine pointer is NULL"<<std::endl;
            exit(EXIT_FAILURE);
        }
    }
    
    // // randomisation
    // if ( randomizeSeq && (symmTrotter==false) ) {
    //     for ( int i=0; i < gates.size(); i++ ) rndInds.push_back(i);
    //     tmp_ptr_gateMPO  = ptr_gateMPO;
    //     tmp_gates        = gates;
    //     tmp_gate_auxInds = gate_auxInds;
    // }

    // STORE ISOMETRIES
    // std::vector< std::vector< ITensor > > iso_store(
        // ptr_model->gates.size(), {ITensor(), ITensor(), ITensor(), ITensor()} );
     std::vector< std::vector< ITensor > > iso_store(
        1, {ITensor(), ITensor(), ITensor(), ITensor()} );

    Args fuArgs = {
        "maxAltLstSqrIter",arg_maxAltLstSqrIter,
        "fuDbg",arg_fuDbg,
        "fuDbgLevel",arg_fuDbgLevel,
        "symmetrizeProtoEnv",symmetrizeProtoEnv,
        "positiveDefiniteProtoEnv",posDefProtoEnv,
        "otNormType",arg_otNormType,
        "epsdistf",epsdistf,

        "isoEpsilon",isoEpsilon,
        "fuIsoInit",arg_fuIsoInit,
        "fuIsoInitNoiseLevel",arg_fuIsoInitNoiseLevel,

        "solver",solver,
        "dbg",solver_dbg,
        "epsregularisation",epsregularisation,

        "cg_convergence_check",cg_convergence_check,
        "cg_gradientNorm_eps",cg_gradientNorm_eps,
        "cg_max_iter",cg_max_iter,

        "method",als_ds_method,

        "pseudoInvCutoff",pseudoInvCutoff,
        "pseudoInvCutoffInsert",pseudoInvCutoffInsert,
        
        //"dynamicEps",dynamicEps,
        //"pseudoInvMaxLogGap",pseudoInvMaxLogGap
    };
    // Diagnostic data
    std::vector<int> diag_ctmIter;
    std::vector< Args > diagData_fu;
    std::vector<double> diag_minCornerSV(1, 0.);
    double best_energy = 1.0e+16;
    Args diag_fu;

    std::string outClusterBestFile = outClusterFile+".best";
    std::ofstream out_file_energy(outClusterFile+".energy.dat", std::ios::out);
    std::ofstream out_file_diag(outClusterFile+".diag.dat", std::ios::out);
    out_file_energy.precision( std::numeric_limits< double >::max_digits10 );
    out_file_diag.precision( std::numeric_limits< double >::max_digits10 );

    // ***** COMPUTING INITIAL ENVIRONMENT ************************************
    std::cout <<"COMPUTING INITIAL ENVIRONMENT "<< std::endl;
    // t_begin_int = std::chrono::steady_clock::now();
        
    // ENTER ENVIRONMENT LOOP
    bool expValEnvConv = false;
    for (int envI=1; envI<=arg_maxInitEnvIter; envI++ ) {

        t_begin_int = std::chrono::steady_clock::now();

        ctmEnv.insLCol_DBG(iso_type, norm_type, accT);
        ctmEnv.insRCol_DBG(iso_type, norm_type, accT);
        ctmEnv.insURow_DBG(iso_type, norm_type, accT);
        ctmEnv.insDRow_DBG(iso_type, norm_type, accT);

        t_end_int = std::chrono::steady_clock::now();
        std::cout << "CTM STEP " << envI <<" T: "<< std::chrono::duration_cast
                <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
                <<" [sec] ";

        if ( envI % 1 == 0 ) {
            t_begin_int = std::chrono::steady_clock::now();

            ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());
            
            e_curr[0]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,0), std::make_pair(1,0));
            e_curr[1]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,0), std::make_pair(0,1));
            e_curr[2]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(1,0), std::make_pair(1,1));
            e_curr[3]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,1), std::make_pair(1,1));

            // e_curr[0]=ev.evalSS(std::make_pair(0,0), std::make_pair(1,0));
            // e_curr[1]=ev.evalSS(std::make_pair(0,0), std::make_pair(0,1));
            // e_curr[2]=ev.evalSS(std::make_pair(1,0), std::make_pair(1,1));
            // e_curr[3]=ev.evalSS(std::make_pair(0,1), std::make_pair(1,1));  

            // e_curr[0]=ev.eval2Smpo_redDenMat2x1(EVBuilder::OP2S_SS, std::make_pair(0,0), std::make_pair(1,0));
            // e_curr[1]=ev.eval2Smpo_redDenMat2x1(EVBuilder::OP2S_SS, std::make_pair(0,0), std::make_pair(0,1));
            // e_curr[2]=ev.eval2Smpo_redDenMat2x1(EVBuilder::OP2S_SS, std::make_pair(1,0), std::make_pair(1,1));
            // e_curr[3]=ev.eval2Smpo_redDenMat2x1(EVBuilder::OP2S_SS, std::make_pair(0,1), std::make_pair(1,1)); 

            t_end_int = std::chrono::steady_clock::now();
            std::cout <<"|| T: "<< std::chrono::duration_cast
                <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
                <<" [sec] E: "<< e_curr[0] <<" "<< e_curr[1] <<" "<< e_curr[2] <<" "
                << e_curr[3] << std::endl;

            // e_curr[0]=ev.eval2Smpo(EVBuilder::OP2S_SZSZ, std::make_pair(0,0), std::make_pair(1,0));
            // e_curr[1]=ev.eval2Smpo(EVBuilder::OP2S_SZSZ, std::make_pair(0,0), std::make_pair(0,1));
            // e_curr[2]=ev.eval2Smpo(EVBuilder::OP2S_SZSZ, std::make_pair(1,0), std::make_pair(1,1));
            // e_curr[3]=ev.eval2Smpo(EVBuilder::OP2S_SZSZ, std::make_pair(0,1), std::make_pair(1,1));

            if ((std::abs(e_prev[0]-e_curr[0]) < arg_envEps) &&
                (std::abs(e_prev[1]-e_curr[1]) < arg_envEps) &&
                (std::abs(e_prev[2]-e_curr[2]) < arg_envEps) &&
                (std::abs(e_prev[3]-e_curr[3]) < arg_envEps) ) {

                diag_ctmIter.push_back(envI);
                std::cout<< "INIT ENV CONVERGED" << std::endl;
                expValEnvConv = true;
            }

            if (envI==arg_maxInitEnvIter) {
                std::cout<< " MAX ENV iterations REACHED ";
                expValEnvConv = true;
            }
            e_prev = e_curr;

            if (expValEnvConv) {
                diag_ctmIter.push_back(envI);
            
                // diagnose spectra
                std::ostringstream oss;
                oss << std::scientific;

                // diagnose spectra
                std::cout << std::endl;
                double tmpVal;
                double minCornerSV = 1.0e+16;
                Args args_dbg_cornerSVD = {"Truncate",false};
                std::cout << "Spectra: " << std::endl;

                ITensor tL(ctmEnv.C_LU[0].inds().front()),sv,tR;
                auto spec = svd(ctmEnv.C_LU[0],tL,sv,tR, args_dbg_cornerSVD);
                tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                    sv.inds().back()(auxEnvDim));
                PrintData(sv);
                minCornerSV = std::min(minCornerSV, tmpVal);
                oss << tmpVal;

                tL = ITensor(ctmEnv.C_RU[0].inds().front());
                spec = svd(ctmEnv.C_RU[0],tL,sv,tR, args_dbg_cornerSVD);
                tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                    sv.inds().back()(auxEnvDim));
                PrintData(sv);
                minCornerSV = std::min(minCornerSV, tmpVal);
                oss <<" "<< tmpVal;

                tL = ITensor(ctmEnv.C_RD[0].inds().front());
                spec = svd(ctmEnv.C_RD[0],tL,sv,tR, args_dbg_cornerSVD);
                tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                    sv.inds().back()(auxEnvDim));
                PrintData(sv);
                minCornerSV = std::min(minCornerSV, tmpVal);
                oss <<" "<< tmpVal;

                tL = ITensor(ctmEnv.C_LD[0].inds().front());
                spec = svd(ctmEnv.C_LD[0],tL,sv,tR, args_dbg_cornerSVD);
                tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                    sv.inds().back()(auxEnvDim));
                PrintData(sv);
                minCornerSV = std::min(minCornerSV, tmpVal);
                oss <<" "<< tmpVal;

                diag_minCornerSV.push_back(minCornerSV);
                std::cout << "MinVals: "<< oss.str() << std::endl;
            
                break;
            }
        }
    }
    // ***** COMPUTING INITIAL ENVIRONMENT DONE *******************************

    ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());
    // Compute initial properties
    ptr_model->setObservablesHeader(out_file_energy);
    auto obs_metaInf = Args("lineNo",0);
    ptr_model->computeAndWriteObservables(ev, out_file_energy, obs_metaInf);
    best_energy = obs_metaInf.getReal("energy");

    // ENTER OPTIMIZATION LOOP
    for (int fuI = 1; fuI <= arg_fuIter; fuI++) {
    	std::cout <<"Full Update - STEP "<< fuI << std::endl;

        // // randomisation
        // if ( randomizeSeq && (symmTrotter==false) && (fuI % gates.size() == 0) ) {
        //     std::cout <<"Randomizing gate sequence"<< std::endl;
        //     std::random_shuffle( rndInds.begin(), rndInds.end() );
        //     for ( int i=0; i < gates.size(); i++ ) { 
        //         ptr_gateMPO[i]  = tmp_ptr_gateMPO[rndInds[i]];
        //         gates[i]        = tmp_gates[rndInds[i]];
        //         gate_auxInds[i] = tmp_gate_auxInds[rndInds[i]];
        //     }
        // }


        // PERFORM FULL UPDATE
        //std::cout << "GATE: " << (fuI-1)%ptr_model->gates.size() << std::endl;
        
        // diag_fu = fullUpdate_ALS_LSCG_IT(*(ptr_gateMPO[(fuI-1)%gates.size()]), cls, ctmEnv, 
        //     gates[(fuI-1)%gates.size()], gate_auxInds[(fuI-1)%gates.size()], 
        //     iso_store[(fuI-1)%gates.size()], fuArgs);

        // diag_fu = fullUpdate_CG_IT(*(ptr_gateMPO[(fuI-1)%gates.size()]), cls, ctmEnv, 
        //     gates[(fuI-1)%gates.size()], gate_auxInds[(fuI-1)%gates.size()], fuArgs);

        // diag_fu = fullUpdate_2site_PINV( *(ptr_gateMPO[(fuI-1)%gates.size()]), cls, ctmEnv, 
        //    gates[(fuI-1)%gates.size()], gate_auxInds[(fuI-1)%gates.size()], fuArgs);

        //auto temp_ptr =  (MPO_3site *) ptr_gateMPO[(fuI-1)%gates.size()];

        // diag_fu = fullUpdate_2site_v2( 
        //     ptr_model->ptr_gateMPO[(fuI-1)%ptr_model->gates.size()], 
        //     cls, 
        //     ctmEnv, gates[(fuI-1)%ptr_model->gates.size()], 
        //     gate_auxInds[(fuI-1)%ptr_model->gates.size()],
        //     iso_store[(fuI-1)%ptr_model->gates.size()], fuArgs);

        // ctmEnv.symmetrizeEnv();
        diag_fu = ptr_engine->performFullUpdate(cls, ctmEnv, fuArgs);

        diagData_fu.push_back(diag_fu);

        if ( fuI == 1 ) {
            out_file_diag << "fuI ctmIter alsSweep" <<" ";
            out_file_diag << diag_fu.getString("siteMaxElem_descriptor") << " ";
            if ( diag_fu.getString("locMinDiag","").length() > 0 )
                out_file_diag << diag_fu.getString("locMinDiag_descriptor") << " ";
            if ( diag_fu.getString("diag_protoEnv","").length() > 0 ) 
                out_file_diag << diag_fu.getString("diag_protoEnv_descriptor") << " ";
            out_file_diag << "MinCornerSV" << " ";
            out_file_diag << std::endl;
        }

        out_file_diag << fuI <<" "<< diag_ctmIter.back() <<" "<< diag_fu.getInt("alsSweep",0)
            <<" "<< diag_fu.getString("siteMaxElem");
        if ( diag_fu.getString("locMinDiag","").length() > 0 )     
            out_file_diag <<" "<< diag_fu.getString("locMinDiag","");
        if ( diag_fu.getString("diag_protoEnv","").length() > 0 ) 
            out_file_diag << " " << diag_fu.getString("diag_protoEnv","");
        out_file_diag <<" "<< diag_minCornerSV.back();
        out_file_diag <<" "<< diag_fu.getReal("ratioNonSymLE",0.0)
            <<" "<< diag_fu.getReal("ratioNonSymFN",0.0);
        out_file_diag <<" "<< diag_fu.getReal("minGapDisc",0.0) 
            <<" "<< diag_fu.getReal("minEvKept",0.0);
        out_file_diag  <<std::endl;

        ctmEnv.updateCluster(cls);
        ev.setCluster(cls);
        // writeCluster(outClusterFile, cls);
        
        // fix gauge by simple-update at dt=0 - identity operators
        if ( arg_su_gauge_fix && (fuI % arg_su_gauge_fix_freq == 0) ) {
            std::cout << "GAUGE FIXING" << std::endl;
            auto num_eps = std::numeric_limits<double>::epsilon();
            t_begin_int = std::chrono::steady_clock::now();

            // Assuming the weights have been initialized
            initClusterWeights(cls);
            setWeights(cls, arg_suWeightsInit);

            Args gfArgs = {"suDbg",arg_gf_dbg,"suDbgLevel",arg_gf_dbgLvl};
            Args gf_diag_fu;
            saveWeights(cls);
            for (int suI = 1; suI <= arg_suIter; suI++) {
                //std::cout <<"Simple Update - STEP "<< suI << std::endl;

                // PERFORM SIMPLE UPDATE
                //std::cout << "GATE: " << (suI-1)%gates.size() << std::endl;

                gf_diag_fu = ptr_gfe->performSimpleUpdate(cls, gfArgs);

                //diagData_fu.push_back(diag_fu);
                
                //check convergence
                if (suI % 8 == 0) {
                    auto weight_distance = weightDist(cls);
                    if ( weight_distance < num_eps*cls.auxBondDim*cls.weights.size() ) {
                        std::cout<<"GF iter: "<< suI <<" dist: "<< weight_distance 
                            <<" CONVERGED"<< std::endl;
                        break;
                    } else {
                        std::cout<<"GF iter: "<< suI <<" dist: "<< weight_distance << std::endl;
                    }
                    saveWeights(cls);
                }
            }

            absorbWeightsToSites(cls);
        
            t_end_int = std::chrono::steady_clock::now();
            std::cout << "GUAGE FIX DONE" << " T: "<< std::chrono::duration_cast
                    <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
                    <<" [sec] "; 
        }

        // SETUP ENVIRONMENT LOOP
        accT = std::vector<double>(8,0.0);
        // reset environment
        if (arg_reinitEnv || ((fuI % arg_obsFreq == 0) && arg_reinitObsEnv) ) 
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
                case CtmEnv::INIT_ENV_obc: {
                    ctmEnv.initOBCEnv();
                    break;
                }
                case CtmEnv::INIT_ENV_pwr: {
                    ctmEnv.initPWREnv();
                    break;
                }
                case CtmEnv::INIT_ENV_rnd: {
                    ctmEnv.initRndEnv(envIsComplex);
                    ctmEnv.symmetrizeEnv();
                    break;
                } 
            }
		
    	// ENTER ENVIRONMENT LOOP
        int currentMaxEnvIter = (fuI % arg_obsFreq == 0) ? arg_obsMaxIter : arg_maxEnvIter; 
		bool expValEnvConv = false;
        for (int envI=1; envI<=currentMaxEnvIter; envI++ ) {
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

	        if ( (currentMaxEnvIter > 1) && (envI % 1 == 0) ) {
                t_begin_int = std::chrono::steady_clock::now();
	            
                ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());

                e_curr[0]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,0), std::make_pair(1,0));
                e_curr[1]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,0), std::make_pair(0,1));
                e_curr[2]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(1,0), std::make_pair(1,1));
                e_curr[3]=ev.eval2Smpo(EVBuilder::OP2S_SS, std::make_pair(0,1), std::make_pair(1,1));

                // e_curr[0]=ev.eval2Smpo(EVBuilder::OP2S_SZSZ, std::make_pair(0,0), std::make_pair(1,0));
                // e_curr[1]=ev.eval2Smpo(EVBuilder::OP2S_SZSZ, std::make_pair(0,0), std::make_pair(0,1));
                // e_curr[2]=ev.eval2Smpo(EVBuilder::OP2S_SZSZ, std::make_pair(1,0), std::make_pair(1,1));
                // e_curr[3]=ev.eval2Smpo(EVBuilder::OP2S_SZSZ, std::make_pair(0,1), std::make_pair(1,1));

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

                    std::cout<< " ENV CONVERGED ";
                    expValEnvConv = true;
                }

                // if max number of iterations has been reached
                if ( envI==currentMaxEnvIter )  {
                    std::cout<< " MAX ENV iterations REACHED ";
                    expValEnvConv = true;
                }
                e_prev = e_curr;

                if (expValEnvConv) {
                    diag_ctmIter.push_back(envI);

                    std::ostringstream oss;
                    oss << std::scientific;

                    // diagnose spectra
                    std::cout << std::endl;
                    Args args_dbg_cornerSVD = {"Truncate",false};
                    double tmpVal;
                    double minCornerSV = 1.0e+16;
                    std::cout << "Spectra: " << std::endl;

                    ITensor tL(ctmEnv.C_LU[0].inds().front()),sv,tR;
                    auto spec = svd(ctmEnv.C_LU[0],tL,sv,tR,args_dbg_cornerSVD);
                    tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                        sv.inds().back()(auxEnvDim));
                    Print(sv);
                    for(int i=1; i<=auxEnvDim; i++) std::cout<< i <<" "<< sv.real(sv.inds().front()(i),
                        sv.inds().back()(i)) << std::endl;
                    minCornerSV = std::min(minCornerSV, tmpVal);
                    oss << tmpVal;

                    tL = ITensor(ctmEnv.C_RU[0].inds().front());
                    spec = svd(ctmEnv.C_RU[0],tL,sv,tR,args_dbg_cornerSVD);
                    tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                        sv.inds().back()(auxEnvDim));
                    Print(sv);
                    for(int i=1; i<=auxEnvDim; i++) std::cout<< i <<" "<< sv.real(sv.inds().front()(i),
                        sv.inds().back()(i)) << std::endl;
                    minCornerSV = std::min(minCornerSV, tmpVal);
                    oss <<" "<< tmpVal;

                    tL = ITensor(ctmEnv.C_RD[0].inds().front());
                    spec = svd(ctmEnv.C_RD[0],tL,sv,tR,args_dbg_cornerSVD);
                    tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                        sv.inds().back()(auxEnvDim));
                    Print(sv);
                    for(int i=1; i<=auxEnvDim; i++) std::cout<< i <<" "<< sv.real(sv.inds().front()(i),
                        sv.inds().back()(i)) << std::endl;
                    minCornerSV = std::min(minCornerSV, tmpVal);
                    oss <<" "<< tmpVal;

                    tL = ITensor(ctmEnv.C_LD[0].inds().front());
                    spec = svd(ctmEnv.C_LD[0],tL,sv,tR,args_dbg_cornerSVD);
                    tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                        sv.inds().back()(auxEnvDim));
                    Print(sv);
                    for(int i=1; i<=auxEnvDim; i++) std::cout<< i <<" "<< sv.real(sv.inds().front()(i),
                        sv.inds().back()(i)) << std::endl;
                    minCornerSV = std::min(minCornerSV, tmpVal);
                    oss <<" "<< tmpVal;

                    diag_minCornerSV.push_back(minCornerSV);
                    std::cout << "MinVals: "<< oss.str() << std::endl;

                    break;
                }

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

            // ctmEnv.symmetrizeEnv();
            auto metaInf = Args("lineNo",fuI);
            ptr_model->computeAndWriteObservables(ev,out_file_energy,metaInf);

            // check energy, preserve the best_energy state obtained so far
            auto current_energy = metaInf.getReal("energy");
            if( best_energy > current_energy ) {
                best_energy = current_energy;
                cls.metaInfo = "BestEnergy(FUStep=" + std::to_string(fuI) + ")";
                writeCluster(outClusterBestFile, cls);
            }

            t_end_int = std::chrono::steady_clock::now();

            std::cout << "Observables computed in T: "<< std::chrono::duration_cast
                    <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
                    <<" [sec] "<< std::endl;

            // t_begin_int = std::chrono::steady_clock::now();
            // t_end_int = std::chrono::steady_clock::now();
            // std::cout << "NN <S.S> computed in T: "<< std::chrono::duration_cast
            //         <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
            //         <<" [sec] "<< std::endl;

            // t_begin_int = std::chrono::steady_clock::now();
            // t_end_int = std::chrono::steady_clock::now();
            // std::cout << "NNN <S.S> computed in T: "<< std::chrono::duration_cast
            //         <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
            //         <<" [sec] "<< std::endl;

            // t_begin_int = std::chrono::steady_clock::now();
            // t_end_int = std::chrono::steady_clock::now();
            // std::cout << "<S> computed in T: "<< std::chrono::duration_cast
            //         <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
            //         <<" [sec] "<< std::endl;
        
            writeCluster(outClusterFile, cls);
        }
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
            case CtmEnv::INIT_ENV_obc: {
                ctmEnv.initOBCEnv();
                break;
            }
            case CtmEnv::INIT_ENV_pwr: {
                ctmEnv.initPWREnv();
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

    obs_metaInf = Args("lineNo",arg_fuIter+1);
	ptr_model->computeAndWriteObservables(ev,out_file_energy,obs_metaInf);

    // COMPUTE CORRELATION FUNCTIONS
    int dist = 40;
    auto site0 = make_pair(0,0);
    auto site1 = make_pair(1,0);

    double sz0 = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, site0);
    double sp0 = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, site0);
    double sm0 = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, site0);
    auto S0S0 = sz0*sz0 + 0.5*(sp0*sm0 + sm0*sp0 );
    std::cout << "S_0 = ( "<< sz0 <<", "<< sp0 <<", "<< sm0 <<")"<< std::endl;

    double sz1 = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, site1);
    double sp1 = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, site1);
    double sm1 = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, site1);
    auto S0S1 = sz0*sz1 + 0.5*(sp0*sm1 + sm0*sp1 );
    std::cout << "S_1 = ( "<< sz1 <<", "<< sp1 <<", "<< sm1 <<")"<< std::endl;

    std::vector<double> SS_disconnected = {S0S0,S0S1};

    auto szsz = ev.expVal_1sO1sO_H( 
        EVBuilder::MPO_S_Z, EVBuilder::MPO_S_Z,
        site0, dist);

    auto spsm = ev.expVal_1sO1sO_H( 
        EVBuilder::MPO_S_P, EVBuilder::MPO_S_M,
        site0, dist);

    // auto smsp = ev.expVal_1sO1sO_H( 
    //     MPO_S_M, MPO_S_P,
    //     make_pair(0,0), 20);

    out_file_energy << std::endl << "CORRELATION FUNCTIONS" << std::endl;
    out_file_energy << "r "<< "szsz "<< "spsm "<< "SS=szsz+0.5(spsm+smsp) "
        <<"abs(SS) "<< "abs(SS_conn)" << std::endl;
    out_file_energy << std::endl;

    for (int i=0; i<dist; i++) {
        out_file_energy << (i+1) <<" "<< szsz[i].real() <<" "<< spsm[i].real()<<" "
            << (szsz[i].real() + spsm[i].real()) <<" "
            << std::abs(szsz[i].real() + spsm[i].real())<<" "
            << std::abs(szsz[i].real() + spsm[i].real() - SS_disconnected[(i+1) % 2]) <<" "
            << std::endl;
    }

    ev.analyseTransferMatrix("ARPACK");
    //ev.analyseTransferMatrix("rsvd");
    //ev.analyseTransferMatrix("gesdd");

    // Store final new cluster
    writeCluster(outClusterFile, cls);

    // write out diagnostic data of full-update
    // for (int i=0; i<arg_fuIter; i++) {
    //     std::cout<< i <<" "<< diag_ctmIter[i] <<" "<< diagData_fu[i].getInt("alsSweep",0)
    //         <<" "<< diagData_fu[i].getString("siteMaxElem")
    //         <<" "<< diagData_fu[i].getReal("finalDist0",0.0)
    //         <<" "<< diagData_fu[i].getReal("finalDist1",0.0);
    //     if (arg_fuDbg && (arg_fuDbgLevel >=1))
    //         std::cout<<" "<< diagData_fu[i].getReal("ratioNonSymLE",0.0)
    //         <<" "<< diagData_fu[i].getReal("ratioNonSymFN",0.0);
    //     std::cout<<std::endl;
    // }

    ctmEnv.computeSVDspec();
    ctmEnv.printSVDspec();
}
