#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include "json.hpp"
#include "ctm-cluster-env_v2.h"
#include "cluster-ev-builder.h"
#include "ctm-cluster-io.h"
#include "ctm-cluster.h"
#include "mpo.h"
#include "models.h"
#include "engine.h"
//#include "simple-update_v2.h"
#include "itensor/all.h"
#include "itensor-svd-solvers.h"
#include "rsvd-solver.h"
#include "lapacksvd-solver.h"

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
    bool symmTrotter  = json_model_params.value("symmTrotter",true);
    bool randomizeSeq = json_model_params.value("randomizeSeq",false);
	
    std::string sitesInit(jsonCls.value("sitesInit","FILE"));
    std::string suWeightsInit(jsonCls.value("suWeightsInit","DELTA"));

    int arg_suIter  = jsonCls["suIter"].get<int>();
    int arg_obsFreq = jsonCls["obsFreq"].get<int>();
    bool arg_stopEnergyInc = jsonCls.value("stopEnergyInc",false);
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
    auto rsvd_power   = json_ctmrg_params.value("rsvd_power",2);
    auto rsvd_reortho = json_ctmrg_params.value("rsvd_reortho",1);
    auto rsvd_oversampling = json_ctmrg_params.value("rsvd_oversampling",10);
    int arg_maxEnvIter     = json_ctmrg_params["maxEnvIter"].get<int>();
    int arg_maxInitEnvIter = json_ctmrg_params["initMaxEnvIter"].get<int>();
    double arg_envEps  = json_ctmrg_params["envEpsilon"].get<double>();
    bool arg_reinitEnv = json_ctmrg_params["reinitEnv"].get<bool>();
    bool arg_envDbg    = json_ctmrg_params["dbg"].get<bool>();
    int arg_envDbgLvl  = json_ctmrg_params["dbgLvl"].get<int>();
    // end reading CTMRG parameters

    // ***** INITIALIZE SIMPLE UPDATE ALGORITHM DONE **************************

	// ***** INITIALIZE CLUSTER ***********************************************
    Cluster cls;
    
    // set auxiliary dimension to the desired one
    cls.auxBondDim = auxBondDim;
    

    // choose initial wavefunction
    if (sitesInit == "FILE") {
        cls = readCluster(inClusterFile);
        initClusterSites(cls);
        initClusterWeights(cls);
        setWeights(cls, suWeightsInit);
        setOnSiteTensorsFromFile(cls, inClusterFile);
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

        setSites(cls, sitesInit);
        initClusterWeights(cls);
        setWeights(cls, suWeightsInit);
    }
    std::cout << cls;
    // ***** INITIALIZE CLUSTER DONE ******************************************

    // ***** INITIALIZE MODEL *************************************************
    // DEFINE MODEL AND GATE SEQUENCE
    std::unique_ptr<Model>  ptr_model;
    std::unique_ptr<Engine> ptr_engine;     // simple update engine

    ptr_model =  getModel(json_model_params);
    ptr_engine = buildEngine(json_model_params);

    // ***** INITIALIZE MODEL DONE ********************************************

    // *****
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

    std::vector<double> accT(8,0.0); // holds timings for CTM moves
    std::chrono::steady_clock::time_point t_begin_int, t_end_int;
    // *****

    // ***** INITIALIZE ENVIRONMENT *******************************************
    Cluster evCls = cls;
    absorbWeightsToSites(evCls);

    // ***** Select SVD solver to use *****************************************
    std::unique_ptr<SvdSolver> pSvdSolver;
    if (env_SVD_METHOD == "rsvd") {
        pSvdSolver = std::unique_ptr<RsvdSolver>(new RsvdSolver());
    } 
    else if (env_SVD_METHOD == "gesdd") {
        pSvdSolver = std::unique_ptr<GESDDSolver>(new GESDDSolver());
    }  
    else if (env_SVD_METHOD == "itensor") {
        pSvdSolver = std::unique_ptr<SvdSolver>(new SvdSolver());
    } 
    else {
        std::cout<<"WARNING: Unsupported or no SvdSolver specified."
            <<" Using itensor"<<std::endl;
        // TODO? set jsonCls["ctmrg"]["env_SVD_METHOD"] = "itensor";
        pSvdSolver = std::unique_ptr<SvdSolver>(new SvdSolver());
    }

    CtmEnv ctmEnv(arg_ioEnvTag, auxEnvDim, evCls, *pSvdSolver,
        {"isoPseudoInvCutoff",arg_isoPseudoInvCutoff,
         "SVD_METHOD",env_SVD_METHOD,
         "rsvd_power",rsvd_power,
         "rsvd_reortho",rsvd_reortho,
         "rsvd_oversampling",rsvd_oversampling,
         "dbg",arg_envDbg,
         "dbgLevel",arg_envDbgLvl}
        );
    ctmEnv.init(arg_initEnvType, envIsComplex, arg_envDbg);
    
    // INITIALIZE EXPECTATION VALUE BUILDER
    EVBuilder ev(arg_ioEnvTag, evCls, ctmEnv.getCtmData_DBG());
    ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());
    
    ctmEnv.updateCluster(evCls);
    ev.setCluster(evCls);


    std::vector<double> diag_minCornerSV(1, 0.);
    bool expValEnvConv = false;
    // COMPUTE INITIAL OBSERVABLES
    for (int envI=1; envI<=arg_maxEnvIter; envI++ ) {
        t_begin_int = std::chrono::steady_clock::now();

        ctmEnv.move_singleDirection(0, iso_type, cls, accT);
        ctmEnv.move_singleDirection(0, iso_type, cls, accT);
        ctmEnv.move_singleDirection(1, iso_type, cls, accT);
        ctmEnv.move_singleDirection(1, iso_type, cls, accT);
        ctmEnv.move_singleDirection(2, iso_type, cls, accT);
        ctmEnv.move_singleDirection(2, iso_type, cls, accT);
        ctmEnv.move_singleDirection(3, iso_type, cls, accT);
        ctmEnv.move_singleDirection(3, iso_type, cls, accT);

        t_end_int = std::chrono::steady_clock::now();
        std::cout << "CTM STEP " << envI <<" T: "<< std::chrono::duration_cast
                <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
                <<" [sec] "; 

        if ( (arg_maxEnvIter > 1) && (envI % 1 == 0) ) {
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

                expValEnvConv = true;
                std::cout<< " ENV CONVERGED ";
            }

            if ( envI==arg_maxEnvIter )  {
                expValEnvConv = true;
                std::cout<< " MAX ENV iterations REACHED ";
            }
            e_prev = e_curr;

            if (expValEnvConv) {
                diag_ctmIter.push_back(envI);

                std::ostringstream oss;
                oss << std::scientific;

                // diagnose spectra
                std::cout << std::endl;
                double tmpVal;
                double minCornerSV = 1.0e+16;
                Args args_dbg_cornerSVD = {"Truncate",false};
                std::cout << "Spectra: " << std::endl;

                ITensor tL(ctmEnv.C_LU[0].inds().front()),sv,tR;
                auto spec = svd(ctmEnv.C_LU[0],tL,sv,tR,args_dbg_cornerSVD);
                tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                    sv.inds().back()(auxEnvDim));
                PrintData(sv);
                minCornerSV = std::min(minCornerSV, tmpVal);
                oss << tmpVal;

                tL = ITensor(ctmEnv.C_RU[0].inds().front());
                spec = svd(ctmEnv.C_RU[0],tL,sv,tR,args_dbg_cornerSVD);
                tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                    sv.inds().back()(auxEnvDim));
                PrintData(sv);
                minCornerSV = std::min(minCornerSV, tmpVal);
                oss <<" "<< tmpVal;

                tL = ITensor(ctmEnv.C_RD[0].inds().front());
                spec = svd(ctmEnv.C_RD[0],tL,sv,tR,args_dbg_cornerSVD);
                tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                    sv.inds().back()(auxEnvDim));
                PrintData(sv);
                minCornerSV = std::min(minCornerSV, tmpVal);
                oss <<" "<< tmpVal;

                tL = ITensor(ctmEnv.C_LD[0].inds().front());
                spec = svd(ctmEnv.C_LD[0],tL,sv,tR,args_dbg_cornerSVD);
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
        std::cout << std::endl;
    }

    std::cout <<"accT [mSec]: "<< accT[0] <<" "<< accT[1] <<" "<< accT[2]
        <<" "<< accT[3] << std::endl;
    std::cout <<"isoZ [mSec]: "<< accT[4] <<" "<< accT[5] <<" "<< accT[6]
        <<" "<< accT[7] << std::endl;

    
    ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());
    // Compute initial properties
    ptr_model->setObservablesHeader(out_file_energy);
    auto metaInf = Args("lineNo",0);
    ptr_model->computeAndWriteObservables(ev, out_file_energy,metaInf);
    // ***** INITIALIZE ENVIRONMENT DONE **************************************


    // ########################################################################
    // # SETUP OPTIMIZATION LOOP                                              #
    // ########################################################################

    Args suArgs = {
        "suDbg",arg_suDbg,
        "suDbgLevel",arg_suDbgLevel
    };

    // ENTER OPTIMIZATION LOOP
    for (int suI = 1; suI <= arg_suIter; suI++) {
        std::cout <<"Simple Update - STEP "<< suI << std::endl;

        // PERFORM SIMPLE UPDATE
        diag_fu = ptr_engine->performSimpleUpdate(cls, suArgs);

        diagData_fu.push_back(diag_fu);

        if (suI % arg_obsFreq == 0) {
            evCls = cls;
            absorbWeightsToSites(evCls);
            ctmEnv.updateCluster(evCls);
            ev.setCluster(evCls);
            
            // reset environment
            if (arg_reinitEnv) ctmEnv.init(arg_initEnvType, envIsComplex, arg_envDbg);
                

            // ENTER ENVIRONMENT LOOP
            bool expValEnvConv = false;
            for (int envI=1; envI<=arg_maxEnvIter; envI++ ) {
                t_begin_int = std::chrono::steady_clock::now();

                ctmEnv.move_unidirectional(0, iso_type, cls, accT);
                ctmEnv.move_unidirectional(1, iso_type, cls, accT);
                ctmEnv.move_unidirectional(2, iso_type, cls, accT);
                ctmEnv.move_unidirectional(3, iso_type, cls, accT);

                t_end_int = std::chrono::steady_clock::now();
                std::cout << "CTM STEP " << envI <<" T: "<< std::chrono::duration_cast
                        <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
                        <<" [sec] "; 

                if ( (arg_maxEnvIter > 1) && (envI % 1 == 0) ) {
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

                    if ( envI==arg_maxEnvIter )  {
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
                        PrintData(sv);
                        minCornerSV = std::min(minCornerSV, tmpVal);
                        oss << tmpVal;

                        tL = ITensor(ctmEnv.C_RU[0].inds().front());
                        spec = svd(ctmEnv.C_RU[0],tL,sv,tR,args_dbg_cornerSVD);
                        tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                            sv.inds().back()(auxEnvDim));
                        PrintData(sv);
                        minCornerSV = std::min(minCornerSV, tmpVal);
                        oss <<" "<< tmpVal;

                        tL = ITensor(ctmEnv.C_RD[0].inds().front());
                        spec = svd(ctmEnv.C_RD[0],tL,sv,tR,args_dbg_cornerSVD);
                        tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                            sv.inds().back()(auxEnvDim));
                        PrintData(sv);
                        minCornerSV = std::min(minCornerSV, tmpVal);
                        oss <<" "<< tmpVal;

                        tL = ITensor(ctmEnv.C_LD[0].inds().front());
                        spec = svd(ctmEnv.C_LD[0],tL,sv,tR,args_dbg_cornerSVD);
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
                std::cout << std::endl;
            }

            std::cout <<"accT [mSec]: "<< accT[0] <<" "<< accT[1] <<" "<< accT[2]
                <<" "<< accT[3] << std::endl;
            std::cout <<"isoZ [mSec]: "<< accT[4] <<" "<< accT[5] <<" "<< accT[6]
                <<" "<< accT[7] << std::endl;

            out_file_diag << suI <<" "<< diag_ctmIter.back()
                <<" "<< diag_minCornerSV.back();
        //     <<" "<< diag_fu.getString("siteMaxElem")
        //     <<" "<< diag_fu.getReal("finalDist0",0.0)
        //     <<" "<< diag_fu.getReal("finalDist1",0.0);
        // out_file_diag <<" "<< diag_fu.getReal("ratioNonSymLE",0.0)
        //     <<" "<< diag_fu.getReal("ratioNonSymFN",0.0);
        // out_file_diag <<" "<< diag_fu.getReal("minGapDisc",0.0) 
        //     <<" "<< diag_fu.getReal("minEvKept",0.0);
            out_file_diag << std::endl;
            
            ev.setCtmData_Full(ctmEnv.getCtmData_Full_DBG());

            t_begin_int = std::chrono::steady_clock::now();

            auto metaInf = Args("lineNo",suI);
            ptr_model->computeAndWriteObservables(ev,out_file_energy,metaInf);

            t_end_int = std::chrono::steady_clock::now();

            std::cout << "Observables computed in T: "<< std::chrono::duration_cast
                    <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
                    <<" [sec] "<< std::endl;

            writeCluster(outClusterFile, evCls);
            
            // TODO current energy is higher than energy at previous step STOP
            // if (arg_stopEnergyInc && *energyDiff*) {
            //     break;
            //     std::cout<< "CURRENT ENERGY > PREVIOUS ENERGY: STOPPING"<<std::endl;
            // }
        }

    }

    // SIMPLE UPDATE FINISHED
    std::cout <<"SIMPLE UPDATE DONE"<< std::endl;
}
