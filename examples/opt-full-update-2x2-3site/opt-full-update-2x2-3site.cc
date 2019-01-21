#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include "json.hpp"
#include "ctm-cluster-basic.h"
#include "ctm-cluster-io.h"
#include "ctm-cluster-env_v2.h"
#include "cluster-ev-builder.h"
#include "mpo.h"
#include "models.h"
#include "engine.h"
#include "rsvd-solver.h"
#include "lapacksvd-solver.h"
#include "linsyssolvers-lapack.h"

using namespace itensor;

int main( int argc, char *argv[] ) {

    // ***** INITIALIZE FULL UPDATE ALGORITHM *********************************
	std::string arg_initFile = std::string(argv[1]);
	std::ifstream simfile(arg_initFile, std::ios::in);

	nlohmann::json jsonCls;
	simfile >> jsonCls;

    // write simulation parameters to log file
    std::cout << jsonCls.dump(4) << std::endl;

	//read cluster infile OR initialize by one of the predefined
	//options FILE, RND, RND_AB, AFM, RVB, ...
	std::string initBy(jsonCls["initBy"].get<std::string>());

	int physDim, auxBondDim;
	std::string inClusterFile;
	if (initBy=="FILE") inClusterFile = jsonCls["inClusterFile"].get<std::string>();
	double initStateNoise = jsonCls.value("initStateNoise",1.0e-16);
    physDim    = jsonCls["physDim"].get<int>();
	auxBondDim = jsonCls["auxBondDim"].get<int>();

	// read cluster outfile
	std::string outClusterFile(jsonCls["outClusterFile"].get<std::string>());

	// read Hamiltonian and Trotter decomposition
    auto json_model_params(jsonCls["model"]); 
    bool symmTrotter  = json_model_params.value("symmTrotter",true);
    bool randomizeSeq = json_model_params.value("randomizeSeq",false);

	// full update parameters
    int arg_fuIter  = jsonCls["fuIter"].get<int>();
    int arg_obsFreq = jsonCls["obsFreq"].get<int>();
    bool arg_fuTrialInit = jsonCls.value("fuTrialInit",true);
    bool arg_fuDbg = jsonCls["fuDbg"].get<bool>();
    int arg_fuDbgLevel = jsonCls["fuDbgLevel"].get<int>();
    std::string arg_otNormType = jsonCls["otNormType"].get<std::string>();


    int arg_maxAltLstSqrIter = jsonCls["maxAltLstSqrIter"].get<int>();
    std::string arg_fuIsoInit = jsonCls["fuIsoInit"].get<std::string>();
    double arg_fuIsoInitNoiseLevel = jsonCls["fuIsoInitNoiseLevel"].get<double>();
    bool symmetrizeProtoEnv = jsonCls["symmetrizeProtoEnv"].get<bool>();
    bool posDefProtoEnv = jsonCls["positiveDefiniteProtoEnv"].get<bool>();
    // Iterative ALS procedure
    double epsdistf = jsonCls.value("epsdistf",1.0e-8);
    auto json_als_params(jsonCls["als"]);
    std::string linsolver = json_als_params.value("solver","UNSUPPORTED");
    bool solver_dbg    = json_als_params.value("dbg",false);
    double epsregularisation = json_als_params.value("epsregularisation",0.0);

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
    double arg_suTol = json_gauge_fix_params.value("suTol",1.0e-08);
    bool arg_gf_dbg = json_gauge_fix_params.value("suDbg",false);
    int arg_gf_dbgLvl = json_gauge_fix_params.value("suDbgLevel",0);
    

	// read CTMRG parameters
    auto json_ctmrg_params(jsonCls["ctmrg"]);
	int auxEnvDim = json_ctmrg_params["auxEnvDim"].get<int>();
	std::string arg_ioEnvTag(json_ctmrg_params["ioEnvTag"].get<std::string>());
	CtmEnv::init_env_type arg_initEnvType(toINIT_ENV(json_ctmrg_params["initEnvType"].get<std::string>()));
	bool envIsComplex = json_ctmrg_params["envIsComplex"].get<bool>();
	CtmEnv::isometry_type iso_type(toISOMETRY(json_ctmrg_params["isoType"].get<std::string>()));
    CtmEnv::normalization_type norm_type(toNORMALIZATION(json_ctmrg_params["normType"].get<std::string>()));
    std::string env_SVD_METHOD(json_ctmrg_params["env_SVD_METHOD"].get<std::string>());
    auto rsvd_power   = json_ctmrg_params.value("rsvd_power",2);
    auto rsvd_reortho = json_ctmrg_params.value("rsvd_reortho",1);
    auto rsvd_oversampling = json_ctmrg_params.value("rsvd_oversampling",10);
	int arg_maxEnvIter     = json_ctmrg_params["maxEnvIter"].get<int>();
    int arg_maxInitEnvIter = json_ctmrg_params["initMaxEnvIter"].get<int>();
    int arg_obsMaxIter    = json_ctmrg_params.value("obsMaxIter",arg_maxInitEnvIter);
    double arg_envEps     = json_ctmrg_params["envEpsilon"].get<double>();
    bool arg_reinitEnv    = json_ctmrg_params["reinitEnv"].get<bool>();
    bool arg_reinitObsEnv = json_ctmrg_params.value("reinitObsEnv",false);
    bool arg_envDbg       = json_ctmrg_params["dbg"].get<bool>();
    int arg_envDbgLvl     = json_ctmrg_params["dbgLvl"].get<int>();
    // end reading CTMRG parameters
    // ***** INITIALIZE FULL UPDATE ALGORITHM DONE ****************************

	// ----- INITIALIZE CLUSTER -----------------------------------------------
	std::unique_ptr<Cluster> p_cls;

	if (initBy=="FILE") {
		std::ifstream infile(inClusterFile, std::ios::in);
        nlohmann::json jsonCls = nlohmann::json::parse(infile);

        // preprocess parameters of input cluster
        jsonCls["auxBondDim"] = auxBondDim;
        for(auto & site : jsonCls["sites"]) {
            site["auxDim"] = auxBondDim;
        }

        p_cls = p_readCluster(jsonCls);
        // initClusterSites(cls);
        // initClusterWeights(cls);
        // setWeights(*p_cls, suWeightsInit);
        // setOnSiteTensorsFromFile(cls, inClusterFile);
    } else {
        p_cls = std::unique_ptr<Cluster_2x2_ABCD>( 
            new Cluster_2x2_ABCD(initBy, auxBondDim, physDim));
        // cls = Cluster_2x2_ABCD(initBy, auxBondDim, physDim);
    }
    std::cout << *p_cls;

    // add random noise to initial state
    {
        ITensor temp; 
        double eps = initStateNoise;
        auto setMeanTo0 = [](Real r) { return (r-0.5); };

        for(auto& st : p_cls->sites) {
            temp = st.second;
            randomize(temp);
            temp.apply(setMeanTo0);
            st.second += eps*temp;
        }
    }

    // write simulations params into cluster
    p_cls->simParam = jsonCls;
    // ----- END DEFINE CLUSTER -----------------------------------------------

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

    // ***** Select LinSys solver to use **************************************
    std::unique_ptr<LinSysSolver> pLinSysSolver;
    if (linsolver == "cholesky") {
        pLinSysSolver = std::unique_ptr<CholeskySolver>(new CholeskySolver());
    } else if (linsolver == "pseudoinverse") {
        pLinSysSolver = std::unique_ptr<PseudoInvSolver>(new PseudoInvSolver());
    } else {
        std::cout<<"WARNING: Unsupported LinSysSolver specified. Using default"<<std::endl;
        pLinSysSolver = std::unique_ptr<PseudoInvSolver>(new PseudoInvSolver());
    }
    
    // INITIALIZE ENVIRONMENT
    auto env_args = Args("Name","CTMRG_parameters");
    for (nlohmann::json::iterator it = json_ctmrg_params.begin(); 
        it != json_ctmrg_params.end(); ++it) { 
        std::string key = it.key();
        auto val        = it.value();
        if (val.is_string()) env_args.add(key,val.get<std::string>());
        else if (val.is_boolean()) env_args.add(key,(bool) val); 
        else if (val.is_number_integer()) env_args.add(key,(int) val); 
        else if (val.is_number_float()) env_args.add(key,(double) val); 
    }

    CtmEnv ctmEnv(arg_ioEnvTag, auxEnvDim, *p_cls, *pSvdSolver, env_args);
    ctmEnv.init(arg_initEnvType, envIsComplex, arg_envDbg);
    
    // INITIALIZE EXPECTATION VALUE BUILDER
    EVBuilder ev(arg_ioEnvTag, *p_cls, ctmEnv);
    std::cout << ev;

    // hold ctm observables used in the convergence criterion
    std::vector<double> e_curr(4,0.0), e_prev(4,0.0);

    using time_point = std::chrono::steady_clock::time_point;
    time_point t_begin_int, t_end_int;
    auto get_s = [](time_point ti, time_point tf) { return std::chrono::duration_cast
            <std::chrono::microseconds>(tf - ti).count()/1.0e+06; };

    // ##################################################################
    // # SETUP OPTIMIZATION LOOP                                        #
    // ##################################################################

    // DEFINE MODEL AND GATE SEQUENCE
    std::unique_ptr<Model>  ptr_model;
    std::unique_ptr<Engine> ptr_engine;     // full update engine
    std::unique_ptr<Engine> ptr_gfe;        // gauge fixing engine

    ptr_model =  getModel(json_model_params);
    ptr_engine = buildEngine(json_model_params, pLinSysSolver.get() );//, *pLinSysSolver);
    
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
    
    Args fuArgs = {
        "maxAltLstSqrIter",arg_maxAltLstSqrIter,
        "fuTrialInit",arg_fuTrialInit,
        "fuDbg",arg_fuDbg,
        "fuDbgLevel",arg_fuDbgLevel,
        "symmetrizeProtoEnv",symmetrizeProtoEnv,
        "positiveDefiniteProtoEnv",posDefProtoEnv,
        "otNormType",arg_otNormType,
        "epsdistf",epsdistf,

        "fuIsoInit",arg_fuIsoInit,
        "fuIsoInitNoiseLevel",arg_fuIsoInitNoiseLevel,

        "solver",linsolver,
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
    std::vector<double> accT(12,0.0); // holds timings for CTM moves
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

        ctmEnv.move_unidirectional(CtmEnv::DIRECTION::LEFT, iso_type, accT);
        ctmEnv.move_unidirectional(CtmEnv::DIRECTION::UP, iso_type, accT);
        ctmEnv.move_unidirectional(CtmEnv::DIRECTION::RIGHT, iso_type, accT);
        ctmEnv.move_unidirectional(CtmEnv::DIRECTION::DOWN, iso_type, accT);

        t_end_int = std::chrono::steady_clock::now();
        std::cout << "CTM STEP " << envI <<" T: "<< get_s(t_begin_int,t_end_int) <<" [sec] ";

        if ( envI % 1 == 0 ) {
            t_begin_int = std::chrono::steady_clock::now();
            
            e_curr[0]=ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(0,0), Vertex(1,0));
            e_curr[1]=ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(0,0), Vertex(0,1));
            e_curr[2]=ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(1,0), Vertex(1,1));
            e_curr[3]=ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(0,1), Vertex(1,1));

           t_end_int = std::chrono::steady_clock::now();
            std::cout <<"|| T: "<< get_s(t_begin_int,t_end_int) <<" [sec] E: "
                << e_curr[0] <<" "<< e_curr[1] <<" "<< e_curr[2] <<" "<< e_curr[3] << std::endl;

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

                std::ostringstream oss;
                oss << std::scientific;

                // Compute spectra of Corner matrices
                std::cout << std::endl;
                double tmpVal;
                double minCornerSV = 1.0e+16;
                Args args_dbg_cornerSVD = {"Truncate",false};
                std::cout << "Spectra: " << std::endl;

                ITensor tL(ctmEnv.C_LU.at(ctmEnv.p_cluster->siteIds[0]).inds().front()),sv,tR;
                auto spec = svd(ctmEnv.C_LU.at(ctmEnv.p_cluster->siteIds[0]),
                    tL,sv,tR,args_dbg_cornerSVD);
                tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                    sv.inds().back()(auxEnvDim));
                PrintData(sv);
                minCornerSV = std::min(minCornerSV, tmpVal);
                oss << tmpVal;

                tL = ITensor(ctmEnv.C_RU.at(ctmEnv.p_cluster->siteIds[0]).inds().front());
                spec = svd(ctmEnv.C_RU.at(ctmEnv.p_cluster->siteIds[0]),
                    tL,sv,tR,args_dbg_cornerSVD);
                tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                    sv.inds().back()(auxEnvDim));
                PrintData(sv);
                minCornerSV = std::min(minCornerSV, tmpVal);
                oss <<" "<< tmpVal;

                tL = ITensor(ctmEnv.C_RD.at(ctmEnv.p_cluster->siteIds[0]).inds().front());
                spec = svd(ctmEnv.C_RD.at(ctmEnv.p_cluster->siteIds[0]),
                    tL,sv,tR,args_dbg_cornerSVD);
                tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                    sv.inds().back()(auxEnvDim));
                PrintData(sv);
                minCornerSV = std::min(minCornerSV, tmpVal);
                oss <<" "<< tmpVal;

                tL = ITensor(ctmEnv.C_LD.at(ctmEnv.p_cluster->siteIds[0]).inds().front());
                spec = svd(ctmEnv.C_LD.at(ctmEnv.p_cluster->siteIds[0]),
                    tL,sv,tR,args_dbg_cornerSVD);
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

    std::cout <<"Timings(CTMRG) :"<<"Projectors "<<"AbsorbReduce "<<"N/A "<<"Postprocess"<< std::endl;
    std::cout <<"accT [mSec]: "<< accT[0] <<" "<< accT[1] <<" "<< accT[2]
        <<" "<< accT[3] << std::endl;
    std::cout <<"Timings(Projectors): "<<"Enlarge "<<"N/A "<<"SVD "<<"Contract"<< std::endl;
    std::cout <<"isoZ [mSec]: "<< accT[4] <<" "<< accT[5] <<" "<< accT[6]
        <<" "<< accT[7] << std::endl;
    std::cout <<"Timings(AbsorbReduce): "<<"C "<<"T "<<"Ct "<<"N/A"<< std::endl;
    std::cout <<"[mSec]: "<< accT[8] <<" "<< accT[9] <<" "<< accT[10]
        <<" "<< accT[11] << std::endl;
    // ***** COMPUTING INITIAL ENVIRONMENT DONE *******************************

    // Compute initial properties
    ptr_model->setObservablesHeader(out_file_energy);
    auto obs_metaInf = Args("lineNo",0);
    ptr_model->computeAndWriteObservables(ev, out_file_energy, obs_metaInf);
    best_energy = obs_metaInf.getReal("energy");

    // ENTER OPTIMIZATION LOOP
    for (int fuI = 1; fuI <= arg_fuIter; fuI++) {
    	std::cout <<"Full Update - STEP "<< fuI << std::endl;

        // ctmEnv.symmetrizeEnv();
        diag_fu = ptr_engine->performFullUpdate(*p_cls, ctmEnv, fuArgs);

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

        
        // fix gauge by simple-update at dt=0 - identity operators
        if ( arg_su_gauge_fix && (fuI % arg_su_gauge_fix_freq == 0) ) {
            std::cout << "GAUGE FIXING" << std::endl;
            auto num_eps = std::numeric_limits<double>::epsilon();
            t_begin_int = std::chrono::steady_clock::now();

            auto const tol = (true) ? arg_suTol : num_eps*p_cls->auxBondDim*p_cls->weights.size();

            // Assuming the weights have been initialized
            initClusterWeights(*p_cls);
            setWeights(*p_cls, arg_suWeightsInit);

            Args gfArgs = {"suDbg",arg_gf_dbg,"suDbgLevel",arg_gf_dbgLvl};
            Args gf_diag_fu;
            saveWeights(*p_cls);
            for (int suI = 1; suI <= arg_suIter; suI++) {
                //std::cout <<"Simple Update - STEP "<< suI << std::endl;

                // PERFORM SIMPLE UPDATE
                gf_diag_fu = ptr_gfe->performSimpleUpdate(*p_cls, gfArgs);
                
                //check convergence
                if (suI % 8 == 0) {
                    auto weight_distance = weightDist(*p_cls);
                    if ( weight_distance < tol ) {
                        std::cout<<"GF iter: "<< suI <<" dist: "<< weight_distance 
                            <<" CONVERGED"<< std::endl;
                        break;
                    } else {
                        std::cout<<"GF iter: "<< suI <<" dist: "<< weight_distance << std::endl;
                    }
                    saveWeights(*p_cls);
                }
            }

            p_cls->absorbWeightsToSites();
        
            t_end_int = std::chrono::steady_clock::now();
            std::cout << "GUAGE FIX DONE" << " T: "<< get_s(t_begin_int,t_end_int) <<" [sec] "; 
        }

        // SETUP ENVIRONMENT LOOP
        accT = std::vector<double>(12,0.0);
        // reset environment
        if (arg_reinitEnv || ((fuI % arg_obsFreq == 0) && arg_reinitObsEnv) ) 
            ctmEnv.init(arg_initEnvType, envIsComplex, arg_envDbg);
		
    	// ENTER ENVIRONMENT LOOP
        int currentMaxEnvIter = (fuI % arg_obsFreq == 0) ? arg_obsMaxIter : arg_maxEnvIter; 
		bool expValEnvConv = false;
        for (int envI=1; envI<=currentMaxEnvIter; envI++ ) {
            t_begin_int = std::chrono::steady_clock::now();

	        ctmEnv.move_unidirectional(CtmEnv::DIRECTION::LEFT, iso_type, accT);
            ctmEnv.move_unidirectional(CtmEnv::DIRECTION::UP, iso_type, accT);
            ctmEnv.move_unidirectional(CtmEnv::DIRECTION::RIGHT, iso_type, accT);
            ctmEnv.move_unidirectional(CtmEnv::DIRECTION::DOWN, iso_type, accT);

            t_end_int = std::chrono::steady_clock::now();
            std::cout << "CTM STEP " << envI <<" T: "<< get_s(t_begin_int,t_end_int) <<" [sec] "; 

	        if ( (currentMaxEnvIter > 1) && (envI % 1 == 0) ) {
                t_begin_int = std::chrono::steady_clock::now();

                e_curr[0]=ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(0,0), Vertex(1,0));
                e_curr[1]=ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(0,0), Vertex(0,1));
                e_curr[2]=ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(1,0), Vertex(1,1));
                e_curr[3]=ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(0,1), Vertex(1,1));

                t_end_int = std::chrono::steady_clock::now();

	            std::cout<<" || E in T: "<< get_s(t_begin_int,t_end_int) <<" [sec] E: "
                    << e_curr[0] <<" "<< e_curr[1] <<" "<< e_curr[2] <<" "<< e_curr[3]; 

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

                    // Compute spectra of Corner matrices
                    std::cout << std::endl;
                    double tmpVal;
                    double minCornerSV = 1.0e+16;
                    Args args_dbg_cornerSVD = {"Truncate",false};
                    std::cout << "Spectra: " << std::endl;

                    ITensor tL(ctmEnv.C_LU.at(ctmEnv.p_cluster->siteIds[0]).inds().front()),sv,tR;
                    auto spec = svd(ctmEnv.C_LU.at(ctmEnv.p_cluster->siteIds[0]),
                        tL,sv,tR,args_dbg_cornerSVD);
                    tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                        sv.inds().back()(auxEnvDim));
                    PrintData(sv);
                    minCornerSV = std::min(minCornerSV, tmpVal);
                    oss << tmpVal;

                    tL = ITensor(ctmEnv.C_RU.at(ctmEnv.p_cluster->siteIds[0]).inds().front());
                    spec = svd(ctmEnv.C_RU.at(ctmEnv.p_cluster->siteIds[0]),
                        tL,sv,tR,args_dbg_cornerSVD);
                    tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                        sv.inds().back()(auxEnvDim));
                    PrintData(sv);
                    minCornerSV = std::min(minCornerSV, tmpVal);
                    oss <<" "<< tmpVal;

                    tL = ITensor(ctmEnv.C_RD.at(ctmEnv.p_cluster->siteIds[0]).inds().front());
                    spec = svd(ctmEnv.C_RD.at(ctmEnv.p_cluster->siteIds[0]),
                        tL,sv,tR,args_dbg_cornerSVD);
                    tmpVal = sv.real(sv.inds().front()(auxEnvDim),
                        sv.inds().back()(auxEnvDim));
                    PrintData(sv);
                    minCornerSV = std::min(minCornerSV, tmpVal);
                    oss <<" "<< tmpVal;

                    tL = ITensor(ctmEnv.C_LD.at(ctmEnv.p_cluster->siteIds[0]).inds().front());
                    spec = svd(ctmEnv.C_LD.at(ctmEnv.p_cluster->siteIds[0]),
                        tL,sv,tR,args_dbg_cornerSVD);
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

	    std::cout <<"Timings(CTMRG) :"<<"Projectors "<<"AbsorbReduce "<<"N/A "<<"Postprocess"<< std::endl;
        std::cout <<"accT [mSec]: "<< accT[0] <<" "<< accT[1] <<" "<< accT[2]
            <<" "<< accT[3] << std::endl;
        std::cout <<"Timings(Projectors): "<<"Enlarge "<<"N/A "<<"SVD "<<"Contract"<< std::endl;
        std::cout <<"isoZ [mSec]: "<< accT[4] <<" "<< accT[5] <<" "<< accT[6]
            <<" "<< accT[7] << std::endl;
        std::cout <<"Timings(AbsorbReduce): "<<"C "<<"T "<<"Ct "<<"N/A"<< std::endl;
        std::cout <<"[mSec]: "<< accT[8] <<" "<< accT[9] <<" "<< accT[10]
            <<" "<< accT[11] << std::endl;

        if (fuI % arg_obsFreq == 0) {
            t_begin_int = std::chrono::steady_clock::now();

            // ctmEnv.symmetrizeEnv();
            auto metaInf = Args("lineNo",fuI);
            ptr_model->computeAndWriteObservables(ev,out_file_energy,metaInf);

            // check energy, preserve the best_energy state obtained so far
            auto current_energy = metaInf.getReal("energy");
            if( best_energy > current_energy ) {
                best_energy = current_energy;
                p_cls->metaInfo = "BestEnergy(FUStep=" + std::to_string(fuI) + ")";
                writeCluster(outClusterBestFile, *p_cls);
            }

            t_end_int = std::chrono::steady_clock::now();

            std::cout << "Observables computed in T: "<< get_s(t_begin_int,t_end_int) 
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
        
            writeCluster(outClusterFile, *p_cls);
        }
    }

    // FULL UPDATE FINISHED - COMPUTING FINAL ENVIRONMENT
    std::cout <<"FULL UPDATE DONE - COMPUTING FINAL ENVIRONMENT "<< std::endl;
    // t_begin_int = std::chrono::steady_clock::now();
    accT = std::vector<double>(12,0.0);

    // reset environment
    if (arg_reinitEnv) ctmEnv.init(arg_initEnvType, envIsComplex, arg_envDbg);

    // ENTER ENVIRONMENT LOOP
    for (int envI=1; envI<=arg_maxInitEnvIter; envI++ ) {

        ctmEnv.move_unidirectional(CtmEnv::DIRECTION::LEFT, iso_type, accT);
        ctmEnv.move_unidirectional(CtmEnv::DIRECTION::UP, iso_type, accT);
        ctmEnv.move_unidirectional(CtmEnv::DIRECTION::RIGHT, iso_type, accT);
        ctmEnv.move_unidirectional(CtmEnv::DIRECTION::DOWN, iso_type, accT);

        if ( envI % 1 == 0 ) {
        
            t_end_int = std::chrono::steady_clock::now();
            std::cout << "CTM STEP " << envI <<" T: "<< get_s(t_begin_int,t_end_int) <<" [sec] E: "
                << e_curr[0] <<" "<< e_curr[1] <<" "<< e_curr[2] <<" "<< e_curr[3] << std::endl;
            
            e_curr[0]=ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(0,0), Vertex(1,0));
            e_curr[1]=ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(0,0), Vertex(0,1));
            e_curr[2]=ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(1,0), Vertex(1,1));
            e_curr[3]=ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(0,1), Vertex(1,1));

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

    std::cout <<"Timings(CTMRG) :"<<"Projectors "<<"AbsorbReduce "<<"N/A "<<"Postprocess"<< std::endl;
    std::cout <<"accT [mSec]: "<< accT[0] <<" "<< accT[1] <<" "<< accT[2]
        <<" "<< accT[3] << std::endl;
    std::cout <<"Timings(Projectors): "<<"Enlarge "<<"N/A "<<"SVD "<<"Contract"<< std::endl;
    std::cout <<"isoZ [mSec]: "<< accT[4] <<" "<< accT[5] <<" "<< accT[6]
        <<" "<< accT[7] << std::endl;
    std::cout <<"Timings(AbsorbReduce): "<<"C "<<"T "<<"Ct "<<"N/A"<< std::endl;
    std::cout <<"[mSec]: "<< accT[8] <<" "<< accT[9] <<" "<< accT[10]
        <<" "<< accT[11] << std::endl;

    obs_metaInf = Args("lineNo",arg_fuIter+1);
	ptr_model->computeAndWriteObservables(ev,out_file_energy,obs_metaInf);

    // COMPUTE CORRELATION FUNCTIONS
    int dist = 40;
    auto site0 = Vertex(0,0);
    auto site1 = Vertex(1,0);

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

    // std::vector<double> SS_disconnected = {S0S0,S0S1};

    // auto szsz = ev.expVal_1sO1sO_H( 
    //     EVBuilder::MPO_S_Z, EVBuilder::MPO_S_Z,
    //     site0, dist);

    // auto spsm = ev.expVal_1sO1sO_H( 
    //     EVBuilder::MPO_S_P, EVBuilder::MPO_S_M,
    //     site0, dist);

    // auto smsp = ev.expVal_1sO1sO_H( 
    //     MPO_S_M, MPO_S_P,
    //     make_pair(0,0), 20);

    // out_file_energy << std::endl << "CORRELATION FUNCTIONS" << std::endl;
    // out_file_energy << "r "<< "szsz "<< "spsm "<< "SS=szsz+0.5(spsm+smsp) "
    //     <<"abs(SS) "<< "abs(SS_conn)" << std::endl;
    // out_file_energy << std::endl;

    // for (int i=0; i<dist; i++) {
    //     out_file_energy << (i+1) <<" "<< szsz[i].real() <<" "<< spsm[i].real()<<" "
    //         << (szsz[i].real() + spsm[i].real()) <<" "
    //         << std::abs(szsz[i].real() + spsm[i].real())<<" "
    //         << std::abs(szsz[i].real() + spsm[i].real() - SS_disconnected[(i+1) % 2]) <<" "
    //         << std::endl;
    // }

    ev.analyzeTransferMatrix(Vertex(0,0), CtmEnv::DIRECTION::RIGHT);
    ev.analyzeTransferMatrix(Vertex(0,0), CtmEnv::DIRECTION::DOWN);

    // Store final new cluster
    writeCluster(outClusterFile, *p_cls);

    // ctmEnv.computeSVDspec();
    // ctmEnv.printSVDspec();
}
