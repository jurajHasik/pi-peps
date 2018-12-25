#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include "json.hpp"
#include "itensor/all.h"
#include "ctm-cluster-env_v2.h"
#include "cluster-ev-builder.h"
#include "ctm-cluster-io.h"
#include "ctm-cluster.h"
#include "mpo.h"
#include "models.h"
#include "itensor-svd-solvers.h"
#include "lapacksvd-solver.h"
#include "rsvd-solver.h"

using namespace itensor;

int main( int argc, char *argv[] ) {

    // ***** INITIALIZE CTMRG ALGORITHM *******************************
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
    double arg_envEps  = json_ctmrg_params["envEpsilon"].get<double>();
    bool arg_envDbg    = json_ctmrg_params["dbg"].get<bool>();
    int arg_envDbgLvl  = json_ctmrg_params["dbgLvl"].get<int>();
    // end reading CTMRG parameters

    // ***** INITIALIZE CTMRG ALGORITHM DONE **************************

	// ***** INITIALIZE CLUSTER ***********************************************
    Cluster cls;
    
    // choose initial wavefunction
    if (sitesInit == "FILE") {
        cls = readCluster(inClusterFile);
        // initClusterSites(cls);
        // initClusterWeights(cls);
        // setOnSiteTensorsFromFile(cls, inClusterFile);
    }
    std::cout << cls;
    // ***** INITIALIZE CLUSTER DONE ******************************************

    // ***** INITIALIZE MODEL *************************************************
    // DEFINE MODEL
    std::unique_ptr<Model> ptr_model;
    ptr_model = getModel(json_model_params);
    // ***** INITIALIZE MODEL DONE ********************************************

    // *****
    std::ofstream out_file_energy(outClusterFile+".energy.dat", std::ios::out);
    std::ofstream out_file_diag(outClusterFile+".diag.dat", std::ios::out);
    out_file_energy.precision( std::numeric_limits< double >::max_digits10 );
    out_file_diag.precision( std::numeric_limits< double >::max_digits10 );

    using time_point = std::chrono::steady_clock::time_point;
    time_point t_iso_begin, t_iso_end;
    auto get_s = [](time_point ti, time_point tf) { return std::chrono::duration_cast
            <std::chrono::microseconds>(tf - ti).count()/1.0e+06; };

    // Diagnostic data
    std::vector<int> diag_ctmIter;
    std::vector< Args > diagData_ctmrg;

    // hold energies
    std::vector<double> e_curr(4,0.0), e_prev(4,0.0);

    std::vector<double> accT(8,0.0); // holds timings for CTM moves
    time_point t_begin_int, t_end_int;
    // *****

    // ***** INITIALIZE ENVIRONMENT *******************************************

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

    CtmEnv ctmEnv(arg_ioEnvTag, auxEnvDim, cls, *pSvdSolver,
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
    EVBuilder ev(arg_ioEnvTag, cls, ctmEnv);
    
    std::vector<double> diag_minCornerSV(1, 0.);
    bool expValEnvConv = false;
    // PERFORM CTMRG
    for (int envI=1; envI<=arg_maxEnvIter; envI++ ) {
        t_begin_int = std::chrono::steady_clock::now();

        ctmEnv.move_unidirectional(CtmEnv::DIRECTION::LEFT, iso_type, accT);
        ctmEnv.move_unidirectional(CtmEnv::DIRECTION::UP, iso_type, accT);
        ctmEnv.move_unidirectional(CtmEnv::DIRECTION::RIGHT, iso_type, accT);
        ctmEnv.move_unidirectional(CtmEnv::DIRECTION::DOWN, iso_type, accT);

        t_end_int = std::chrono::steady_clock::now();
        std::cout << "CTM STEP " << envI <<" T: "<< get_s(t_begin_int,t_end_int) <<" [sec] "; 

        // CHECK CONVERGENCE
        if ( (arg_maxEnvIter > 1) && (envI % 1 == 0) ) {
            t_begin_int = std::chrono::steady_clock::now();
            
            e_curr[0]=ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(0,0), Vertex(1,0), true);
            e_curr[1]=ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(0,0), Vertex(0,1), true);
            e_curr[2]=ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(1,0), Vertex(1,1), true);
            e_curr[3]=ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(0,1), Vertex(1,1), true);

            t_end_int = std::chrono::steady_clock::now();

            std::cout<<" || E in T: "<< get_s(t_begin_int,t_end_int) <<" [sec] E: "
                << e_curr[0] <<" "<< e_curr[1] <<" "<< e_curr[2] <<" "<< e_curr[3]; 

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

            // Perform loop termination
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
    // ***** CTMRG DONE **************************************
    std::cout <<"Timings(CTMRG) :"<<"Projectors "<<"AbsorbReduce "<<"N/A "<<"Postprocess"<< std::endl;
    std::cout <<"accT [mSec]: "<< accT[0] <<" "<< accT[1] <<" "<< accT[2]
        <<" "<< accT[3] << std::endl;
    std::cout <<"Timings(Projectors) :"<<"Enlarge "<<"N/A "<<"SVD "<<"Contract"<< std::endl;
    std::cout <<"isoZ [mSec]: "<< accT[4] <<" "<< accT[5] <<" "<< accT[6]
        <<" "<< accT[7] << std::endl;


    // Compute final observables
    ptr_model->setObservablesHeader(out_file_energy);
    auto metaInf = Args("lineNo",0);
    t_begin_int = std::chrono::steady_clock::now();
    ptr_model->computeAndWriteObservables(ev, out_file_energy, metaInf);
    t_end_int = std::chrono::steady_clock::now();
    std::cout << "Observables computed in T: "<< get_s(t_begin_int,t_end_int) 
        <<" [sec] "<< std::endl;
    
    // FINISHED
    std::cout <<"FINISHED"<< std::endl;
}
