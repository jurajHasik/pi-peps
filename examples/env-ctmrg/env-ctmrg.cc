#include "pi-peps/config.h"
#include "json.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
DISABLE_WARNINGS
#include "itensor/all.h"
ENABLE_WARNINGS
#include "pi-peps/cluster-ev-builder.h"
#include "pi-peps/ctm-cluster-io.h"
#include "pi-peps/ctm-cluster.h"
#include "pi-peps/ctm-env.h"
#include "pi-peps/model-factory.h"
#include "pi-peps/mpo.h"
#include "pi-peps/svdsolver-factory.h"
#include "pi-peps/transfer-op.h"

using namespace itensor;

int main(int argc, char* argv[]) {
  // ***** INITIALIZE CTMRG ALGORITHM *******************************
  std::string arg_simFile = std::string(argv[1]);
  std::ifstream simfile(arg_simFile, std::ios::in);

  nlohmann::json jsonCls;
  simfile >> jsonCls;

  // write simulation parameters to log file
  std::cout << jsonCls.dump(4) << std::endl;

  // read cluster infile OR initialize by one of the predefined
  // options FILE, RND, RND_AB, AFM, RVB, ...
  auto json_cluster(jsonCls["cluster"]);
  std::string initBy(json_cluster["initBy"].get<std::string>());
  std::string inClusterFile = json_cluster.value("inClusterFile", "DEFAULT");
  int physDim = json_cluster["physDim"].get<int>();
  int auxBondDim = json_cluster["auxBondDim"].get<int>();

  // read cluster outfile
  std::string outClusterFile(jsonCls["outClusterFile"].get<std::string>());

  // read Hamiltonian and Trotter decomposition
  auto json_model_params(jsonCls["model"]);
  json_model_params["physDim"] = physDim;

  std::string sitesInit(jsonCls.value("sitesInit", "FILE"));

  // read CTMRG parameters
  auto json_ctmrg_params(jsonCls["ctmrg"]);
  int auxEnvDim = json_ctmrg_params["auxEnvDim"].get<int>();
  CtmEnv::init_env_type arg_initEnvType(
    toINIT_ENV(json_ctmrg_params["initEnvType"].get<std::string>()));
  CtmEnv::isometry_type iso_type(
    toISOMETRY(json_ctmrg_params["isoType"].get<std::string>()));
  double arg_isoPseudoInvCutoff =
    json_ctmrg_params["isoPseudoInvCutoff"].get<double>();
  CtmEnv::normalization_type norm_type(
    toNORMALIZATION(json_ctmrg_params["normType"].get<std::string>()));
  std::string env_SVD_METHOD(
    json_ctmrg_params["env_SVD_METHOD"].get<std::string>());
  auto rsvd_power = json_ctmrg_params.value("rsvd_power", 2);
  auto rsvd_reortho = json_ctmrg_params.value("rsvd_reortho", 1);
  auto rsvd_oversampling = json_ctmrg_params.value("rsvd_oversampling", 10);
  int arg_maxEnvIter = json_ctmrg_params["maxEnvIter"].get<int>();
  double arg_envEps = json_ctmrg_params["envEpsilon"].get<double>();
  bool arg_envDbg = json_ctmrg_params["dbg"].get<bool>();
  int arg_envDbgLvl = json_ctmrg_params["dbgLvl"].get<int>();
  // end reading CTMRG parameters

  // ***** INITIALIZE CTMRG ALGORITHM DONE **************************

  // ***** INITIALIZE CLUSTER ***********************************************
  // Cluster cls;
  std::unique_ptr<Cluster> p_cls;

  // choose initial wavefunction
  if (initBy == "FILE" and inClusterFile != "DEFAULT") {
    std::ifstream infile(inClusterFile, std::ios::in);
    nlohmann::json json_cluster_file = nlohmann::json::parse(infile);

    // preprocess parameters of input cluster
    // set initBy to FILE
    json_cluster_file["initBy"] = "FILE";
    json_cluster_file["auxBondDim"] = auxBondDim;
    for (auto& site : json_cluster_file["sites"]) {
      site["auxDim"] = auxBondDim;
    }

    p_cls = p_readCluster(json_cluster_file);
    // initClusterSites(cls);
    // initClusterWeights(cls);
    // setWeights(*p_cls, suWeightsInit);
    // setOnSiteTensorsFromFile(cls, inClusterFile);
  } else if (initBy == "FILE" and inClusterFile == "DEFAULT") {
    throw std::runtime_error("No cluster input file  given for inClusterFile");
  } else {
    ClusterFactory cf = ClusterFactory();
    p_cls = cf.create(json_cluster);
  }
  p_cls->normalize();
  std::cout << *p_cls;
  // ***** INITIALIZE CLUSTER DONE ******************************************

  // ***** INITIALIZE MODEL *************************************************
  // DEFINE MODEL
  ModelFactory mf = ModelFactory();
  auto ptr_model = mf.create(json_model_params);
  // ***** INITIALIZE MODEL DONE ********************************************

  // *****
  std::ofstream out_file_energy(outClusterFile + ".energy.dat", std::ios::out);
  std::ofstream out_file_diag(outClusterFile + ".diag.dat", std::ios::out);
  out_file_energy.precision(std::numeric_limits<double>::max_digits10);
  out_file_diag.precision(std::numeric_limits<double>::max_digits10);

  using time_point = std::chrono::steady_clock::time_point;
  time_point t_iso_begin, t_iso_end;
  auto get_s = [](time_point ti, time_point tf) {
    return std::chrono::duration_cast<std::chrono::microseconds>(tf - ti)
             .count() /
           1.0e+06;
  };

  // Diagnostic data
  std::vector<int> diag_ctmIter;
  std::vector<Args> diagData_ctmrg;

  // hold energies
  std::vector<double> e_curr(4, 0.0), e_prev(4, 0.0);

  std::vector<double> accT(12, 0.0);  // holds timings for CTM moves
  time_point t_begin_int, t_end_int;
  // *****

  // ***** INITIALIZE ENVIRONMENT *******************************************

  // ***** Select SVD solver to use *****************************************
  SvdSolverFactory sf = SvdSolverFactory();
  auto pSvdSolver = sf.create(env_SVD_METHOD);

  // CtmEnv ctmEnv(arg_ioEnvTag, auxEnvDim, cls, *pSvdSolver,
  CtmEnv ctmEnv("default", auxEnvDim, *p_cls, *pSvdSolver,
                {"isoPseudoInvCutoff", arg_isoPseudoInvCutoff, "SVD_METHOD",
                 env_SVD_METHOD, "rsvd_power", rsvd_power, "rsvd_reortho",
                 rsvd_reortho, "rsvd_oversampling", rsvd_oversampling, "dbg",
                 arg_envDbg, "dbgLevel", arg_envDbgLvl});
  ctmEnv.init(arg_initEnvType, false, arg_envDbg);

  // INITIALIZE EXPECTATION VALUE BUILDER
  // EVBuilder ev(arg_ioEnvTag, cls, ctmEnv);
  EVBuilder ev("default", *p_cls, ctmEnv);
  std::cout << ev;

  ptr_model->setObservablesHeader(out_file_energy);

  std::vector<double> diag_minCornerSV(1, 0.);
  int max_moves = 1;
  bool expValEnvConv = false;
  // PERFORM CTMRG
  for (int envI = 1; envI <= arg_maxEnvIter; envI++) {
    t_begin_int = std::chrono::steady_clock::now();

    ctmEnv.move_unidirectional(CtmEnv::DIRECTION::LEFT, iso_type, accT, max_moves);
    // ctmEnv.move_unidirectional(CtmEnv::DIRECTION::UP, iso_type, accT);
    ctmEnv.move_unidirectional(CtmEnv::DIRECTION::RIGHT, iso_type, accT, max_moves);
    ctmEnv.move_unidirectional(CtmEnv::DIRECTION::UP, iso_type, accT, max_moves);
    ctmEnv.move_unidirectional(CtmEnv::DIRECTION::DOWN, iso_type, accT, max_moves);

    t_end_int = std::chrono::steady_clock::now();
    std::cout << "CTM STEP " << envI << " T: " << get_s(t_begin_int, t_end_int)
              << " [sec] ";

    // CHECK CONVERGENCE
    if ((arg_maxEnvIter > 1) && (envI % 1 == 0)) {
      t_begin_int = std::chrono::steady_clock::now();

      e_curr[0] =
        analyzeBoundaryVariance(ev, Vertex(0, 0), CtmEnv::DIRECTION::RIGHT);
      e_curr[1] =
        analyzeBoundaryVariance(ev, Vertex(0, 0), CtmEnv::DIRECTION::DOWN);
      e_curr[2] =
        analyzeBoundaryVariance(ev, Vertex(1, 1), CtmEnv::DIRECTION::RIGHT);
      e_curr[3] =
        analyzeBoundaryVariance(ev, Vertex(1, 1), CtmEnv::DIRECTION::DOWN);

      t_end_int = std::chrono::steady_clock::now();
      std::cout << " || Var(boundary) in T: " << get_s(t_begin_int, t_end_int)
                << " [sec] : " << e_curr[0] << " " << e_curr[1] << " "
                << e_curr[2] << " " << e_curr[3];
      out_file_diag << envI <<" "<< e_curr[0] << " " << e_curr[1] << " "
                << e_curr[2] << " " << e_curr[3] << std::endl;

      auto metaInf = Args("lineNo", envI);
      t_begin_int = std::chrono::steady_clock::now();
      ptr_model->computeAndWriteObservables(ev, out_file_energy, metaInf);
      t_end_int = std::chrono::steady_clock::now();

      std::cout << " || Observables in T: " << get_s(t_begin_int, t_end_int)
                << " [sec] " << std::endl;

      // if the difference between energies along NN links is lower then
      // arg_envEps consider the environment converged
      if ((std::abs(e_prev[0] - e_curr[0]) < arg_envEps) &&
          (std::abs(e_prev[1] - e_curr[1]) < arg_envEps) &&
          (std::abs(e_prev[2] - e_curr[2]) < arg_envEps) &&
          (std::abs(e_prev[3] - e_curr[3]) < arg_envEps)) {
        expValEnvConv = true;
        std::cout << " ENV CONVERGED ";
      }

      if (envI == arg_maxEnvIter) {
        expValEnvConv = true;
        std::cout << " MAX ENV iterations REACHED ";
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
        Args args_dbg_cornerSVD = {"Truncate", false};
        std::cout << "Spectra: " << std::endl;

        ITensor tL(ctmEnv.C_LU.at(ctmEnv.p_cluster->siteIds[0]).inds().front()),
          sv, tR;
        auto spec = svd(ctmEnv.C_LU.at(ctmEnv.p_cluster->siteIds[0]), tL, sv,
                        tR, args_dbg_cornerSVD);
        tmpVal =
          sv.real(sv.inds().front()(auxEnvDim), sv.inds().back()(auxEnvDim));
        PrintData(sv);
        minCornerSV = std::min(minCornerSV, tmpVal);
        oss << tmpVal;

        tL =
          ITensor(ctmEnv.C_RU.at(ctmEnv.p_cluster->siteIds[0]).inds().front());
        spec = svd(ctmEnv.C_RU.at(ctmEnv.p_cluster->siteIds[0]), tL, sv, tR,
                   args_dbg_cornerSVD);
        tmpVal =
          sv.real(sv.inds().front()(auxEnvDim), sv.inds().back()(auxEnvDim));
        PrintData(sv);
        minCornerSV = std::min(minCornerSV, tmpVal);
        oss << " " << tmpVal;

        tL =
          ITensor(ctmEnv.C_RD.at(ctmEnv.p_cluster->siteIds[0]).inds().front());
        spec = svd(ctmEnv.C_RD.at(ctmEnv.p_cluster->siteIds[0]), tL, sv, tR,
                   args_dbg_cornerSVD);
        tmpVal =
          sv.real(sv.inds().front()(auxEnvDim), sv.inds().back()(auxEnvDim));
        PrintData(sv);
        minCornerSV = std::min(minCornerSV, tmpVal);
        oss << " " << tmpVal;

        tL =
          ITensor(ctmEnv.C_LD.at(ctmEnv.p_cluster->siteIds[0]).inds().front());
        spec = svd(ctmEnv.C_LD.at(ctmEnv.p_cluster->siteIds[0]), tL, sv, tR,
                   args_dbg_cornerSVD);
        tmpVal =
          sv.real(sv.inds().front()(auxEnvDim), sv.inds().back()(auxEnvDim));
        PrintData(sv);
        minCornerSV = std::min(minCornerSV, tmpVal);
        oss << " " << tmpVal;

        diag_minCornerSV.push_back(minCornerSV);
        std::cout << "MinVals: " << oss.str() << std::endl;

        break;
      }
    }
    std::cout << std::endl;
  }
  // ***** CTMRG DONE **************************************
  std::cout << "Timings(CTMRG) :"
            << "Projectors "
            << "AbsorbReduce "
            << "N/A "
            << "Postprocess" << std::endl;
  std::cout << "accT [mSec]: " << accT[0] << " " << accT[1] << " " << accT[2]
            << " " << accT[3] << std::endl;
  std::cout << "Timings(Projectors): "
            << "Enlarge "
            << "N/A "
            << "SVD "
            << "Contract" << std::endl;
  std::cout << "isoZ [mSec]: " << accT[4] << " " << accT[5] << " " << accT[6]
            << " " << accT[7] << std::endl;
  std::cout << "Timings(AbsorbReduce): "
            << "C "
            << "T "
            << "Ct "
            << "N/A" << std::endl;
  std::cout << "[mSec]: " << accT[8] << " " << accT[9] << " " << accT[10] << " "
            << accT[11] << std::endl;

  // Compute final observables
  auto metaInf = Args("lineNo", arg_maxEnvIter);
  t_begin_int = std::chrono::steady_clock::now();
  ptr_model->computeAndWriteObservables(ev, out_file_energy, metaInf);
  t_end_int = std::chrono::steady_clock::now();
  std::cout << "Observables computed in T: " << get_s(t_begin_int, t_end_int)
            << " [sec] " << std::endl;

  // compute spin directions
  std::vector<double> svec(3, 0.0);
  std::cout << "siteId sz sp sm" << std::endl;
  for (auto const& id : p_cls->siteIds) {
    svec[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, p_cls->idToV.at(id));
    svec[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, p_cls->idToV.at(id));
    svec[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, p_cls->idToV.at(id));

    std::cout << id << " " << svec[0] << " " << svec[1] << " " << svec[2]
              << std::endl;
  }

#ifdef PEPS_WITH_ARPACK
  std::cout << "Transfer matrix spectrum analysis: " << std::endl;
  analyzeTransferMatrix(ev, Vertex(0, 0), CtmEnv::DIRECTION::RIGHT, 5);
  analyzeTransferMatrix(ev, Vertex(0, 0), CtmEnv::DIRECTION::DOWN, 5);
#endif

  std::cout << "S(0,0).S(r,0)"<< std::endl;
  auto ss_h = ev.corrf_SS(Vertex(0,0), CtmEnv::DIRECTION::RIGHT, 20);
  for(int i=0; i<20; i++)
    std::cout<< i <<" "<< ss_h[i].real() << std::endl;
  std::cout << "S(0,0).S(0,r)"<< std::endl;
  auto ss_v = ev.corrf_SS(Vertex(0,0), CtmEnv::DIRECTION::DOWN, 20);
  for(int i=0; i<20; i++)
    std::cout<< i <<" "<< ss_v[i].real() << std::endl;

  // FINISHED
  std::cout << "FINISHED" << std::endl;
}
