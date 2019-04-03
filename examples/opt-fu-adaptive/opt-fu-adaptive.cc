#include "pi-peps/config.h"
#include "json.hpp"
#include "pi-peps/cluster-ev-builder.h"
#include "pi-peps/cluster-factory.h"
#include "pi-peps/ctm-cluster-basic.h"
#include "pi-peps/ctm-cluster-io.h"
#include "pi-peps/ctm-env.h"
#include "pi-peps/engine-factory.h"
#include "pi-peps/linalg/linsyssolvers-lapack.h"
#include "pi-peps/model-factory.h"
#include "pi-peps/mpo.h"
#include "pi-peps/svdsolver-factory.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

using namespace itensor;

int main(int argc, char* argv[]) {
  // ***** INITIALIZE FULL UPDATE ALGORITHM *********************************
  std::string arg_initFile = std::string(argv[1]);
  std::ifstream simfile(arg_initFile, std::ios::in);

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

  double initStateNoise = jsonCls.value("initStateNoise", 0.0);

  // read cluster outfile
  std::string outClusterFile(jsonCls["outClusterFile"].get<std::string>());

  // read Hamiltonian and Trotter decomposition
  auto json_model_params(jsonCls["model"]);
  json_model_params["physDim"] = physDim;
  bool symmTrotter = json_model_params.value("symmTrotter", true);
  bool randomizeSeq = json_model_params.value("randomizeSeq", false);

  // full update parameters
  bool arg_decreaseTimestep = jsonCls.value("decreaseTimestep", true);
  double arg_dtFraction = jsonCls.value("dtFraction", 0.5);
  double arg_minTimestep = jsonCls.value("minTimestep", 1.0e-6);
  int arg_fuIter = jsonCls["fuIter"].get<int>();
  int arg_obsFreq = jsonCls["obsFreq"].get<int>();
  bool arg_fuTrialInit = jsonCls.value("fuTrialInit", true);
  bool arg_fuDbg = jsonCls["fuDbg"].get<bool>();
  int arg_fuDbgLevel = jsonCls["fuDbgLevel"].get<int>();
  std::string arg_otNormType = jsonCls["otNormType"].get<std::string>();

  int arg_maxAltLstSqrIter = jsonCls["maxAltLstSqrIter"].get<int>();
  bool symmetrizeProtoEnv = jsonCls["symmetrizeProtoEnv"].get<bool>();
  bool posDefProtoEnv = jsonCls["positiveDefiniteProtoEnv"].get<bool>();
  // Iterative ALS procedure
  double epsdistf = jsonCls.value("epsdistf", 1.0e-8);
  auto json_als_params(jsonCls["als"]);
  std::string linsolver = json_als_params.value("solver", "UNSUPPORTED");
  bool solver_dbg = json_als_params.value("dbg", false);
  double epsregularisation = json_als_params.value("epsregularisation", 0.0);

  // direct linear solver params
  std::string als_ds_method = json_als_params.value("method", "LU");

  // pseudoinverse solver params
  double pseudoInvCutoff = json_als_params.value("pseudoInvCutoff", 1.0e-8);
  double pseudoInvCutoffInsert =
    json_als_params.value("pseudoInvCutoffInsert", 0.0);

  // legacy
  // double pseudoInvMaxLogGap = jsonCls["pseudoInvMaxLogGap"].get<double>();
  bool dynamicEps = jsonCls.value("dynamicEps", false);

  // gauge Fixing by simple update with identity operators
  bool arg_su_gauge_fix = jsonCls.value("suGaugeFix", false);
  int arg_su_gauge_fix_freq = jsonCls.value("suGaugeFixFreq", arg_obsFreq);
  auto json_gauge_fix_params(jsonCls["gaugeFix"]);
  std::string arg_suWeightsInit =
    json_gauge_fix_params.value("suWeightsInit", "DELTA");
  int arg_suIter = json_gauge_fix_params.value("suIter", 128);
  double arg_suTol = json_gauge_fix_params.value("suTol", 1.0e-12);
  bool arg_gf_dbg = json_gauge_fix_params.value("suDbg", false);
  int arg_gf_dbgLvl = json_gauge_fix_params.value("suDbgLevel", 0);

  // read CTMRG parameters
  auto json_ctmrg_params(jsonCls["ctmrg"]);
  int auxEnvDim = json_ctmrg_params["auxEnvDim"].get<int>();
  CtmEnv::init_env_type arg_initEnvType(
    toINIT_ENV(json_ctmrg_params["initEnvType"].get<std::string>()));
  bool envIsComplex = json_ctmrg_params.value("envIsComplex",false);
  CtmEnv::isometry_type iso_type(
    toISOMETRY(json_ctmrg_params["isoType"].get<std::string>()));
  CtmEnv::normalization_type norm_type(
    toNORMALIZATION(json_ctmrg_params["normType"].get<std::string>()));
  std::string env_SVD_METHOD(
    json_ctmrg_params["env_SVD_METHOD"].get<std::string>());
  auto rsvd_power = json_ctmrg_params.value("rsvd_power", 2);
  auto rsvd_reortho = json_ctmrg_params.value("rsvd_reortho", 1);
  auto rsvd_oversampling = json_ctmrg_params.value("rsvd_oversampling", 10);
  int arg_maxEnvIter = json_ctmrg_params["maxEnvIter"].get<int>();
  int arg_maxInitEnvIter = json_ctmrg_params["initMaxEnvIter"].get<int>();
  int arg_obsMaxIter =
    json_ctmrg_params.value("obsMaxIter", arg_maxInitEnvIter);
  double arg_envEps = json_ctmrg_params["envEpsilon"].get<double>();
  bool arg_reinitEnv = json_ctmrg_params["reinitEnv"].get<bool>();
  bool arg_reinitObsEnv = json_ctmrg_params.value("reinitObsEnv", false);
  bool arg_envDbg = json_ctmrg_params["dbg"].get<bool>();
  int arg_envDbgLvl = json_ctmrg_params["dbgLvl"].get<int>();
  // end reading CTMRG parameters
  // ***** INITIALIZE FULL UPDATE ALGORITHM DONE ****************************

  // ----- INITIALIZE CLUSTER -----------------------------------------------
  std::unique_ptr<Cluster> p_cls;

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
  std::cout << *p_cls;

  // add random noise to initial state
  {
    ITensor temp;
    double eps = initStateNoise;
    auto setMeanTo0 = [](Real r) { return (r - 0.5); };

    for (auto& st : p_cls->sites) {
      temp = st.second;
      randomize(temp);
      temp.apply(setMeanTo0);
      st.second += eps * temp;
    }
  }

  // write simulations params into cluster
  p_cls->simParam = jsonCls;

  for (int y = -3; y < 4; y++) {
    for (int x = -3; x < 4; x++) {
      std::cout << "[" << x << "," << y
                << "]=" << p_cls->vertexToId(Vertex(x, y));
    }
    std::cout << std::endl;
  }
  // ----- END DEFINE CLUSTER -----------------------------------------------

  // ***** Select SVD solver to use *****************************************
  SvdSolverFactory sf = SvdSolverFactory();
  auto pSvdSolver = sf.create(env_SVD_METHOD);

  // ***** Select LinSys solver to use **************************************
  std::unique_ptr<LinSysSolver> pLinSysSolver;
  if (linsolver == "cholesky") {
    pLinSysSolver = std::unique_ptr<CholeskySolver>(new CholeskySolver());
  } else if (linsolver == "pseudoinverse") {
    pLinSysSolver = std::unique_ptr<PseudoInvSolver>(new PseudoInvSolver());
  } else {
    std::cout << "WARNING: Unsupported LinSysSolver specified. Using default"
              << std::endl;
    pLinSysSolver = std::unique_ptr<PseudoInvSolver>(new PseudoInvSolver());
  }

  // INITIALIZE ENVIRONMENT
  auto env_args = Args("Name", "CTMRG_parameters");
  for (nlohmann::json::iterator it = json_ctmrg_params.begin();
       it != json_ctmrg_params.end(); ++it) {
    std::string key = it.key();
    auto val = it.value();
    if (val.is_string())
      env_args.add(key, val.get<std::string>());
    else if (val.is_boolean())
      env_args.add(key, (bool)val);
    else if (val.is_number_integer())
      env_args.add(key, (int)val);
    else if (val.is_number_float())
      env_args.add(key, (double)val);
  }

  CtmEnv ctmEnv("default", auxEnvDim, *p_cls, *pSvdSolver, env_args);

  // INITIALIZE EXPECTATION VALUE BUILDER
  EVBuilder ev("default", *p_cls, ctmEnv);
  std::cout << ev;

  // hold ctm observables used in the convergence criterion
  std::vector<double> e_curr(4, 0.0), e_prev(4, 0.0);

  using time_point = std::chrono::steady_clock::time_point;
  time_point t_begin_int, t_end_int;
  auto get_s = [](time_point ti, time_point tf) {
    return std::chrono::duration_cast<std::chrono::microseconds>(tf - ti)
             .count() /
           1.0e+06;
  };

  // ##################################################################
  // # SETUP OPTIMIZATION LOOP                                        #
  // ##################################################################

  // DEFINE MODEL AND GATE SEQUENCE
  ModelFactory mf = ModelFactory();
  EngineFactory ef = EngineFactory();
  auto ptr_model = mf.create(json_model_params);
  auto ptr_engine =
    ef.build(json_model_params, pLinSysSolver.get());  // full update engine
  std::unique_ptr<Engine> ptr_gfe = nullptr;
  if (arg_su_gauge_fix)
    ptr_gfe = ef.build(json_gauge_fix_params);  // gauge fixing engine

  Args fuArgs = {
    "maxAltLstSqrIter",
    arg_maxAltLstSqrIter,
    "fuTrialInit",
    arg_fuTrialInit,
    "fuDbg",
    arg_fuDbg,
    "fuDbgLevel",
    arg_fuDbgLevel,
    "symmetrizeProtoEnv",
    symmetrizeProtoEnv,
    "positiveDefiniteProtoEnv",
    posDefProtoEnv,
    "otNormType",
    arg_otNormType,
    "epsdistf",
    epsdistf,

    "solver",
    linsolver,
    "dbg",
    solver_dbg,
    "epsregularisation",
    epsregularisation,

    "method",
    als_ds_method,

    "pseudoInvCutoff",
    pseudoInvCutoff,
    "pseudoInvCutoffInsert",
    pseudoInvCutoffInsert,
  };
  // Diagnostic data
  std::vector<std::string> diag_log;

  Args diagData_ctm;
  std::vector<Args> diagData_fu;

  double best_energy = 1.0e+16;
  std::vector<double> accT(12, 0.0);  // holds timings for CTM moves
  std::vector<double> past_energy;
  Args diag_fu;

  auto past_tensors = p_cls->sites;

  std::string outClusterBestFile = outClusterFile + ".best";
  std::ofstream out_file_energy(outClusterFile + ".energy.dat", std::ios::out);
  std::ofstream out_file_diag(outClusterFile + ".diag.dat", std::ios::out);
  out_file_energy.precision(std::numeric_limits<double>::max_digits10);
  out_file_diag.precision(std::numeric_limits<double>::max_digits10);

  auto printBondSpectra = [&p_cls] {
    auto printS = [](Real r) { std::cout << std::scientific << r << " "; };

    std::cout << "BOND SPECTRA - START" << std::endl;
    // loop over link weights and perform svd uniquely
    std::vector<std::string> lwIds;
    for (auto const& stw : p_cls->siteToWeights)
      for (auto const& lw : stw.second)
        if (std::find(lwIds.begin(), lwIds.end(), lw.wId) == lwIds.end()) {
          nlohmann::json jentry;
          jentry["sites"] = lw.sId;
          jentry["directions"] = lw.dirs;
          jentry["weightId"] = lw.wId;

          std::cout << lw.sId[0] << "-" << lw.dirs[0] << "--" << lw.dirs[1]
                    << "-" << lw.sId[1] << " ";

          std::vector<Index> indsL;
          for (int i = 0; i < 4; i++)
            if (i != lw.dirs[0])
              indsL.push_back(p_cls->AIc(lw.sId[0], i));
          indsL.push_back(p_cls->mphys.at(lw.sId[0]));
          ITensor tmpL(indsL), S, tmpR;

          auto tmpT = p_cls->sites.at(lw.sId[0]) *
                      delta(p_cls->AIc(lw.sId[0], lw.dirs[0]),
                            p_cls->AIc(lw.sId[1], lw.dirs[1])) *
                      p_cls->sites.at(lw.sId[1]);

          svd(tmpT, tmpL, S, tmpR,
              {"Minm", p_cls->AIc(lw.sId[0], lw.dirs[0]).m(), "Maxm",
               p_cls->AIc(lw.sId[0], lw.dirs[0]).m()});
          S *= 1.0 / S.real(S.inds()[0](1), S.inds()[1](1));
          S.visit(printS);
          std::cout << std::endl;

          lwIds.push_back(lw.wId);
        }
    std::cout << "BOND SPECTRA - END" << std::endl;
  };

  auto computeEnvironment = [&ctmEnv, &ev, &iso_type, &arg_envEps,
                             &arg_initEnvType, &envIsComplex, &arg_envDbg,
                             &arg_envDbgLvl,
                             &get_s](int maxIter, bool reinitEnv) {
    time_point t_begin_int, t_end_int;
    std::vector<double> accT(12, 0.0);
    std::vector<double> e_curr(4, 0.0), e_prev(4, 0.0);
    bool expValEnvConv = false;

    Args diagData_ctm = Args::global();

    if (reinitEnv)
      ctmEnv.init(arg_initEnvType, envIsComplex, arg_envDbg);

    for (int envI = 1; envI <= maxIter; envI++) {
      t_begin_int = std::chrono::steady_clock::now();

      ctmEnv.move_unidirectional(CtmEnv::DIRECTION::LEFT, iso_type, accT);
      ctmEnv.move_unidirectional(CtmEnv::DIRECTION::UP, iso_type, accT);
      ctmEnv.move_unidirectional(CtmEnv::DIRECTION::RIGHT, iso_type, accT);
      ctmEnv.move_unidirectional(CtmEnv::DIRECTION::DOWN, iso_type, accT);

      t_end_int = std::chrono::steady_clock::now();
      std::cout << "CTM STEP " << envI
                << " T: " << get_s(t_begin_int, t_end_int) << " [sec] ";

      if (envI % 1 == 0) {
        t_begin_int = std::chrono::steady_clock::now();
        e_curr[0] =
          ev.analyzeBoundaryVariance(Vertex(0, 0), CtmEnv::DIRECTION::RIGHT);
        e_curr[1] =
          ev.analyzeBoundaryVariance(Vertex(0, 0), CtmEnv::DIRECTION::DOWN);
        e_curr[2] =
          ev.analyzeBoundaryVariance(Vertex(1, 1), CtmEnv::DIRECTION::RIGHT);
        e_curr[3] =
          ev.analyzeBoundaryVariance(Vertex(1, 1), CtmEnv::DIRECTION::DOWN);
        t_end_int = std::chrono::steady_clock::now();

        std::cout << " || Var(boundary) in T: " << get_s(t_begin_int, t_end_int)
                  << " [sec] : " << e_curr[0] << " " << e_curr[1] << " "
                  << e_curr[2] << " " << e_curr[3] << std::endl;

        if ((std::abs(e_prev[0] - e_curr[0]) < arg_envEps) &&
            (std::abs(e_prev[1] - e_curr[1]) < arg_envEps) &&
            (std::abs(e_prev[2] - e_curr[2]) < arg_envEps) &&
            (std::abs(e_prev[3] - e_curr[3]) < arg_envEps)) {
          std::cout << "INIT ENV CONVERGED" << std::endl;
          expValEnvConv = true;
        }

        if (envI == maxIter) {
          std::cout << " MAX ENV iterations REACHED ";
          expValEnvConv = true;
        }
        e_prev = e_curr;

        if (expValEnvConv) {
          // maximal value of transfer-op variance
          std::vector<double>::iterator result =
            std::max_element(std::begin(e_curr), std::end(e_curr));
          auto max_boundaryVar = *result;

          std::ostringstream oss;
          oss << std::scientific;

          // Compute spectra of Corner matrices
          std::cout << std::endl;
          double tmpVal;
          double max_tailCornerSV = 0.0;
          Args args_dbg_cornerSVD = {"Truncate", false};
          std::cout << "Spectra: " << std::endl;

          ITensor tL(
            ctmEnv.C_LU.at(ctmEnv.p_cluster->siteIds[0]).inds().front()),
            sv, tR;
          auto spec = svd(ctmEnv.C_LU.at(ctmEnv.p_cluster->siteIds[0]), tL, sv,
                          tR, args_dbg_cornerSVD);
          tmpVal =
            sv.real(sv.inds().front()(ctmEnv.x), sv.inds().back()(ctmEnv.x));
          if (arg_envDbg)
            PrintData(sv);
          max_tailCornerSV = std::max(max_tailCornerSV, tmpVal);
          oss << tmpVal;

          tL = ITensor(
            ctmEnv.C_RU.at(ctmEnv.p_cluster->siteIds[0]).inds().front());
          spec = svd(ctmEnv.C_RU.at(ctmEnv.p_cluster->siteIds[0]), tL, sv, tR,
                     args_dbg_cornerSVD);
          tmpVal =
            sv.real(sv.inds().front()(ctmEnv.x), sv.inds().back()(ctmEnv.x));
          if (arg_envDbg)
            PrintData(sv);
          max_tailCornerSV = std::max(max_tailCornerSV, tmpVal);
          oss << " " << tmpVal;

          tL = ITensor(
            ctmEnv.C_RD.at(ctmEnv.p_cluster->siteIds[0]).inds().front());
          spec = svd(ctmEnv.C_RD.at(ctmEnv.p_cluster->siteIds[0]), tL, sv, tR,
                     args_dbg_cornerSVD);
          tmpVal =
            sv.real(sv.inds().front()(ctmEnv.x), sv.inds().back()(ctmEnv.x));
          if (arg_envDbg)
            PrintData(sv);
          max_tailCornerSV = std::max(max_tailCornerSV, tmpVal);
          oss << " " << tmpVal;

          tL = ITensor(
            ctmEnv.C_LD.at(ctmEnv.p_cluster->siteIds[0]).inds().front());
          spec = svd(ctmEnv.C_LD.at(ctmEnv.p_cluster->siteIds[0]), tL, sv, tR,
                     args_dbg_cornerSVD);
          tmpVal =
            sv.real(sv.inds().front()(ctmEnv.x), sv.inds().back()(ctmEnv.x));
          if (arg_envDbg)
            PrintData(sv);
          max_tailCornerSV = std::max(max_tailCornerSV, tmpVal);
          oss << " " << tmpVal;

          std::cout << "MinVals: " << oss.str() << std::endl;

          // record diagnostic data
          diagData_ctm =
            Args("ctmI", envI, "max_tailCornerSV", max_tailCornerSV,
                 "maxBoundaryVariance", max_boundaryVar);

          break;
        }
      }
    }

    if (arg_envDbg && (arg_envDbgLvl > 1)) {
      std::cout << "Timings(CTMRG) :"
                << "Projectors "
                << "AbsorbReduce "
                << "N/A "
                << "Postprocess" << std::endl;
      std::cout << "accT [mSec]: " << accT[0] << " " << accT[1] << " "
                << accT[2] << " " << accT[3] << std::endl;
      std::cout << "Timings(Projectors): "
                << "Enlarge "
                << "N/A "
                << "SVD "
                << "Contract" << std::endl;
      std::cout << "isoZ [mSec]: " << accT[4] << " " << accT[5] << " "
                << accT[6] << " " << accT[7] << std::endl;
      std::cout << "Timings(AbsorbReduce): "
                << "C "
                << "T "
                << "Ct "
                << "N/A" << std::endl;
      std::cout << "[mSec]: " << accT[8] << " " << accT[9] << " " << accT[10]
                << " " << accT[11] << std::endl;
    }

    return diagData_ctm;
  };

  // ***** COMPUTING INITIAL ENVIRONMENT ************************************
  std::cout << "COMPUTING INITIAL ENVIRONMENT " << std::endl;
  diagData_ctm = computeEnvironment(arg_maxInitEnvIter, true);
  // ***** COMPUTING INITIAL ENVIRONMENT DONE *******************************

  // Compute initial properties
  ptr_model->setObservablesHeader(out_file_energy);
  auto obs_metaInf = Args("lineNo", 0);
  ptr_model->computeAndWriteObservables(ev, out_file_energy, obs_metaInf);
  best_energy = obs_metaInf.getReal("energy");
  past_energy.push_back(best_energy);

  // ENTER OPTIMIZATION LOOP
  for (int fuI = 1; fuI <= arg_fuIter; fuI++) {
    std::cout << "Full Update - STEP " << fuI << std::endl;

    // ctmEnv.symmetrizeEnv();
    diag_fu = ptr_engine->performFullUpdate(*p_cls, ctmEnv, fuArgs);

    diagData_fu.push_back(diag_fu);

    if (fuI == 1) {
      out_file_diag << "fuI ctmIter alsSweep"
                    << " ";
      out_file_diag << diag_fu.getString("siteMaxElem_descriptor") << " ";
      if (diag_fu.getString("locMinDiag", "").length() > 0)
        out_file_diag << diag_fu.getString("locMinDiag_descriptor") << " ";
      if (diag_fu.getString("diag_protoEnv", "").length() > 0)
        out_file_diag << diag_fu.getString("diag_protoEnv_descriptor") << " ";
      out_file_diag << "max_tailCornerSV"
                    << " "
                    << "MaxBoundaryVar"
                    << " " << std::endl;
    }

    out_file_diag << fuI << " " << diagData_ctm.getInt("ctmI", -1) << " "
                  << diag_fu.getInt("alsSweep", 0) << " "
                  << diag_fu.getString("siteMaxElem");
    if (diag_fu.getString("locMinDiag", "").length() > 0)
      out_file_diag << " " << diag_fu.getString("locMinDiag", "");
    if (diag_fu.getString("diag_protoEnv", "").length() > 0)
      out_file_diag << " " << diag_fu.getString("diag_protoEnv", "");
    out_file_diag << " " << diagData_ctm.getReal("max_tailCornerSV", -1.0)
                  << " " << diagData_ctm.getReal("maxBoundaryVariance", -1.0)
                  << " " << diag_fu.getReal("ratioNonSymLE", 0.0) << " "
                  << diag_fu.getReal("ratioNonSymFN", 0.0) << " "
                  << diag_fu.getReal("minGapDisc", 0.0) << " "
                  << diag_fu.getReal("minEvKept", 0.0) << std::endl;

    // fix gauge by simple-update at dt=0 - identity operators
    if (arg_su_gauge_fix && (fuI % arg_su_gauge_fix_freq == 0)) {
      std::cout << "GAUGE FIXING" << std::endl;

      printBondSpectra();

      auto num_eps = std::numeric_limits<double>::epsilon();
      t_begin_int = std::chrono::steady_clock::now();

      int max_auxBondDim = 0;
      for (auto const& siteAuxI : p_cls->caux)
        for (auto const& ai : siteAuxI.second)
          max_auxBondDim = std::max(ai.m(), (long int)max_auxBondDim);
      auto const tol =
        (true) ? arg_suTol : num_eps * max_auxBondDim * p_cls->weights.size();

      // Assuming the weights have been initialized
      initClusterWeights(*p_cls);
      setWeights(*p_cls, arg_suWeightsInit);

      Args gfArgs = {"suDbg", arg_gf_dbg, "suDbgLevel", arg_gf_dbgLvl};
      Args gf_diag_fu;
      saveWeights(*p_cls);
      for (int suI = 1; suI <= arg_suIter; suI++) {
        // std::cout <<"Simple Update - STEP "<< suI << std::endl;

        // PERFORM SIMPLE UPDATE
        gf_diag_fu = ptr_gfe->performSimpleUpdate(*p_cls, gfArgs);

        // check convergence
        if (suI % 8 == 0) {
          auto weight_distance = weightDist(*p_cls);
          if (weight_distance < tol) {
            std::cout << "GF iter: " << suI << " dist: " << weight_distance
                      << " CONVERGED" << std::endl;
            break;
          } else {
            std::cout << "GF iter: " << suI << " dist: " << weight_distance
                      << std::endl;
          }
          saveWeights(*p_cls);
        }
      }

      p_cls->absorbWeightsToSites();
      p_cls->weights_absorbed = false;

      t_end_int = std::chrono::steady_clock::now();
      std::cout << "GUAGE FIX DONE"
                << " T: " << get_s(t_begin_int, t_end_int) << " [sec] ";
    }

    // SETUP ENVIRONMENT LOOP
    // reset environment ?
    bool reinitEnv =
      arg_reinitEnv || ((fuI % arg_obsFreq == 0) && arg_reinitObsEnv);
    int currentMaxEnvIter =
      (fuI % arg_obsFreq == 0) ? arg_obsMaxIter : arg_maxEnvIter;
    // ENTER ENVIRONMENT LOOP
    diagData_ctm = computeEnvironment(currentMaxEnvIter, reinitEnv);

    if (fuI % arg_obsFreq == 0) {
      t_begin_int = std::chrono::steady_clock::now();

      // ctmEnv.symmetrizeEnv();
      auto metaInf = Args("lineNo", fuI);
      ptr_model->computeAndWriteObservables(ev, out_file_energy, metaInf);

      // check energy, preserve the best_energy state obtained so far
      auto current_energy = metaInf.getReal("energy");
      if (best_energy > current_energy) {
        best_energy = current_energy;
        p_cls->metaInfo = "BestEnergy(FUStep=" + std::to_string(fuI) + ")";
        writeCluster(outClusterBestFile, *p_cls);
        past_tensors = p_cls->sites;
      }
      // check if current energy > previous energy
      if ((current_energy > best_energy) && arg_decreaseTimestep) {
        std::ostringstream oss;
        oss << std::scientific;
        oss << fuI << ": ENERGY INCREASED: E(i)-E(i-1)="
            << current_energy - best_energy;
        // TODO revert to previous tensors and recompute the environment
        oss << " Reverting to previous tensors";
        p_cls->sites = past_tensors;
        computeEnvironment(arg_maxInitEnvIter, true);
        // decrease time-step
        auto current_dt = json_model_params["tau"].get<double>();
        json_model_params["tau"] = current_dt * arg_dtFraction;
        jsonCls["model"] = json_model_params;
        oss << " Timestep decreased: " << current_dt << " -> "
            << current_dt * arg_dtFraction;
        // regenerate model with new lower timestep
        ptr_engine = ef.build(json_model_params, pLinSysSolver.get());
        // update simulation parameters on cluster
        p_cls->simParam = jsonCls;
        diag_log.push_back(oss.str());
        std::cout << oss.str() << std::endl;
        if (current_dt < arg_minTimestep) {
          std::cout << "Timstep too small. Stopping simulation" << std::endl;
          break;
        }
      }
      // past_energy.push_back(current_energy);

      t_end_int = std::chrono::steady_clock::now();

      std::cout << "Observables computed in T: "
                << get_s(t_begin_int, t_end_int) << " [sec] " << std::endl;

      // Compute spectra of Corner matrices
      if (arg_envDbg) {
        std::cout << std::endl;
        Args args_dbg_cornerSVD = {"Truncate", false};
        std::cout << "Spectra: " << std::endl;

        ITensor tL(ctmEnv.C_LU.at(ctmEnv.p_cluster->siteIds[0]).inds().front()),
          sv, tR;
        auto spec = svd(ctmEnv.C_LU.at(ctmEnv.p_cluster->siteIds[0]), tL, sv,
                        tR, args_dbg_cornerSVD);
        PrintData(sv);

        tL =
          ITensor(ctmEnv.C_RU.at(ctmEnv.p_cluster->siteIds[0]).inds().front());
        spec = svd(ctmEnv.C_RU.at(ctmEnv.p_cluster->siteIds[0]), tL, sv, tR,
                   args_dbg_cornerSVD);
        PrintData(sv);

        tL =
          ITensor(ctmEnv.C_RD.at(ctmEnv.p_cluster->siteIds[0]).inds().front());
        spec = svd(ctmEnv.C_RD.at(ctmEnv.p_cluster->siteIds[0]), tL, sv, tR,
                   args_dbg_cornerSVD);
        PrintData(sv);

        tL =
          ITensor(ctmEnv.C_LD.at(ctmEnv.p_cluster->siteIds[0]).inds().front());
        spec = svd(ctmEnv.C_LD.at(ctmEnv.p_cluster->siteIds[0]), tL, sv, tR,
                   args_dbg_cornerSVD);
        PrintData(sv);
      }

      printBondSpectra();

      writeCluster(outClusterFile, *p_cls);
    }
  }

  // FULL UPDATE FINISHED - COMPUTING FINAL ENVIRONMENT
  std::cout << "FULL UPDATE DONE - COMPUTING FINAL ENVIRONMENT " << std::endl;
  diagData_ctm = computeEnvironment(arg_maxInitEnvIter, arg_reinitObsEnv);

  obs_metaInf = Args("lineNo", arg_fuIter + 1);
  ptr_model->computeAndWriteObservables(ev, out_file_energy, obs_metaInf);

  // Store final new cluster
  writeCluster(outClusterFile, *p_cls);

  for (auto const& log_entry : diag_log)
    std::cout << log_entry << std::endl;
}
