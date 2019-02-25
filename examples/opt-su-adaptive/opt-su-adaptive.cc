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
#include "pi-peps/cluster-factory.h"
#include "pi-peps/ctm-cluster-env_v2.h"
#include "pi-peps/ctm-cluster-io.h"
#include "pi-peps/ctm-cluster.h"
#include "pi-peps/engine-factory.h"
#include "pi-peps/model-factory.h"
#include "pi-peps/mpo.h"
#include "pi-peps/svdsolver-factory.h"

using namespace itensor;

int main(int argc, char* argv[]) {
  // ***** INITIALIZE SIMPLE UPDATE ALGORITHM *******************************
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

  // TODO if no file input cluster is supplied, construct from one of the
  // templates
  std::string inClusterFile = json_cluster.value("inClusterFile", "DEFAULT");

  int physDim = json_cluster["physDim"].get<int>();
  int auxBondDim = json_cluster["auxBondDim"].get<int>();
  double initStateNoise = jsonCls.value("initStateNoise", 1.0e-16);

  // read cluster outfile
  std::string outClusterFile(jsonCls["outClusterFile"].get<std::string>());

  // read Hamiltonian and Trotter decomposition
  auto json_model_params(jsonCls["model"]);
  json_model_params["physDim"] = physDim;
  bool symmTrotter = json_model_params.value("symmTrotter", true);
  bool randomizeSeq = json_model_params.value("randomizeSeq", false);

  std::string suWeightsInit(jsonCls.value("suWeightsInit", "DELTA"));

  int arg_suIter = jsonCls["suIter"].get<int>();
  int arg_obsFreq = jsonCls["obsFreq"].get<int>();
  bool arg_decreaseTimestep = jsonCls.value("decreaseTimestep", true);
  double arg_dtFraction = jsonCls.value("dtFraction", 0.5);
  double arg_minTimestep = jsonCls.value("minTimestep", 1.0e-6);
  bool arg_suDbg = jsonCls["suDbg"].get<bool>();
  int arg_suDbgLevel = jsonCls["suDbgLevel"].get<int>();

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
  bool arg_reinitEnv = json_ctmrg_params["reinitEnv"].get<bool>();
  bool arg_envDbg = json_ctmrg_params["dbg"].get<bool>();
  int arg_envDbgLvl = json_ctmrg_params["dbgLvl"].get<int>();
  // end reading CTMRG parameters

  // ***** INITIALIZE SIMPLE UPDATE ALGORITHM DONE **************************

  // ***** INITIALIZE CLUSTER ***********************************************
  std::unique_ptr<Cluster> p_cls;

  // set auxiliary dimension to the desired one
  // cls.auxBondDim = auxBondDim;

  // choose initial wavefunction
  if (initBy == "FILE" and inClusterFile != "DEFAULT") {
    std::ifstream infile(inClusterFile, std::ios::in);
    nlohmann::json json_cluster_file = nlohmann::json::parse(infile);

    // preprocess parameters of input cluster
    json_cluster_file["initBy"] = "FILE";
    json_cluster_file["auxBondDim"] = auxBondDim;
    for (auto& site : json_cluster_file["sites"]) {
      site["auxDim"] = auxBondDim;
    }

    p_cls = p_readCluster(json_cluster_file);
    // initClusterSites(cls);
    // initClusterWeights(cls);
    setWeights(*p_cls, suWeightsInit);
    // setOnSiteTensorsFromFile(cls, inClusterFile);
  } else if (initBy == "FILE" and inClusterFile == "DEFAULT") {
    throw std::runtime_error("No cluster input file  given for inClusterFile");
  } else {
    // one of the defined initialization procedures
    ClusterFactory cf = ClusterFactory();
    p_cls = cf.create(json_cluster);
    initClusterWeights(*p_cls);
    setWeights(*p_cls, suWeightsInit);
  }

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

  std::cout << *p_cls;

  for (int y = 0; y < 4; y++) {
    for (int x = 0; x < 4; x++) {
      std::cout << "[" << x << "," << y
                << "]=" << p_cls->vertexToId(Vertex(x, y)) << " ";
    }
    std::cout << std::endl;
  }
  // ***** INITIALIZE CLUSTER DONE ******************************************

  // ***** INITIALIZE MODEL *************************************************
  // DEFINE MODEL AND GATE SEQUENCE
  ModelFactory mf = ModelFactory();
  EngineFactory ef = EngineFactory();
  auto ptr_model = mf.create(json_model_params);
  auto ptr_engine = ef.build(json_model_params);

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
  std::vector<Args> diagData_fu;
  Args diag_fu;

  // hold energies
  std::vector<double> e_curr(4, 0.0), e_prev(4, 0.0);

  std::vector<double> accT(12, 0.0);  // holds timings for CTM moves
  time_point t_begin_int, t_end_int;

  // ***** INITIALIZE ENVIRONMENT *******************************************

  // ***** Select SVD solver to use *****************************************
  SvdSolverFactory sf = SvdSolverFactory();
  auto pSvdSolver = sf.create(env_SVD_METHOD);

  CtmEnv ctmEnv("default", auxEnvDim, *p_cls, *pSvdSolver,
                {"isoPseudoInvCutoff", arg_isoPseudoInvCutoff, "SVD_METHOD",
                 env_SVD_METHOD, "rsvd_power", rsvd_power, "rsvd_reortho",
                 rsvd_reortho, "rsvd_oversampling", rsvd_oversampling, "dbg",
                 arg_envDbg, "dbgLevel", arg_envDbgLvl});
  ctmEnv.init(arg_initEnvType, false, arg_envDbg);

  // INITIALIZE EXPECTATION VALUE BUILDER
  EVBuilder ev("default", *p_cls, ctmEnv);
  std::cout << ev;

  for (int y = 0; y < 4; y++) {
    for (int x = 0; x < 4; x++) {
      std::cout << "[" << x << "," << y
                << "]=" << ev.p_cluster->vertexToId(Vertex(x, y)) << " ";
    }
    std::cout << std::endl;
  }

  auto printBondSpectra_weights = [&p_cls] {
    auto printS = [](Real r) { std::cout << std::scientific << r << " "; };

    std::cout << "BOND SPECTRA(WEIGHTS) - START" << std::endl;
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
          p_cls->weights.at(lw.wId).visit(printS);
          std::cout << std::endl;

          lwIds.push_back(lw.wId);
        }
    std::cout << "BOND SPECTRA(WEIGHTS) - END" << std::endl;
  };
  auto printBondSpectra_sites = [&p_cls] {
    auto printS = [](Real r) { std::cout << std::scientific << r << " "; };

    std::cout << "BOND SPECTRA(SITES) - START" << std::endl;
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
          // S *= 1.0/S.real(S.inds()[0](1),S.inds()[1](1));
          S.visit(printS);
          std::cout << std::endl;

          lwIds.push_back(lw.wId);
        }
    std::cout << "BOND SPECTRA(SITES) - END" << std::endl;
  };

  auto computeEnvironment = [&ctmEnv, &ev, &iso_type, &arg_envEps,
                             &arg_initEnvType, &arg_envDbg, &arg_envDbgLvl,
                             &get_s](int maxIter, bool reinitEnv) {
    time_point t_begin_int, t_end_int;
    std::vector<double> accT(12, 0.0);
    std::vector<double> e_curr(4, 0.0), e_prev(4, 0.0);
    bool expValEnvConv = false;

    Args diagData_ctm = Args::global();

    if (reinitEnv)
      ctmEnv.init(arg_initEnvType, false, arg_envDbg);

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

  // COMPUTE INITIAL OBSERVABLES
  Args diagData_ctm;

  p_cls->absorbWeightsToSites();
  diagData_ctm = computeEnvironment(arg_maxEnvIter, arg_reinitEnv);

  out_file_diag << 0 << " " << diagData_ctm.getInt("ctmI", -1) << " "
                << diagData_ctm.getReal("max_tailCornerSV", -1.0) << " "
                << diagData_ctm.getReal("maxBoundaryVariance", -1.0)
                << std::endl;
  // ***** INITIAL CTMRG DONE **************************************

  // Compute initial properties
  ptr_model->setObservablesHeader(out_file_energy);
  auto metaInf = Args("lineNo", 0);
  t_begin_int = std::chrono::steady_clock::now();
  ptr_model->computeAndWriteObservables(ev, out_file_energy, metaInf);
  t_end_int = std::chrono::steady_clock::now();
  std::cout << "Observables computed in T: " << get_s(t_begin_int, t_end_int)
            << " [sec] " << std::endl;
  printBondSpectra_weights();
  printBondSpectra_sites();
  p_cls->absorbWeightsToLinks();
  // ***** INITIALIZE ENVIRONMENT DONE **************************************

  // ########################################################################
  // # SETUP OPTIMIZATION LOOP                                              #
  // ########################################################################

  // Diagnostic data
  std::vector<std::string> diag_log;

  double best_energy = metaInf.getReal("energy");
  auto past_tensors = p_cls->sites;
  auto past_weights = p_cls->weights;

  Args suArgs = {"suDbg", arg_suDbg, "suDbgLevel", arg_suDbgLevel};

  // ENTER OPTIMIZATION LOOP
  for (int suI = 1; suI <= arg_suIter; suI++) {
    std::cout << "Simple Update - STEP " << suI << std::endl;

    // PERFORM SIMPLE UPDATE
    diag_fu = ptr_engine->performSimpleUpdate(*p_cls, suArgs);

    diagData_fu.push_back(diag_fu);

    if (suI % arg_obsFreq == 0) {
      printBondSpectra_weights();

      p_cls->absorbWeightsToSites();
      printBondSpectra_sites();

      // ENTER ENVIRONMENT LOOP
      diagData_ctm = computeEnvironment(arg_maxEnvIter, arg_reinitEnv);
      // ***** CTMRG DONE **************************************

      out_file_diag << suI << " " << diagData_ctm.getInt("ctmI", -1) << " "
                    << diagData_ctm.getReal("max_tailCornerSV", -1.0) << " "
                    << diagData_ctm.getReal("maxBoundaryVariance", -1.0)
                    << std::endl;

      t_begin_int = std::chrono::steady_clock::now();
      auto metaInf = Args("lineNo", suI);
      ptr_model->computeAndWriteObservables(ev, out_file_energy, metaInf);
      t_end_int = std::chrono::steady_clock::now();
      std::cout << "Observables computed in T: "
                << get_s(t_begin_int, t_end_int) << " [sec] " << std::endl;

      p_cls->absorbWeightsToLinks();

      // check energy, preserve the best_energy state obtained so far
      auto current_energy = metaInf.getReal("energy");
      if (best_energy > current_energy) {
        best_energy = current_energy;
        p_cls->metaInfo = "BestEnergy(FUStep=" + std::to_string(suI) + ")";
        p_cls->simParam = jsonCls;
        p_cls->absorbWeightsToSites();
        writeCluster(outClusterFile, *p_cls);
        p_cls->absorbWeightsToLinks();
        past_tensors = p_cls->sites;
        past_weights = p_cls->weights;
      }
      // check if current energy > previous energy
      if ((current_energy > best_energy) && arg_decreaseTimestep) {
        std::ostringstream oss;
        oss << std::scientific;
        oss << suI << ": ENERGY INCREASED: E(i)-E(i-1)="
            << current_energy - best_energy;
        // TODO revert to previous tensors and recompute the environment
        oss << " Reverting to previous tensors";
        p_cls->sites = past_tensors;
        p_cls->weights = past_weights;
        // decrease time-step
        auto current_dt = json_model_params["tau"].get<double>();
        json_model_params["tau"] = current_dt * arg_dtFraction;
        jsonCls["model"] = json_model_params;
        oss << " Timestep decreased: " << current_dt << " -> "
            << current_dt * arg_dtFraction;
        // regenerate model with new lower timestep
        ptr_engine = ef.build(json_model_params);
        // update simulation parameters on cluster
        p_cls->simParam = jsonCls;
        diag_log.push_back(oss.str());
        std::cout << oss.str() << std::endl;
        if (current_dt < arg_minTimestep) {
          std::cout << "Timstep too small. Stopping simulation" << std::endl;
          break;
        }
      }

      // TODO current energy is higher than energy at previous step STOP
      // if (arg_stopEnergyInc && *energyDiff*) {
      //     break;
      //     std::cout<< "CURRENT ENERGY > PREVIOUS ENERGY:
      //     STOPPING"<<std::endl;
      // }
    }
  }

  // SIMPLE UPDATE FINISHED
  std::cout << "SIMPLE UPDATE DONE" << std::endl;
  for (auto const& log_entry : diag_log)
    std::cout << log_entry << std::endl;
}
