#include "pi-peps/config.h"
#include "pi-peps/models/j1j2-2x2-ABCD.h"

namespace itensor {

  // ----- Trotter gates (2site, ...) MPOs ------------------------------
  // TODO implement anisotropy del
  MPO_3site getMPO3s_J1J2(double tau, double J1, double J2, double del) {
    int physDim = 2;  // dimension of Hilbert space of spin s=1/2 DoF

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s3 = Index("S3", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);
    Index s3p = prime(s3);

    ITensor h3 = ITensor(s1, s2, s3, s1p, s2p, s3p);

    ITensor nnS1S2 =
      J1 * SU2_getSpinOp(SU2_S_Z, s1) * SU2_getSpinOp(SU2_S_Z, s2) +
      J1 * 0.5 *
        (SU2_getSpinOp(SU2_S_P, s1) * SU2_getSpinOp(SU2_S_M, s2) +
         SU2_getSpinOp(SU2_S_M, s1) * SU2_getSpinOp(SU2_S_P, s2));

    // Nearest-neighbour terms S_1.S_2 and S_2.S_3
    h3 += nnS1S2 * delta(s3, s3p);
    h3 += (nnS1S2 * delta(s1, s3) * delta(s1p, s3p)) * delta(s1, s1p);

    ITensor nnnS1S3 =
      J2 * (SU2_getSpinOp(SU2_S_Z, s1) * SU2_getSpinOp(SU2_S_Z, s3) +
            0.5 * (SU2_getSpinOp(SU2_S_P, s1) * SU2_getSpinOp(SU2_S_M, s3) +
                   SU2_getSpinOp(SU2_S_M, s1) * SU2_getSpinOp(SU2_S_P, s3)));

    h3 += nnnS1S3 * delta(s2, s2p);

    auto cmbI = combiner(s1, s2, s3);
    h3 = (cmbI * h3) * prime(cmbI);
    ITensor u3 = expHermitian(h3, {-tau, 0.0});
    u3 = (cmbI * u3) * prime(cmbI);

    return symmMPO3Sdecomp(u3, s1, s2, s3);
  }

  OpNS getOP4s_J1J2(double tau, double J1, double J2, double del) {
    int physDim = 2;  // dimension of Hilbert space of spin s=1/2 DoF

    Index s1 = Index("S1", physDim, PHYS);
    Index s2 = Index("S2", physDim, PHYS);
    Index s3 = Index("S3", physDim, PHYS);
    Index s4 = Index("S4", physDim, PHYS);
    Index s1p = prime(s1);
    Index s2p = prime(s2);
    Index s3p = prime(s3);
    Index s4p = prime(s4);

    ITensor h4 = ITensor(s1, s2, s3, s4, s1p, s2p, s3p, s4p);

    ITensor nnS1S2 =
      J1 * (SU2_getSpinOp(SU2_S_Z, s1) * SU2_getSpinOp(SU2_S_Z, s2) +
            0.5 * (SU2_getSpinOp(SU2_S_P, s1) * SU2_getSpinOp(SU2_S_M, s2) +
                   SU2_getSpinOp(SU2_S_M, s1) * SU2_getSpinOp(SU2_S_P, s2)));

    h4 += nnS1S2 * delta(s3, s3p) * delta(s4, s4p);  // S1S2id3id4
    h4 += (nnS1S2 * delta(s1, s3) * delta(s1p, s3p)) * delta(s1, s1p) *
          delta(s4, s4p);  // id1S2S3id4
    h4 += (nnS1S2 * delta(s2, s4) * delta(s2p, s4p)) * delta(s2, s2p) *
          delta(s3, s3p);  // S1id2id3S4
    h4 += (nnS1S2 * delta(s2, s4) * delta(s2p, s4p) * delta(s1, s3) *
           delta(s1p, s3p)) *
          delta(s1, s1p) * delta(s2, s2p);  // id1id2S3S4

    ITensor nnnS1S3 =
      J2 * (SU2_getSpinOp(SU2_S_Z, s1) * SU2_getSpinOp(SU2_S_Z, s3) +
            0.5 * (SU2_getSpinOp(SU2_S_P, s1) * SU2_getSpinOp(SU2_S_M, s3) +
                   SU2_getSpinOp(SU2_S_M, s1) * SU2_getSpinOp(SU2_S_P, s3)));

    h4 += nnnS1S3 * delta(s2, s2p) * delta(s4, s4p);
    h4 += (nnnS1S3 * delta(s1, s2) * delta(s1p, s2p) * delta(s3, s4) *
           delta(s3p, s4p)) *
          delta(s1, s1p) * delta(s3, s3p);

    auto cmbI = combiner(s1, s2, s3, s4);
    h4 = (cmbI * h4) * prime(cmbI);
    ITensor u4 = expHermitian(h4, {-tau, 0.0});
    u4 = (cmbI * u4) * prime(cmbI);

    auto op4s = OpNS(4);

    op4s.op = u4;
    op4s.pi = {s1, s2, s3, s4};

    return op4s;
  }
  // ----- END Trotter gates (3site, ...) MPOs --------------------------

  // ----- Definition of model base class and its particular instances --
  J1J2Model_2x2_ABCD::J1J2Model_2x2_ABCD(double arg_J1,
                                         double arg_J2,
                                         double arg_del)
    : J1(arg_J1), J2(arg_J2), del(arg_del) {}

  void J1J2Model_2x2_ABCD::setObservablesHeader(std::ofstream& output) {
    output << "STEP, "
           << "SS AB (0,0)(1,0), "
           << "SS AC (0,0)(0,1), "
           << "SS BD (1,0)(1,1), "
           << "SS CD (0,1)(1,1), "
           << "SS BA (1,0)(2,0), "
           << "SS CA (0,1)(0,2), "
           << "SS DB (1,1)(1,2), "
           << "SS DC (1,1)(2,1), "
           << "Avg SS_NN, "
           << "Avg SS_NNN, "
           << "Avg mag=|S|, "
           << "Energy" << std::endl;
  }

  void J1J2Model_2x2_ABCD::computeAndWriteObservables(EVBuilder const& ev,
                                                      std::ofstream& output,
                                                      Args& metaInf) {
    auto lineNo = metaInf.getInt("lineNo", 0);

    std::vector<double> evNN;
    std::vector<double> evNNN;
    std::vector<double> ev_sA(3, 0.0);
    std::vector<double> ev_sB(3, 0.0);
    std::vector<double> ev_sC(3, 0.0);
    std::vector<double> ev_sD(3, 0.0);

    bool compute_SS_NNN = (std::abs(J2) > 1.0e-15);

    evNN.push_back(
      ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(0, 0), Vertex(1, 0)));  // AB
    evNN.push_back(
      ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(0, 0), Vertex(0, 1)));  // AC
    evNN.push_back(
      ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(1, 0), Vertex(1, 1)));  // BD
    evNN.push_back(
      ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(0, 1), Vertex(1, 1)));  // CD

    evNN.push_back(
      ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(1, 0), Vertex(2, 0)));  // BA
    evNN.push_back(
      ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(0, 1), Vertex(0, 2)));  // CA
    evNN.push_back(
      ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(1, 1), Vertex(1, 2)));  // DB
    evNN.push_back(
      ev.eval2Smpo(EVBuilder::OP2S_SS, Vertex(1, 1), Vertex(2, 1)));  // DC

    // compute energies NNN links
    if (compute_SS_NNN) {
      evNNN.push_back(ev.eval2x2Diag11(EVBuilder::OP2S_SS, Vertex(0, 0)));
      evNNN.push_back(ev.eval2x2Diag11(EVBuilder::OP2S_SS, Vertex(1, 1)));
      evNNN.push_back(ev.eval2x2Diag11(EVBuilder::OP2S_SS, Vertex(1, 0)));
      evNNN.push_back(ev.eval2x2Diag11(EVBuilder::OP2S_SS, Vertex(2, 1)));

      evNNN.push_back(ev.eval2x2DiagN11(EVBuilder::OP2S_SS, Vertex(0, 0)));
      evNNN.push_back(ev.eval2x2DiagN11(EVBuilder::OP2S_SS, Vertex(1, 1)));
      evNNN.push_back(ev.eval2x2DiagN11(EVBuilder::OP2S_SS, Vertex(1, 0)));
      evNNN.push_back(ev.eval2x2DiagN11(EVBuilder::OP2S_SS, Vertex(0, 1)));
    }

    ev_sA[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(0, 0));
    ev_sA[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(0, 0));
    ev_sA[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(0, 0));

    ev_sB[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(1, 0));
    ev_sB[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(1, 0));
    ev_sB[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(1, 0));

    ev_sC[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(0, 1));
    ev_sC[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(0, 1));
    ev_sC[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(0, 1));

    ev_sD[0] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_Z, Vertex(1, 1));
    ev_sD[1] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_P, Vertex(1, 1));
    ev_sD[2] = ev.eV_1sO_1sENV(EVBuilder::MPO_S_M, Vertex(1, 1));

    output << lineNo << " ";
    // write individual NN SS terms and average over all non-eq links
    double avgSS_NN = 0.;
    for (unsigned int j = 0; j < evNN.size(); j++) {
      output << " " << evNN[j];
      avgSS_NN += evNN[j];
    }
    avgSS_NN = avgSS_NN / 8.0;
    output << " " << avgSS_NN;

    // write average NNN SS term over all non-eq NNN
    double avgSS_NNN = 0.;
    if (compute_SS_NNN) {
      for (unsigned int j = 0; j < evNNN.size(); j++)
        avgSS_NNN += evNNN[j];
      avgSS_NNN = avgSS_NNN / 8.0;
    }
    output << " " << avgSS_NNN;

    // write magnetization
    double evMag_avg = 0.;
    evMag_avg = 0.25 * (std::sqrt(ev_sA[0] * ev_sA[0] + ev_sA[1] * ev_sA[1]) +
                        std::sqrt(ev_sB[0] * ev_sB[0] + ev_sB[1] * ev_sB[1]) +
                        std::sqrt(ev_sC[0] * ev_sC[0] + ev_sC[1] * ev_sC[1]) +
                        std::sqrt(ev_sD[0] * ev_sD[0] + ev_sD[1] * ev_sD[1]));
    output << " " << evMag_avg;

    // write Energy
    double energy = 2.0 * avgSS_NN * J1 + 2.0 * avgSS_NNN * J2;
    output << " " << energy;

    // return energy in metaInf
    metaInf.add("energy", energy);

    output << std::endl;
  }

  std::unique_ptr<Model> J1J2Model_2x2_ABCD::create(
    nlohmann::json& json_model) {
    double arg_J1 = json_model["J1"].get<double>();
    double arg_J2 = json_model["J2"].get<double>();
    double arg_del = 0.0;

    return std::unique_ptr<Model>(
      new J1J2Model_2x2_ABCD(arg_J1, arg_J2, arg_del));
  }

  std::unique_ptr<Engine> J1J2Model_2x2_ABCD::buildEngine(
    nlohmann::json& json_model) {
    double arg_J1 = json_model["J1"].get<double>();
    double arg_J2 = json_model["J2"].get<double>();
    double arg_del = 0.0;
    double arg_tau = json_model["tau"].get<double>();

    // gate sequence
    std::string arg_fuGateSeq = json_model["fuGateSeq"].get<std::string>();

    // symmetrize Trotter Sequence
    bool arg_symmTrotter = json_model.value("symmTrotter", false);

    if (arg_fuGateSeq == "SYM3") {
      TrotterEngine<MPO_3site>* pe = new TrotterEngine<MPO_3site>();

      pe->td.gateMPO.push_back(
        getMPO3s_J1J2(arg_tau, arg_J1 / 4.0, arg_J2 / 2.0, arg_del));

      pe->td.tgates = {
        TrotterGate<MPO_3site>(Vertex(0, 0), {Shift(1, 0), Shift(0, 1)},
                               &pe->td.gateMPO[0]),
        TrotterGate<MPO_3site>(Vertex(1, 0), {Shift(1, 0), Shift(0, 1)},
                               &pe->td.gateMPO[0]),
        TrotterGate<MPO_3site>(Vertex(0, 1), {Shift(1, 0), Shift(0, 1)},
                               &pe->td.gateMPO[0]),
        TrotterGate<MPO_3site>(Vertex(1, 1), {Shift(1, 0), Shift(0, 1)},
                               &pe->td.gateMPO[0]),
        TrotterGate<MPO_3site>(Vertex(0, 0), {Shift(-1, 0), Shift(0, -1)},
                               &pe->td.gateMPO[0]),
        TrotterGate<MPO_3site>(Vertex(1, 0), {Shift(-1, 0), Shift(0, -1)},
                               &pe->td.gateMPO[0]),
        TrotterGate<MPO_3site>(Vertex(0, 1), {Shift(-1, 0), Shift(0, -1)},
                               &pe->td.gateMPO[0]),
        TrotterGate<MPO_3site>(Vertex(1, 1), {Shift(-1, 0), Shift(0, -1)},
                               &pe->td.gateMPO[0]),
        TrotterGate<MPO_3site>(Vertex(0, 0), {Shift(1, 0), Shift(0, -1)},
                               &pe->td.gateMPO[0]),
        TrotterGate<MPO_3site>(Vertex(1, 0), {Shift(1, 0), Shift(0, -1)},
                               &pe->td.gateMPO[0]),
        TrotterGate<MPO_3site>(Vertex(0, 1), {Shift(1, 0), Shift(0, -1)},
                               &pe->td.gateMPO[0]),
        TrotterGate<MPO_3site>(Vertex(1, 1), {Shift(1, 0), Shift(0, -1)},
                               &pe->td.gateMPO[0]),
        TrotterGate<MPO_3site>(Vertex(0, 0), {Shift(-1, 0), Shift(0, 1)},
                               &pe->td.gateMPO[0]),
        TrotterGate<MPO_3site>(Vertex(1, 0), {Shift(-1, 0), Shift(0, 1)},
                               &pe->td.gateMPO[0]),
        TrotterGate<MPO_3site>(Vertex(0, 1), {Shift(-1, 0), Shift(0, 1)},
                               &pe->td.gateMPO[0]),
        TrotterGate<MPO_3site>(Vertex(1, 1), {Shift(-1, 0), Shift(0, 1)},
                               &pe->td.gateMPO[0])};

      std::cout << "J1J2Model_2x2_ABCD SYM3 ENGINE constructed" << std::endl;
      if (arg_symmTrotter)
        pe->td.symmetrize();
      return std::unique_ptr<Engine>(pe);
    } else if (arg_fuGateSeq == "4SITE") {
      TrotterEngine<OpNS>* pe = new TrotterEngine<OpNS>();

      pe->td.gateMPO.push_back(
        getOP4s_J1J2(arg_tau, arg_J1 / 2.0, arg_J2, arg_del / 2.0));

      pe->td.tgates = {
        TrotterGate<OpNS>(Vertex(0, 0), {Shift(1, 0), Shift(0, 1)},
                          &pe->td.gateMPO[0]),
        TrotterGate<OpNS>(Vertex(1, 0), {Shift(1, 0), Shift(0, 1)},
                          &pe->td.gateMPO[0]),
        TrotterGate<OpNS>(Vertex(0, 1), {Shift(1, 0), Shift(0, 1)},
                          &pe->td.gateMPO[0]),
        TrotterGate<OpNS>(Vertex(1, 1), {Shift(1, 0), Shift(0, 1)},
                          &pe->td.gateMPO[0])};

      std::cout << "J1J2Model_2x2_ABCD 4SITE ENGINE constructed" << std::endl;
      if (arg_symmTrotter)
        pe->td.symmetrize();
      return std::unique_ptr<Engine>(pe);
    } else {
      std::cout << "Unsupported gate sequence: " << arg_fuGateSeq << std::endl;
      exit(EXIT_FAILURE);
    }

    return nullptr;
  }
  // ----- END Definition of model class --------------------------------

}  // namespace itensor
