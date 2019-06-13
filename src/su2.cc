#include "pi-peps/config.h"
#include "pi-peps/su2.h"

using namespace std;

itensor::ITensor SU2_getSpinOp(SU2O su2o, itensor::Index const& s, bool DBG) {
  auto s1 = prime(s);
  int dimS = s.m();

  // Construct MPO
  auto Op = itensor::ITensor(s, s1);

  switch (su2o) {
    case SU2_Id: {
      if (DBG)
        std::cout << ">>>>> Constructing 1sO: Id <<<<<" << std::endl;
      for (int i = 1; i <= dimS; i++)
        Op.set(s1(i), s(i), 1.0);
      break;
    }
    case SU2_S_Z: {
      if (DBG)
        std::cout << ">>>>> Constructing 1sO: Sz <<<<<" << std::endl;
      for (int i = 1; i <= dimS; i++)
        Op.set(s1(i), s(i), -0.5 * (-(dimS - 1) + (i - 1) * 2));
      break;
    }
    case SU2_S_Z2: {
      if (DBG)
        std::cout << ">>>>> Constructing 1sO: Sz^2 <<<<<" << std::endl;
      for (int i = 1; i <= dimS; i++)
        Op.set(s1(i), s(i), pow(0.5 * (-(dimS - 1) + (i - 1) * 2), 2.0));
      break;
    }
    /*
     * The s^+ operator maps states with s^z = x to states with
     * s^z = x+1 . Therefore as a matrix it must act as follows
     * on vector of basis elements of spin S representation (in
     * this particular order) |S M>
     *
     *     |-S  >    C_+|-S+1>           0 1 0 0 ... 0
     * s^+ |-S+1>  = C_+|-S+2>  => S^+ = 0 0 1 0 ... 0 x C_+
     *      ...         ...              ...
     *     | S-1>    C_+| S  >           0    ...  0 1
     *     | S  >     0                  0    ...  0 0
     *
     * where C_+ = sqrt(S(S+1)-M(M+1))
     *
     */
    case SU2_S_P: {
      if (DBG)
        std::cout << ">>>>> Constructing 1sO: S^+ <<<<<" << std::endl;
      for (int i = 1; i <= dimS - 1; i++)
        Op.set(s1(i), s(i + 1),
               pow(0.5 * (dimS - 1) * (0.5 * (dimS - 1) + 1) -
                     (-0.5 * (dimS - 1) + (i - 1)) *
                       (-0.5 * (dimS - 1) + (i - 1) + 1),
                   0.5));
      break;
    }
    /*
     * The s^- operator maps states with s^z = x to states with
     * s^z = x-1 . Therefore as a matrix it must act as follows
     * on vector of basis elements of spin S representation (in
     * this particular order) |S M>
     *
     *     |-S  >     0                  0 0 0 0 ... 0
     * s^- |-S+1>  = C_-|-S  >  => S^- = 1 0 0 0 ... 0 x C_-
     *      ...         ...              ...
     *     | S-1>    C_-| S-2>           0   ... 1 0 0
     *     | S  >    C_-| S-1>           0   ... 0 1 0
     *
     * where C_- = sqrt(S(S+1)-M(M-1))
     *
     */
    case SU2_S_M: {
      if (DBG)
        std::cout << ">>>>> Constructing 1sO: S^- <<<<<" << std::endl;
      for (int i = 2; i <= dimS; i++)
        Op.set(s1(i), s(i - 1),
               pow(0.5 * (dimS - 1) * (0.5 * (dimS - 1) + 1) -
                     (-0.5 * (dimS - 1) + (i - 1)) *
                       (-0.5 * (dimS - 1) + (i - 1) - 1),
                   0.5));
      break;
    }
    default: {
      std::cout << "Invalid MPO selection" << std::endl;
      exit(EXIT_FAILURE);
      break;
    }
  }

  if (DBG)
    PrintData(Op);

  return Op;
}

itensor::ITensor SU2_getRotOp(itensor::Index const& s) {
  auto s1 = prime(s);
  int dimS = s.m();

  // Operator corresponds to "odd" site of bipartite AKLT
  // state - perform rotation on physical indices
  /*
   * I(s)'--|Op|--I(s) =>
   *
   * I(s)'''--|R1|--I(s)'--|Op|--I(s)--|R2|--I(s)''
   *
   * where Rot is a real symmetric rotation matrix, thus R1 = R2
   * defined below. Then one has to set indices of rotated
   * Op to proper prime level
   *
   */
  auto R1 = itensor::ITensor(s, s1);
  for (int i = 1; i <= dimS; i++) {
    R1.set(s(i), s1(dimS + 1 - i), pow(-1, i - 1));
  }

  return R1;
}

itensor::ITensor SU2_applyRot(itensor::Index const& s,
                              itensor::ITensor const& op) {
  auto s1 = prime(s);

  auto res = prime(op);
  auto R1 = SU2_getRotOp(s);
  auto R2 = (R1 * delta(s, prime(s, 3))) * delta(s1, prime(s, 2));
  res = (R1 * res * R2) * delta(prime(s, 3), s1);

  return res;
}

double SU2_getCG(int j1, int j2, int j, int m1, int m2, int m) {
  // (!) Use Dynkin notation to pass desired irreps

  double getCG = 0.0;
  if (m == m1 + m2) {
    double pref =
      sqrt((j + 1.0) * Factorial((j + j1 - j2) / 2) *
           Factorial((j - j1 + j2) / 2) * Factorial((j1 + j2 - j) / 2) /
           Factorial((j1 + j2 + j) / 2 + 1)) *
      sqrt(Factorial((j + m) / 2) * Factorial((j - m) / 2) *
           Factorial((j1 - m1) / 2) * Factorial((j1 + m1) / 2) *
           Factorial((j2 - m2) / 2) * Factorial((j2 + m2) / 2));
    // write(*,'("<m1=",I2," m2=",I2,"|m=",I2,"> pref = ",1f10.5)') &
    //      & m1, m2, m, pref
    int min_k = min((j1 + j2) / 2, j2);
    double sum_k = 0.0;
    for (int k = 0; k <= min_k + 1; k++) {
      if (((j1 + j2 - j) / 2 - k >= 0) && ((j1 - m1) / 2 - k >= 0) &&
          ((j2 + m2) / 2 - k >= 0) && ((j - j2 + m1) / 2 + k >= 0) &&
          ((j - j1 - m2) / 2 + k >= 0)) {
        sum_k +=
          pow(-1, k) /
          (Factorial(k) * Factorial((j1 + j2 - j) / 2 - k) *
           Factorial((j1 - m1) / 2 - k) * Factorial((j2 + m2) / 2 - k) *
           Factorial((j - j2 + m1) / 2 + k) * Factorial((j - j1 - m2) / 2 + k));
      }
    }
    getCG = pref * sum_k;
  }
  return getCG;
}

int Factorial(int x) {
  if (x == 0) {
    return 1;
  } else if (x == 1) {
    return 1;
  }
  return x * Factorial(x - 1);
}
