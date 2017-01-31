#include "su2.h"

using namespace std;

double SU2_getCG(int j1, int j2, int j, int m1, int m2, int m) {
    // (!) Use Dynkin notation to pass desired irreps
    
    double getCG = 0.0;
    if (m == m1+m2) {
        double pref = sqrt((j+1.0)*Factorial((j+j1-j2)/2)*
    		Factorial((j-j1+j2)/2)*Factorial((j1+j2-j)/2)
    		/Factorial((j1+j2+j)/2+1))*sqrt(Factorial((j+m)/2)
    		*Factorial((j-m)/2)*Factorial((j1-m1)/2)*Factorial((j1+m1)/2)
    		*Factorial((j2-m2)/2)*Factorial((j2+m2)/2));
    //write(*,'("<m1=",I2," m2=",I2,"|m=",I2,"> pref = ",1f10.5)') &
    //      & m1, m2, m, pref
        int min_k = min((j1+j2)/2,j2);
        double sum_k = 0.0;
        for(int k=0; k <= min_k+1; k++) {
            if ( ((j1+j2-j)/2-k >= 0) && ((j1-m1)/2-k >= 0) &&
                 ((j2+m2)/2-k >= 0) && ((j-j2+m1)/2+k >= 0) &&
                 ((j-j1-m2)/2+k >= 0) ) {
                    sum_k += pow(-1,k)/( Factorial(k)
                        *Factorial((j1+j2-j)/2-k)
                        *Factorial((j1-m1)/2-k)
                        *Factorial((j2+m2)/2-k)
                        *Factorial((j-j2+m1)/2+k)
                        *Factorial((j-j1-m2)/2+k) );
            }
        }
        getCG = pref*sum_k;
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