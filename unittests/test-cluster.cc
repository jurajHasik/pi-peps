#include <iostream>
#include "ctm-cluster.h"

using namespace itensor;

int main( int argc, char *argv[] ) {

	auto s1 = Shift(1,1);
	auto s2 = Shift(2,3);
	auto s3 = Shift(-2,4);

	auto s4 = s1+s2;
	if (s4 != Shift(3,4)) std::cout<<"ERROR 1"<<std::endl;
	
	auto s5 = s2+s3;
	if (s5 != Shift(0,7)) std::cout<<"ERROR 2"<<std::endl;

	s2 += s1;
	if (s2 != Shift(3,4)) std::cout<<"ERROR 3"<<std::endl;

	s3 -= s1;
	if (s5 != Shift(-3,3)) std::cout<<"ERROR 4"<<std::endl;
}
