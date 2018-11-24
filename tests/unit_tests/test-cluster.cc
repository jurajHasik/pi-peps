#include <iostream>
#include <gtest/gtest.h>
#include "ctm-cluster.h"

using namespace itensor;

TEST(Cluster0, Default_cotr){

	auto s1 = Shift(1,1);
	auto s2 = Shift(2,3);
	auto s3 = Shift(-2,4);

	auto s4 = s1+s2;
	EXPECT_TRUE(s4 == Shift(3,4));

	auto s5 = s2+s3;
	EXPECT_TRUE(s5 == Shift(0,7));

	s2 += s1;
	EXPECT_TRUE(s2 == Shift(3,4));

	s3 -= s1;
	EXPECT_TRUE(s3 == Shift(-3,3));
}

TEST(Cluster1, Default_cotr){

	auto s1 = Shift(1,1);
	auto s2 = Shift(2,3);
	auto v1 = Vertex(-2,4);
	auto v2 = Vertex(3,2);

	auto v3 = v1+s1;
	EXPECT_TRUE(v3 == Vertex(-1,5));

	auto v4 = v1-s2;
	EXPECT_TRUE(v4 == Vertex(-4,1));

	v2 += s2;
	EXPECT_TRUE(v2 == Vertex(5,5));

	v2 -= s1;
	EXPECT_TRUE(v2 == Vertex(4,4));
}
