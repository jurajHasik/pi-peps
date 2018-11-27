#ifndef __CLS_BASIC_H_
#define __CLS_BASIC_H_


#include <iostream>
#include <vector>
#include <map>
#include "json.hpp"
#include "itensor/all.h"
#include "ctm-cluster.h"

/* 
 *   0 1 2
 * 0 A A A
 * 1 A A A
 * 2 A A A
 *
 */
struct Cluster_1x1_A : Cluster {

	Cluster_1x1_A() : Cluster(1,1) {}

	std::string virtual vertexToId(Vertex const& v) const override { 
		auto elemV = Vertex(0,0);
		return vToId.at(elemV); 
    }

};


/* 
 *   0 1 2         0 1 
 * 0 A B A  -->  0 A B and shift rule 
 * 1 B A B
 * 2 A B A
 *
 */
struct Cluster_2x2_ABBA : Cluster {

	Cluster_2x2_ABBA() : Cluster(2,1) {}

	std::string virtual vertexToId(Vertex const& v) const override { 
		auto elemV = Vertex((v.r[0]+v.r[1]) % 2, 0);
		return vToId.at(elemV);
    }

};

/* 
 *   0 1 2 3
 * 0 A B A B
 * 1 C D C D
 * 2 A B A A
 *
 */
struct Cluster_2x2_ABCD : Cluster {

	Cluster_2x2_ABCD() : Cluster(2,2) {}

	std::string virtual vertexToId(Vertex const& v) const override { 
		auto elemV = Vertex(
			(v.r[0] + std::abs(v.r[0])*2)% 2, 
			(v.r[1] + std::abs(v.r[1])*2)% 2);
		return vToId.at(elemV);
    }

};

#endif