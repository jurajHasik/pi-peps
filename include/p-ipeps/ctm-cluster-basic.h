#ifndef __CLS_BASIC_H_
#define __CLS_BASIC_H_


#include "p-ipeps/config.h"
#include <iostream>
#include <vector>
#include <map>
#include "json.hpp"
DISABLE_WARNINGS
#include "itensor/all.h"
ENABLE_WARNINGS
#include "p-ipeps/ctm-cluster.h"

namespace itensor {

/* 
 *   0 1 2
 * 0 A A A
 * 1 A A A
 * 2 A A A
 *
 */
struct Cluster_1x1_A : Cluster {

	Cluster_1x1_A();

    Cluster_1x1_A(int ad, int pd);

    Cluster_1x1_A(std::string init_type, int ad, int pd);

	std::string virtual vertexToId(Vertex const& v) const override;

    static std::unique_ptr<Cluster> create(nlohmann::json const& json_cluster);
};


/* 
 *   0 1 2         0 1 
 * 0 A B A  -->  0 A B and shift rule 
 * 1 B A B
 * 2 A B A
 *
 */
struct Cluster_2x2_ABBA : Cluster {

	Cluster_2x2_ABBA();

    Cluster_2x2_ABBA(int ad, int pd);

    Cluster_2x2_ABBA(std::string init_type, int ad, int pd);

	std::string virtual vertexToId(Vertex const& v) const override;

    void init_RANDOM();

    void init_AFM();

    void init_ALIGNX();

    void init_ALIGNZ();

    void init_VBS();

    static std::unique_ptr<Cluster> create(nlohmann::json const& json_cluster);
};

/* 
 *   0 1 2 3
 * 0 A B A B
 * 1 C D C D
 * 2 A B A A
 *
 */
struct Cluster_2x2_ABCD : Cluster {

	Cluster_2x2_ABCD();

	Cluster_2x2_ABCD(int ad, int pd);

	Cluster_2x2_ABCD(std::string init_type, int ad, int pd);

	std::string virtual vertexToId(Vertex const& v) const override;

    void init_RANDOM();

    void init_RANDOM_BIPARTITE();

    void init_AFM();

	void init_ALIGNX();

	void init_ALIGNZ();

	void init_VBS();

    static std::unique_ptr<Cluster> create(nlohmann::json const& json_cluster);
};

} //

#endif
