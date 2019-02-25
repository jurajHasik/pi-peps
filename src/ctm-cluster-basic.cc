#include "p-ipeps/ctm-cluster-basic.h"

namespace itensor {

/* 
 *   0 1 2
 * 0 A A A
 * 1 A A A
 * 2 A A A
 *
 */
Cluster_1x1_A::Cluster_1x1_A() : Cluster(1,1) { cluster_type = "1X1_A"; }

Cluster_1x1_A::Cluster_1x1_A(int ad, int pd) : Cluster(1,1,ad,pd) {
    cluster_type = "1X1_A";
    siteIds = { "A" };
    SI = { {"A",0} };

    cToS  = {
        {std::make_pair(0,0),"A"}
    };
    vToId = { {{0,0},"A"} };
    idToV = { {"A",{0,0}} };

    auto aIA = Index("A", auxBondDim, AUXLINK);
    auto pIA = Index("A", physDim, PHYS);

    aux  = {aIA};
    phys = {pIA};
    maux  = {{"A",aIA}};
    mphys = {{"A",pIA}};
    for (const auto& id : siteIds) { 
        caux[id] = std::vector<Index>(4);
        for(int i=0; i<caux[id].size(); i++) caux[id][i] = prime(maux.at(id),i);
    }

    auto A = ITensor(aIA, prime(aIA,1), prime(aIA,2), prime(aIA,3), pIA);

    sites = {{"A", A}};

    // Define siteToWeights
    siteToWeights["A"] = {
        {{"A","A"},{2,0},"L1"},
        {{"A","A"},{3,1},"L2"}
    };
}

// TODO initialization
Cluster_1x1_A::Cluster_1x1_A(std::string init_type, int ad, int pd) 
    : Cluster_1x1_A(ad,pd) {
    
    // if(init_type == "RANDOM") {
    //     init_RANDOM();
    // } else if (init_type == "ALIGNX") {
    //     init_ALIGNX();
    // } else if (init_type == "ZPRST") {
    //     init_ALIGNZ();
    // } else {
    //     std::cout <<"Unsupported cluster initialization: "<< init_type << std::endl;
    // }
}

std::string Cluster_1x1_A::vertexToId(Vertex const& v) const { 
	auto elemV = Vertex(0,0);
	return vToId.at(elemV); 
}

std::unique_ptr<Cluster> Cluster_1x1_A::create(nlohmann::json const& json_cluster) {
    
    int pd = json_cluster["physDim"].get<int>();
    int ad = json_cluster["auxBondDim"].get<int>();
    std::string init_type = json_cluster["initBy"].get<std::string>();

    return std::unique_ptr<Cluster>(new Cluster_1x1_A(init_type,ad,pd));
}



/* 
 *   0 1 2         0 1 
 * 0 A B A  -->  0 A B and shift rule 
 * 1 B A B
 * 2 A B A
 *
 */
Cluster_2x2_ABBA::Cluster_2x2_ABBA() : Cluster(2,2) { cluster_type = "2X2_ABBA"; }

Cluster_2x2_ABBA::Cluster_2x2_ABBA(int ad, int pd) 
    : Cluster(2,2,ad,pd) { cluster_type = "2X2_ABBA"; }

Cluster_2x2_ABBA::Cluster_2x2_ABBA(std::string init_type, int ad, int pd) 
    : Cluster_2x2_ABBA(ad,pd) {
    
    if (init_type == "FILE") {
        // pass, the elements are initialized outside
        return;
    }

    // Assume initialization of elements by one of the predefined functions
    siteIds = { "A", "B" };
    SI = { {"A",0}, {"B",1} };

    cToS  = {
        {std::make_pair(0,0),"A"},
        {std::make_pair(1,0),"B"},
        {std::make_pair(0,1),"B"},
        {std::make_pair(1,1),"A"}
    };
    vToId = { {{0,0},"A"}, {{1,0},"B"} };
    idToV = { {"A",{0,0}}, {"B",{1,0}} };

    auto aIA = Index("A", auxBondDim, AUXLINK);
    auto aIB = Index("B", auxBondDim, AUXLINK);
    auto pIA = Index("A", physDim, PHYS);
    auto pIB = Index("B", physDim, PHYS);

    aux  = {aIA, aIB};
    phys = {pIA, pIB};
    maux  = {{"A",aIA},{"B",aIB}};
    mphys = {{"A",pIA},{"B",pIB}};
    for (const auto& id : siteIds) { 
        caux[id] = std::vector<Index>(4);
        for(int i=0; i<caux[id].size(); i++) caux[id][i] = prime(maux.at(id),i);
    }

    auto A = ITensor(aIA, prime(aIA,1), prime(aIA,2), prime(aIA,3), pIA);
    auto B = ITensor(aIB, prime(aIB,1), prime(aIB,2), prime(aIB,3), pIB);

    sites = {{"A", A}, {"B", B}};

    // Define siteToWeights
    siteToWeights["A"] = {
        {{"A","B"},{2,0},"L1"},
        {{"A","B"},{0,2},"L2"},
        {{"A","B"},{1,3},"L3"},
        {{"A","B"},{3,1},"L4"}
    };
    siteToWeights["B"] = {
        {{"B","A"},{2,0},"L2"},
        {{"B","A"},{0,2},"L1"},
        {{"B","A"},{1,3},"L4"},
        {{"B","A"},{3,1},"L3"}
    };

    if(init_type == "RANDOM") {
        init_RANDOM();
    } else if (init_type == "AFM") {
        init_AFM();
    } else if (init_type == "XPRST") {
        init_ALIGNX();
    } else if (init_type == "ZPRST") {
        init_ALIGNZ();
    } else if (init_type == "VBS") {
        init_VBS();
    } else {
        std::string message = "[Cluster_2x2_ABBA] Unsupported cluster initialization: "+ init_type;
        throw std::runtime_error(message);
    }
}

std::string Cluster_2x2_ABBA::vertexToId(Vertex const& v) const { 
	int vx = (v.r[0] + std::abs(v.r[0])*lX) % lX;
    int vy = std::abs(v.r[1]);   
    auto elemV = Vertex((vx + vy) % 2, 0);
    return vToId.at(elemV);
}

void Cluster_2x2_ABBA::init_RANDOM() {
    std::cout <<"Initializing by RANDOM TENSORS"<< std::endl;

    auto shift05 = [](Real r){ return r-0.5; };
    for (auto & t : sites) { 
        randomize( t.second );
        t.second.apply(shift05);
    }
}

void Cluster_2x2_ABBA::init_AFM() {
    std::cout <<"Initializing by AFM order A=down, B=up"<< std::endl;

    auto aIA = caux.at("A")[0];
    auto aIB = caux.at("B")[0];

    auto pIA = mphys.at("A");
    auto pIB = mphys.at("B");

    // Spin DOWN on site A, spin   UP on site B
    sites.at("A").set(aIA(1), prime(aIA,1)(1), prime(aIA,2)(1), prime(aIA,3)(1),
        pIA(pIA.m()), 1.0);
    sites.at("B").set(aIB(1), prime(aIB,1)(1), prime(aIB,2)(1), prime(aIB,3)(1),
        pIB(1), 1.0);
} 

void Cluster_2x2_ABBA::init_ALIGNX() {
    std::cout <<"Initializing by PRODUCT STATE along X"<< std::endl;

    auto aIA = caux.at("A")[0];
    auto aIB = caux.at("B")[0];

    auto pIA = mphys.at("A");
    auto pIB = mphys.at("B");

    // compute coefficients
    auto nCk = [](int n, int k) {
        if (k > n) return 0;
        if (k * 2 > n) k = n-k;
        if (k == 0) return 1;

        int result = n;
        for( int i = 2; i <= k; ++i ) {
            result *= (n-i+1);
            result /= i;
        }

        return result;
    };

    double site_norm = 0.0;
    for (int i=1; i<=pIA.m(); i++) {
        sites.at("A").set(aIA(1), prime(aIA,1)(1), prime(aIA,2)(1), prime(aIA,3)(1),
            pIA(i), std::sqrt(nCk(pIA.m()-1,i-1)) );
        site_norm += nCk(pIA.m()-1,i-1);
    }
    sites.at("A") *= 1.0/std::sqrt(site_norm);

    site_norm = 0.0;
    for (int i=1; i<=pIB.m(); i++) {
        sites.at("B").set(aIB(1), prime(aIB,1)(1), prime(aIB,2)(1), prime(aIB,3)(1),
            pIB(i), std::sqrt(nCk(pIB.m()-1,i-1)) );
        site_norm += nCk(pIB.m()-1,i-1);
    }
    sites.at("B") *= 1.0/std::sqrt(site_norm);
}

void Cluster_2x2_ABBA::init_ALIGNZ() {
    std::cout <<"Initializing by PRODUCT STATE along Z +1/2"<< std::endl;

    auto aIA = caux.at("A")[0];
    auto aIB = caux.at("B")[0];

    auto pIA = mphys.at("A");
    auto pIB = mphys.at("B");

    // Spin UP on all sites
    sites.at("A").set(aIA(1), prime(aIA,1)(1), prime(aIA,2)(1), prime(aIA,3)(1),
        pIA(1), 1.0);
    sites.at("B").set(aIB(1), prime(aIB,1)(1), prime(aIB,2)(1), prime(aIB,3)(1),
        pIB(1), 1.0);
}

void Cluster_2x2_ABBA::init_VBS() {
    std::cout <<"Initializing by VERTICAL VBS STATE"<< std::endl;

    auto aIA = caux.at("A")[0];
    auto aIB = caux.at("B")[0];

    auto pIA = mphys.at("A");
    auto pIB = mphys.at("B");

    // Spin UP on all sites
    sites.at("A").set(aIA(1), prime(aIA,1)(1), prime(aIA,2)(1), prime(aIA,3)(1),
        pIA(1), 1.0);
    sites.at("A").set(aIA(1), prime(aIA,1)(1), prime(aIA,2)(1), prime(aIA,3)(2),
        pIA(pIA.m()), -1.0);
    sites.at("B").set(aIB(1), prime(aIB,1)(1), prime(aIB,2)(1), prime(aIB,3)(1),
        pIB(1), 1.0);
    sites.at("B").set(aIB(1), prime(aIB,1)(1), prime(aIB,2)(1), prime(aIB,3)(2),
        pIB(pIB.m()), -1.0);
}

std::unique_ptr<Cluster> Cluster_2x2_ABBA::create(nlohmann::json const& json_cluster) {
    
    int pd = json_cluster["physDim"].get<int>();
    int ad = json_cluster["auxBondDim"].get<int>();
    std::string init_type = json_cluster["initBy"].get<std::string>();

    return std::unique_ptr<Cluster>(new Cluster_2x2_ABBA(init_type,ad,pd));
}


/* 
 *   0 1 2 3
 * 0 A B A B
 * 1 C D C D
 * 2 A B A A
 *
 */
Cluster_2x2_ABCD::Cluster_2x2_ABCD() : Cluster(2,2) { cluster_type = "2X2_ABCD"; }

Cluster_2x2_ABCD::Cluster_2x2_ABCD(int ad, int pd) : Cluster(2,2,ad,pd) 
    { cluster_type = "2X2_ABCD"; }

Cluster_2x2_ABCD::Cluster_2x2_ABCD(std::string init_type, int ad, int pd) 
	: Cluster_2x2_ABCD(ad,pd) {
	
    if (init_type == "FILE") {
        // pass, the elements are initialized outside
        return;
    } 

    // Assume initialization of elements by one of the predefined functions
    siteIds = { "A", "B", "C", "D" };
    SI = { {"A",0}, {"B",1}, {"C",2}, {"D",3} };

    cToS  = {
        {std::make_pair(0,0),"A"},
        {std::make_pair(1,0),"B"},
        {std::make_pair(0,1),"C"},
        {std::make_pair(1,1),"D"}
    };
    vToId = { {{0,0},"A"}, {{1,0},"B"}, {{0,1},"C"}, {{1,1},"D"} };
    idToV = { {"A",{0,0}}, {"B",{1,0}}, {"C",{0,1}}, {"D",{1,1}} };

    auto aIA = Index("A", auxBondDim, AUXLINK);
    auto aIB = Index("B", auxBondDim, AUXLINK);
    auto aIC = Index("C", auxBondDim, AUXLINK);
    auto aID = Index("D", auxBondDim, AUXLINK);
    auto pIA = Index("A", physDim, PHYS);
    auto pIB = Index("B", physDim, PHYS);
    auto pIC = Index("C", physDim, PHYS);
    auto pID = Index("D", physDim, PHYS);

    aux  = {aIA, aIB, aIC, aID};
    phys = {pIA, pIB, pIC, pID};
    maux  = {{"A",aIA},{"B",aIB},{"C",aIC},{"D",aID}};
    mphys = {{"A",pIA},{"B",pIB},{"C",pIC},{"D",pID}};
    for (const auto& id : siteIds) { 
        caux[id] = std::vector<Index>(4);
        for(int i=0; i<caux[id].size(); i++) caux[id][i] = prime(maux.at(id),i);
    }

    auto A = ITensor(aIA, prime(aIA,1), prime(aIA,2), prime(aIA,3), pIA);
    auto B = ITensor(aIB, prime(aIB,1), prime(aIB,2), prime(aIB,3), pIB);
    auto C = ITensor(aIC, prime(aIC,1), prime(aIC,2), prime(aIC,3), pIC);
    auto D = ITensor(aID, prime(aID,1), prime(aID,2), prime(aID,3), pID);

    sites = {{"A", A}, {"B", B}, {"C",C}, {"D",D}};

    // Define siteToWeights
    siteToWeights["A"] = {
        {{"A","B"},{2,0},"L1"},
        {{"A","B"},{0,2},"L2"},
        {{"A","C"},{1,3},"L3"},
        {{"A","C"},{3,1},"L4"}
    };
    siteToWeights["B"] = {
        {{"B","A"},{2,0},"L2"},
        {{"B","A"},{0,2},"L1"},
        {{"B","D"},{1,3},"L5"},
        {{"B","D"},{3,1},"L6"}
    };
    siteToWeights["C"] = {
        {{"C","D"},{2,0},"L7"},
        {{"C","D"},{0,2},"L8"},
        {{"C","A"},{1,3},"L4"},
        {{"C","A"},{3,1},"L3"}
    };
    siteToWeights["D"] = {
        {{"D","B"},{3,1},"L5"},
        {{"D","B"},{1,3},"L6"},
        {{"D","C"},{2,0},"L8"},
        {{"D","C"},{0,2},"L7"}
    };

    if (init_type == "RND_AB") {
        init_RANDOM_BIPARTITE();
    } else if(init_type == "RANDOM") {
        init_RANDOM();
    } else if (init_type == "AFM") {
        init_AFM();
    } else if (init_type == "XPRST") {
        init_ALIGNX();
    } else if (init_type == "ZPRST") {
        init_ALIGNZ();
    } else if (init_type == "VBS") {
        init_VBS();
    } else {
        std::string message = "[Cluster_2x2_ABCD] Unsupported cluster initialization: "+ init_type;
        throw std::runtime_error(message);
    }
}

std::string Cluster_2x2_ABCD::vertexToId(Vertex const& v) const { 
	auto elemV = Vertex(
		(v.r[0] + std::abs(v.r[0])*2)% 2, 
		(v.r[1] + std::abs(v.r[1])*2)% 2);
	return vToId.at(elemV);
}

void Cluster_2x2_ABCD::init_RANDOM() {
	std::cout <<"Initializing by RANDOM TENSORS"<< std::endl;

	auto shift05 = [](Real r){ return r-0.5; };
	for (auto & t : sites) { 
		randomize( t.second );
		t.second.apply(shift05);
	}
}

void Cluster_2x2_ABCD::init_RANDOM_BIPARTITE() {
	std::cout <<"Initializing by RANDOM TENSORS A,B,C=B,D=A"<< std::endl;

    auto aIA = caux.at("A")[0];
    auto aIB = caux.at("B")[0];
    auto aIC = caux.at("C")[0];
    auto aID = caux.at("D")[0];

    auto pIA = mphys.at("A");
    auto pIB = mphys.at("B");
    auto pIC = mphys.at("C");
    auto pID = mphys.at("D");

    randomize(sites.at("A"));
    randomize(sites.at("B"));

    auto shift05 = [](double r){ return r-0.5; };
    sites.at("A").apply(shift05);
    sites.at("B").apply(shift05);

    sites.at("C") = sites.at("B") * delta(pIB, pIC);
    sites.at("D") = sites.at("A") * delta(pIA, pID);
    for (int i=0; i<=3; ++i) {
        sites.at("C") = sites.at("C") * delta(prime(aIB,i), prime(aIC,i));
        sites.at("D") = sites.at("D") * delta(prime(aIA,i), prime(aID,i));
    }
}

void Cluster_2x2_ABCD::init_AFM() {
    std::cout <<"Initializing by AFM order A=down, B=up"<< std::endl;

    auto aIA = caux.at("A")[0];
    auto aIB = caux.at("B")[0];
    auto aIC = caux.at("C")[0];
    auto aID = caux.at("D")[0];

    auto pIA = mphys.at("A");
    auto pIB = mphys.at("B");
    auto pIC = mphys.at("C");
    auto pID = mphys.at("D");

    // Spin DOWN on site A, spin   UP on site B
    // Spin UP   on site C, spin DOWN on site D
    sites.at("A").set(aIA(1), prime(aIA,1)(1), prime(aIA,2)(1), prime(aIA,3)(1),
        pIA(1), 1.0);
    sites.at("B").set(aIB(1), prime(aIB,1)(1), prime(aIB,2)(1), prime(aIB,3)(1),
        pIB(pIB.m()), 1.0);
    sites.at("C").set(aIC(1), prime(aIC,1)(1), prime(aIC,2)(1), prime(aIC,3)(1),
        pIC(pIC.m()), 1.0);
    sites.at("D").set(aID(1), prime(aID,1)(1), prime(aID,2)(1), prime(aID,3)(1),
        pID(1), 1.0);
} 

void Cluster_2x2_ABCD::init_ALIGNX() {
	std::cout <<"Initializing by PRODUCT STATE along X"<< std::endl;

    auto aIA = caux.at("A")[0];
    auto aIB = caux.at("B")[0];
    auto aIC = caux.at("C")[0];
    auto aID = caux.at("D")[0];

    auto pIA = mphys.at("A");
    auto pIB = mphys.at("B");
    auto pIC = mphys.at("C");
    auto pID = mphys.at("D");

     // compute coefficients
    auto nCk = [](int n, int k) {
        if (k > n) return 0;
        if (k * 2 > n) k = n-k;
        if (k == 0) return 1;

        int result = n;
        for( int i = 2; i <= k; ++i ) {
            result *= (n-i+1);
            result /= i;
        }

        return result;
    };

    auto setAlignAlongX = [&nCk] (ITensor & site, Index & ia, Index & ip) {
        double site_norm = 0.0;
        for (int i=1; i<=ip.m(); i++) {
            site.set(ia(1), prime(ia,1)(1), prime(ia,2)(1), prime(ia,3)(1),
                ip(i), std::sqrt(nCk(ip.m()-1,i-1)) );
            site_norm += nCk(ip.m()-1,i-1);
        }
        site *= 1.0/std::sqrt(site_norm);
    };

    setAlignAlongX(sites.at("A"), aIA, pIA);
    setAlignAlongX(sites.at("B"), aIB, pIB);
    setAlignAlongX(sites.at("C"), aIC, pIC);
    setAlignAlongX(sites.at("D"), aID, pID);
}

void Cluster_2x2_ABCD::init_ALIGNZ() {
	std::cout <<"Initializing by PRODUCT STATE along Z +1/2"<< std::endl;

    auto aIA = caux.at("A")[0];
    auto aIB = caux.at("B")[0];
    auto aIC = caux.at("C")[0];
    auto aID = caux.at("D")[0];

    auto pIA = mphys.at("A");
    auto pIB = mphys.at("B");
    auto pIC = mphys.at("C");
    auto pID = mphys.at("D");

    // Spin UP on all sites
    sites.at("A").set(aIA(1), prime(aIA,1)(1), prime(aIA,2)(1), prime(aIA,3)(1),
        pIA(1), 1.0);
    sites.at("B").set(aIB(1), prime(aIB,1)(1), prime(aIB,2)(1), prime(aIB,3)(1),
        pIB(1), 1.0);
    sites.at("C").set(aIC(1), prime(aIC,1)(1), prime(aIC,2)(1), prime(aIC,3)(1),
        pIC(1), 1.0);
    sites.at("D").set(aID(1), prime(aID,1)(1), prime(aID,2)(1), prime(aID,3)(1),
        pID(1), 1.0);
}

void Cluster_2x2_ABCD::init_VBS() {
	std::cout <<"Initializing by VERTICAL VBS STATE"<< std::endl;

    auto aIA = caux.at("A")[0];
    auto aIB = caux.at("B")[0];
    auto aIC = caux.at("C")[0];
    auto aID = caux.at("D")[0];

    auto pIA = mphys.at("A");
    auto pIB = mphys.at("B");
    auto pIC = mphys.at("C");
    auto pID = mphys.at("D");

    // Spin UP on all sites
    sites.at("A").set(aIA(1), prime(aIA,1)(1), prime(aIA,2)(1), prime(aIA,3)(1),
        pIA(1), 1.0);
    sites.at("A").set(aIA(1), prime(aIA,1)(1), prime(aIA,2)(1), prime(aIA,3)(2),
        pIA(pIA.m()), -1.0);
    sites.at("B").set(aIB(1), prime(aIB,1)(1), prime(aIB,2)(1), prime(aIB,3)(1),
        pIB(1), 1.0);
    sites.at("B").set(aIB(1), prime(aIB,1)(1), prime(aIB,2)(1), prime(aIB,3)(2),
        pIB(pIB.m()), -1.0);
    sites.at("C").set(aIC(1), prime(aIC,1)(2), prime(aIC,2)(1), prime(aIC,3)(1),
        pIC(1), 1.0);
    sites.at("C").set(aIC(1), prime(aIC,1)(1), prime(aIC,2)(1), prime(aIC,3)(1),
        pIC(pIC.m()), 1.0);
    sites.at("D").set(aID(1), prime(aID,1)(2), prime(aID,2)(1), prime(aID,3)(1),
        pID(1), 1.0);
    sites.at("D").set(aID(1), prime(aID,1)(1), prime(aID,2)(1), prime(aID,3)(1),
        pID(pID.m()), 1.0);
}

std::unique_ptr<Cluster> Cluster_2x2_ABCD::create(nlohmann::json const& json_cluster) {
    
    int pd = json_cluster["physDim"].get<int>();
    int ad = json_cluster["auxBondDim"].get<int>();
    std::string init_type = json_cluster["initBy"].get<std::string>();

    return std::unique_ptr<Cluster>(new Cluster_2x2_ABCD(init_type,ad,pd));
}


} //namespace itensor