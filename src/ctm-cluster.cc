#include "ctm-cluster.h"

using namespace itensor;

const int Cluster::BRAKET_OFFSET = 4;

Shift operator * (int x, Shift const& s) {
  return s * x;
}

void initClusterSites(Cluster & c, bool dbg) {
    // reset
    c.aux.clear();
    c.phys.clear();
    c.sites.clear();

    for (auto & site : c.siteIds) {
        c.aux.push_back( Index(TAG_I_AUX, c.auxBondDim, AUXLINK) );
        c.phys.push_back( Index(TAG_I_PHYS, c.physDim, PHYS) );
    
        c.sites[site] = ITensor(c.phys.back(), c.aux.back(), 
            prime(c.aux.back(),1), prime(c.aux.back(),2), prime(c.aux.back(),3) );
    }
}

void initClusterWeights(Cluster & c, bool dbg) {
    if (c.siteToWeights.size() == 0) {
        std::cout<<"[initClusterWeights]"<<" no weights stored for this cluster"
            << std::endl;
        throw std::runtime_error("Invalid input");
    }

    // reset
    c.weights.clear();

    // create map holding LinkWeights
    std::map<std::string, LinkWeight> tmpLWs;

    for (const auto& lwSet : c.siteToWeights) // map< string, vector<LinkWeight> >
        for (const auto& lw : lwSet.second) // loop over linkWeights
            if ( (tmpLWs.find( lw.wId ) == tmpLWs.end()) ) tmpLWs[lw.wId] = lw;

    for (const auto& lw : tmpLWs)
        c.weights[lw.second.wId] = ITensor(
            c.AIc(lw.second.sId[0],lw.second.dirs[0]),
            c.AIc(lw.second.sId[1],lw.second.dirs[1])
        );
}

void setWeights(Cluster & c, std::string option, bool dbg) {
    if (c.siteToWeights.size() == 0) {
        std::cout<<"[setWeights]"<<" no weights stored for this cluster"
            << std::endl;
        throw std::runtime_error("Invalid input");
    }

    if(option == "DELTA") {
        for (auto & wEntry : c.weights) // map < string, tensor >
        {
            std::vector<double> tmpD(wEntry.second.inds()[0].m(), 1.0);
            wEntry.second = diagTensor(tmpD, wEntry.second.inds());
        }
    } else {
        std::cout <<"[setWeights] ctm-cluster setWeights unsupported option: "
            << option << std::endl;
        throw std::runtime_error("Invalid option");
    }
}

void saveWeights(Cluster & c, bool dbg)  {
    c.old_weights = c.weights;
}

double weightDist(Cluster const& c) {
    double res = 0.0;
    for (auto const& e : c.weights) {
        res += norm(c.weights.at(e.first) - c.old_weights.at(e.first));
    }
    return res;
}

void setSites(Cluster & c, std::string option, bool dbg) {

    // reset all sites to zero tensors
    for (auto & se : c.sites) se.second.fill(0.0);

    if (option == "RANDOM") {
        std::cout <<"Initializing by RANDOM TENSORS"<< std::endl;

        auto shift05 = [](Real r){ return r-0.5; };
        
        for (auto & se : c.sites) {
            randomize(se.second);
            se.second.apply(shift05);
        }

    } else if (option == "RND_1S") {
        std::cout <<"Initializing by SINGLE RANDOM TENSOR"<< std::endl;

        auto shift05 = [](Real r){ return r-0.5; };

        // create random tensor
        Index pI(TAG_I_PHYS,c.physDim,PHYS), aI(TAG_I_AUX,c.auxBondDim,AUXLINK);
        ITensor tmpT(pI,aI,prime(aI,1),prime(aI,2),prime(aI,3));
        randomize(tmpT);
        tmpT.apply(shift05);

        for (auto & se : c.sites) {
            auto sId = se.first;
            auto tmpDeltaAux = delta(aI,c.aux.at(c.SI.at(sId)));
            
            se.second = (((( (tmpT * delta(pI, c.phys.at(c.SI.at(sId))))
                *tmpDeltaAux )
                *prime(tmpDeltaAux,1) )
                *prime(tmpDeltaAux,2) )
                *prime(tmpDeltaAux,3) );
        }

    } else if (option == "XPRST") {
        std::cout <<"Initializing by PRODUCT STATE along X"<< std::endl;
        
        for (auto & se : c.sites) {
            auto sId = se.first;
            auto tmpPI = c.phys.at(c.SI.at(sId));
            auto tmpAI = c.aux.at(c.SI.at(sId));

            se.second.set(tmpAI(1), prime(tmpAI,1)(1), prime(tmpAI,2)(1), prime(tmpAI,3)(1),
                tmpPI(1), 1.0/std::sqrt(2.0));
            se.second.set(tmpAI(1), prime(tmpAI,1)(1), prime(tmpAI,2)(1), prime(tmpAI,3)(1),
                tmpPI(2), 1.0/std::sqrt(2.0));            
        }

    } else if (option == "ZPRST") {
        std::cout <<"Initializing by PRODUCT STATE along Z, m_z = 1/2"<< std::endl;
        
        for (auto & se : c.sites) {
            auto sId = se.first;
            auto tmpPI = c.phys.at(c.SI.at(sId));
            auto tmpAI = c.aux.at(c.SI.at(sId));

            se.second.set(tmpAI(1), prime(tmpAI,1)(1), prime(tmpAI,2)(1), prime(tmpAI,3)(1),
                tmpPI(1), 1.0);            
        }

    } else if (option == "mZPRST") {
        std::cout <<"Initializing by PRODUCT STATE along Z, m_z = -1/2"<< std::endl;
        
        for (auto & se : c.sites) {
            auto sId = se.first;
            auto tmpPI = c.phys.at(c.SI.at(sId));
            auto tmpAI = c.aux.at(c.SI.at(sId));

            se.second.set(tmpAI(1), prime(tmpAI,1)(1), prime(tmpAI,2)(1), prime(tmpAI,3)(1),
                tmpPI(2), 1.0);            
        }

    } else if (option == "NEEL") {
        if ( !((c.sizeN % 2 == 0) && (c.sizeM % 2 == 0)) ) {
            std::cout <<"ctm-cluster setSites option: NEEL unit cell is not" 
                <<"even in both dimensions"<< std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout <<"Initializing by NEEL STATE"<< std::endl;
        
        for (int y=0; y<c.sizeN; y++) { 
            for (int x=0; x<c.sizeM; x++) {
                auto pos = std::make_pair(x,y);

                auto sId = c.cToS.at(pos);
                auto tmpPI = c.phys.at(c.SI.at(sId));
                auto tmpAI = c.aux.at(c.SI.at(sId));

                int p = 1 + ((y % 2) + (x % 2)) % 2;
                c.sites.at(sId).set(tmpAI(1), prime(tmpAI,1)(1), prime(tmpAI,2)(1), prime(tmpAI,3)(1),
                    tmpPI(p), 1.0);           
            }
        }

    } else {
        std::cout <<"ctm-cluster setSites unsupported option: "<< option << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ITensor contractCluster(Cluster const& c, bool dbg) {
//     std::cout <<">>>> contractCluster called <<<<<"<< std::endl;

//     // Contract all inner links of the cluster
//     ITensor initPlaq, siteT;
//     Index tempI;
//     std::vector<Index> aI;

//     // First on-site tensor
//     initPlaq = c.sites.at( c.cToS.at(std::make_pair(0,0)) );
//     if(dbg) std::cout << c.cToS.at(std::make_pair(0,0));
//     aI.push_back( noprime( 
//         findtype(initPlaq, AUXLINK) ) );

//     for (int col = 1; col < c.sizeM; col ++) {        
//         siteT = c.sites.at( c.cToS.at(std::make_pair(col,0)) );
//         if(dbg) std::cout << c.cToS.at(std::make_pair(col,0));
//         aI.push_back( noprime( 
//             findtype(siteT, AUXLINK) ) );
//         initPlaq *= ( delta( 
//             prime(aI[col-1],2), aI[col] ) * siteT );
//     }

//     if(dbg) std::cout <<" >>>> row 0 contrated <<<<<"<< std::endl;

//     for (int row = 1; row < c.sizeN; row ++) {
//         siteT = c.sites.at( c.cToS.at(std::make_pair(0,row)) );
//         if(dbg) std::cout << c.cToS.at(std::make_pair(0,row));
//         tempI = noprime( 
//             findtype(siteT, AUXLINK) );
//         initPlaq *= ( siteT * delta( 
//             prime(aI[0],3), prime(tempI,1) ) );
//         aI[0] = tempI;

//         for (int col = 1; col < c.sizeM; col ++) {
//             auto siteT = c.sites.at( c.cToS.at(std::make_pair(col,row)) );
//             if(dbg) std::cout << c.cToS.at(std::make_pair(col,row));
//             tempI = noprime( 
//                 findtype(siteT, AUXLINK) );
//             initPlaq *= ( siteT 
//                 * delta( prime(aI[col],3), prime(tempI,1) )
//                 * delta( prime(aI[col-1],2), tempI) );
//             aI[col] = tempI;
//         }

//         if(dbg) std::cout <<" >>>> row "<< row <<" contrated <<<<<"<< std::endl;
//     }

//     if(dbg) std::cout <<">>>> contractCluster done <<<<<"<< std::endl;
//     return initPlaq;
// }

/*
 * Exact contraction of a cluster with ... rectangular 
 * unit cell, physical indices being uncontracted
 *
 */
// ITensor clusterDenMat(Cluster const& c, bool dbg) {
//     std::cout <<">>>> contractCluster called <<<<<"<< std::endl;

//     // Contract all inner links of the cluster
//     ITensor initPlaq, siteT, D, eorD, eocD;
//     Index tempI;
//     std::vector<Index> aI; 

//     std::string tId;   

//     // First on-site tensor - contract by aux indices with primes 0(left) and 1(up) 
//     tId = c.cToS.at(std::make_pair(0,0));
//     siteT = prime(c.sites.at(tId),4);
//     aI.push_back( noprime(findtype(siteT, AUXLINK)) );
//     D = delta(aI[0],prime(aI[0],4));
    
//     initPlaq = c.sites.at(tId)*D*prime(D,1)*siteT;
//     if(dbg) Print(initPlaq);
    
//     for (int col = 1; col < c.sizeM; col ++) {        
//         tId = c.cToS.at(std::make_pair(col,0));
//         siteT = prime(c.sites.at(tId),4);
//         aI.push_back( noprime(findtype(siteT, AUXLINK)) );
//         D = delta(aI[col],prime(aI[col],4));

//         if(dbg) std::cout << tId;
        
//         eorD = ( col == c.sizeM-1 ) ? prime(D,2) : ITensor(1.0);
//         initPlaq *= (c.sites.at(tId)*prime(D,1)*eorD*siteT) 
//             * delta( prime(aI[col-1],2), aI[col] ) 
//             * delta( prime(aI[col-1],6), prime(aI[col],4) );
//     }

//     if(dbg) std::cout <<" >>>> row 0 contrated <<<<<"<< std::endl;
//     if(dbg) Print(initPlaq);

//     for (int row = 1; row < c.sizeN; row ++) {
//         //First tensor of a new row
//         tId = c.cToS.at(std::make_pair(0,row));
//         siteT = prime(c.sites.at(tId),4);
//         if(dbg) std::cout << tId;
        
//         tempI = noprime( findtype(siteT, AUXLINK) );
//         D = delta(tempI,prime(tempI,4));

//         eocD = ( row == c.sizeN-1 ) ? prime(D,3) : ITensor(1.0);
//         initPlaq *= (c.sites.at(tId)*D*eocD*siteT)
//             * delta( prime(aI[0],3), prime(tempI,1)) 
//             * delta( prime(aI[0],7), prime(tempI,5));
//         aI[0] = tempI;

//         if(dbg) Print(initPlaq);

//         for (int col = 1; col < c.sizeM; col ++) {
//             tId = c.cToS.at(std::make_pair(col,row));
//             siteT = prime(c.sites.at(tId),4);
//             if(dbg) std::cout << tId;

//             tempI = noprime( findtype(siteT, AUXLINK) );
//             D = delta(tempI,prime(tempI,4));

//             eocD = ( row == c.sizeN-1 ) ? prime(D,3) : ITensor(1.0);
//             eorD = ( col == c.sizeM-1 ) ? prime(D,2) : ITensor(1.0);
//             initPlaq *= (c.sites.at(tId)*eocD*eorD*siteT)
//                 * delta( prime(aI[col],3), prime(tempI,1) )
//                 * delta( prime(aI[col],7), prime(tempI,5) )
//                 * delta( prime(aI[col-1],2), tempI)
//                 * delta( prime(aI[col-1],6), prime(tempI,4));
//             aI[col] = tempI;
//         }

//         if(dbg) std::cout <<" >>>> row "<< row <<" contrated <<<<<"<< std::endl;
//         if(dbg) Print(initPlaq);
//     }

//     initPlaq.mapprime(PHYS,4,1);

//     if(dbg) std::cout <<">>>> contractCluster done <<<<<"<< std::endl;
//     return initPlaq;
// }

void Cluster::absorbWeightsToSites(bool dbg) {

    auto sqrtT = [](double r) { return std::sqrt(r); };
    auto quadT = [](double r) { return r*r; };

    // apply sqrtT to all wight tensors
    for ( auto & w : weights) w.second.apply(sqrtT);

    for ( auto & siteEntry : sites ) {
        auto sId = siteEntry.first;
        // contract each on-site tensor with its weights
        // and set back the original index
        for ( auto const& stw : siteToWeights.at(sId) ) {
            siteEntry.second *= weights.at(stw.wId);
            siteEntry.second *= delta(weights.at(stw.wId).inds());
            // siteEntry.second *= c.DContract(stw.sId[0],stw.dirs[0],stw.sId[1],stw.dirs[1]);
        }
    }

    // apply quadT to all weight tensors to recover original ones
    for ( auto & w : weights) w.second.apply(quadT);
}

void mvSite(Cluster const& c, std::pair<int,int> &s, int dir) {
    dir = dir % 4;
    switch(dir){
        case 0: {
            mvSite(c, s, std::make_pair(-1,0));
            break;
        }
        case 1: {
            mvSite(c, s, std::make_pair(0,-1));
            break;
        }
        case 2: {
            mvSite(c, s, std::make_pair(1,0));
            break;
        }
        case 3: {
            mvSite(c, s, std::make_pair(0,1));
            break;
        }
    }
}

void mvSite(Cluster const& c, std::pair<int,int> &s, std::pair<int,int> const& disp) {
    s.first  += disp.first;
    s.second += disp.second;

    // apply BC
    s.first = ((s.first % c.sizeM) + c.sizeM) % c.sizeM;
    s.second = ((s.second % c.sizeN) + c.sizeN) % c.sizeN;
}

std::pair<int,int> getNextSite(Cluster const& c, std::pair<int,int> const& s, int dir) {
    std::pair<int,int> res(s);
    mvSite(c, res, dir);
    return res;
}

std::ostream& operator<<(std::ostream& os, Shift const& s) {
    return os <<"Shift("<<s.d[0]<<" "<<s.d[1]<<")";
}

std::ostream& operator<<(std::ostream& os, Vertex const& v) {
    return os <<"Vertex("<<v.r[0]<<" "<<v.r[1]<<")";
}

std::ostream& operator<<(std::ostream& s, Cluster const& c) {
    s <<"Cluster( metaInfo: "<< c.metaInfo 
        << "sizeM: "<< c.sizeM <<", sizeN: "<< c.sizeN
        << " | lX: "<< c.lX <<" , lY: "<< c.lY 
        <<", auxBondDim: "<< c.auxBondDim << std::endl;

    s <<"siteIds: [ ";
    for( const auto& siteId : c.siteIds ) {
        s << siteId <<" ";
    }
    s <<"]"<< std::endl;

    s <<"SI: ["<<std::endl;
    for( const auto& idToPos : c.SI ) {
        s << idToPos.first <<" --> " << idToPos.second <<" --> "
            << c.sites.at(idToPos.first) << std::endl;
    }
    s <<"]"<< std::endl;


    s <<"Indices phys AND aux: ["<< std::endl;
    for( unsigned int i=0; i<c.aux.size(); i++ ) {
        s << c.siteIds[i]<< ": " << c.phys[i] <<" -- "<< c.aux[i] << std::endl;
    }
    s <<"]"<< std::endl;

    s<<"PHYS indices: ["<< std::endl;
    for(auto const& e : c.mphys) s << e.first <<" : "<< e.second << std::endl;
    s<<"]"<< std::endl;

    s<<"AUX indices: ["<< std::endl;
    for(auto const& e : c.caux) {
        s << e.first <<" : ";
        for(auto const& i : e.second) s << i <<" ";
        s << std::endl;
    }
    s<<"]"<< std::endl;

    s <<"clusterToSite: ["<< std::endl; 
    for( const auto& cToSEntry : c.cToS ) {
        s <<"("<< cToSEntry.first.first <<", "<< cToSEntry.first.second 
            <<") -> "<< cToSEntry.second <<" -> " << c.SI.at(cToSEntry.second) << std::endl;
    }
    s << "]" << std::endl;
    
    s <<"VertexToSite: ["<< std::endl; 
    for( const auto& vId : c.vToId ) {
        s << vId.first <<" --> "<< vId.second <<" --> "<< c.getSite(vId.first) << std::endl;
    }
    s << "]" << std::endl;

    s << "siteToWeights: [" << std::endl;
    for( const auto& lwEntrySet : c.siteToWeights ) {
        s << lwEntrySet.first <<" --> ["<< std::endl;
        for ( const auto& lwEntry : lwEntrySet.second) s << lwEntry << std::endl;
        s << "]" << std::endl;
    }
    s << std::endl;

    s << "weights: [" << std::endl;
    for ( const auto& wEntry : c.weights ) { 
        s << wEntry.first << " --> ";
        printfln("%f", wEntry.second);
    }
    s << "]" << std::endl;

    return s;
}

std::ostream& operator<<(std::ostream& s, LinkWeight const& lw) {
    s <<"LinkWeight( "<< lw.wId <<" ["
        << lw.sId[0] <<","<< lw.sId[1] <<"], ["
        << lw.dirs[0] <<","<< lw.dirs[1] <<"])";

    return s;
}
