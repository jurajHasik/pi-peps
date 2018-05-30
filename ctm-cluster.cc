#include "ctm-cluster.h"

using namespace itensor; 

void initClusterSites(Cluster & c, bool dbg) {
    for (auto & site : c.siteIds) {
        c.aux.push_back( Index(TAG_I_AUX, c.auxBondDim, AUXLINK) );
        c.phys.push_back( Index(TAG_I_PHYS, c.physDim, PHYS) );
    
        c.sites[site] = ITensor(c.phys.back(), c.aux.back(), 
            prime(c.aux.back(),1), prime(c.aux.back(),2), prime(c.aux.back(),3) );
    }
}

void initClusterWeights(Cluster & c, bool dbg) {
    // create map holding LinkWeights
    std::map<std::string, LinkWeight> tmpLWs;

    for (const auto& lwSet : c.siteToWeights) 
        for (const auto& lw : lwSet.second) 
            if ( (tmpLWs.find( lw.wId ) == tmpLWs.end()) ) tmpLWs[lw.wId] = lw;

    for (const auto& lw : tmpLWs) 
        c.weights[lw.second.wId] = ITensor(
            prime(c.aux[c.SI.at(lw.second.sId[0])],lw.second.dirs[0]),
            prime(c.aux[c.SI.at(lw.second.sId[1])],lw.second.dirs[1])
            );
}

void setWeights(Cluster & c, std::string option, bool dbg) {
    if(option == "DELTA") {
        for (auto & wEntry : c.weights) {
            std::vector<double> tmpD(c.auxBondDim, 1.0);
            wEntry.second = diagTensor(tmpD, 
                wEntry.second.inds()[0], wEntry.second.inds()[1] );
        }
    } else {
        std::cout <<"ctm-cluster setWeights unsupported option: "
            << option << std::endl;
        exit(EXIT_FAILURE);
    }
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

    } else {
        std::cout <<"ctm-cluster setSites unsupported option: "<< option << std::endl;
    }
}

ITensor contractCluster(Cluster const& c, bool dbg) {
    std::cout <<">>>> contractCluster called <<<<<"<< std::endl;

    // Contract all inner links of the cluster
    ITensor initPlaq, siteT;
    Index tempI;
    std::vector<Index> aI; 

    // First on-site tensor
    initPlaq = c.sites.at( c.cToS.at(std::make_pair(0,0)) );
    if(dbg) std::cout << c.cToS.at(std::make_pair(0,0));
    aI.push_back( noprime( 
        findtype(initPlaq.inds(), AUXLINK) ) );

    for (int col = 1; col < c.sizeM; col ++) {        
        siteT = c.sites.at( c.cToS.at(std::make_pair(col,0)) );
        if(dbg) std::cout << c.cToS.at(std::make_pair(col,0));
        aI.push_back( noprime( 
            findtype(siteT.inds(), AUXLINK) ) );
        initPlaq *= ( delta( 
            prime(aI[col-1],2), aI[col] ) * siteT );
    }

    if(dbg) std::cout <<" >>>> row 0 contrated <<<<<"<< std::endl;

    for (int row = 1; row < c.sizeN; row ++) {
        siteT = c.sites.at( c.cToS.at(std::make_pair(0,row)) );
        if(dbg) std::cout << c.cToS.at(std::make_pair(0,row));
        tempI = noprime( 
            findtype(siteT.inds(), AUXLINK) );
        initPlaq *= ( siteT * delta( 
            prime(aI[0],3), prime(tempI,1) ) );
        aI[0] = tempI;

        for (int col = 1; col < c.sizeM; col ++) {
            auto siteT = c.sites.at( c.cToS.at(std::make_pair(col,row)) );
            if(dbg) std::cout << c.cToS.at(std::make_pair(col,row));
            tempI = noprime( 
                findtype(siteT.inds(), AUXLINK) );
            initPlaq *= ( siteT 
                * delta( prime(aI[col],3), prime(tempI,1) )
                * delta( prime(aI[col-1],2), tempI) );
            aI[col] = tempI;
        }

        if(dbg) std::cout <<" >>>> row "<< row <<" contrated <<<<<"<< std::endl;
    }

    if(dbg) std::cout <<">>>> contractCluster done <<<<<"<< std::endl;
    return initPlaq;
}

/*
 * Exact contraction of a cluster with ... rectangular 
 * unit cell, physical indices being uncontracted
 *
 */
ITensor clusterDenMat(Cluster const& c, bool dbg) {
    std::cout <<">>>> contractCluster called <<<<<"<< std::endl;

    // Contract all inner links of the cluster
    ITensor initPlaq, siteT, D, eorD, eocD;
    Index tempI;
    std::vector<Index> aI; 

    std::string tId;   

    // First on-site tensor - contract by aux indices with primes 0(left) and 1(up) 
    tId = c.cToS.at(std::make_pair(0,0));
    siteT = prime(c.sites.at(tId),4);
    aI.push_back( noprime(findtype(siteT.inds(), AUXLINK)) );
    D = delta(aI[0],prime(aI[0],4));
    
    initPlaq = c.sites.at(tId)*D*prime(D,1)*siteT;
    if(dbg) Print(initPlaq);
    
    for (int col = 1; col < c.sizeM; col ++) {        
        tId = c.cToS.at(std::make_pair(col,0));
        siteT = prime(c.sites.at(tId),4);
        aI.push_back( noprime(findtype(siteT.inds(), AUXLINK)) );
        D = delta(aI[col],prime(aI[col],4));

        if(dbg) std::cout << tId;
        
        eorD = ( col == c.sizeM-1 ) ? prime(D,2) : ITensor(1.0);
        initPlaq *= (c.sites.at(tId)*prime(D,1)*eorD*siteT) 
            * delta( prime(aI[col-1],2), aI[col] ) 
            * delta( prime(aI[col-1],6), prime(aI[col],4) );
    }

    if(dbg) std::cout <<" >>>> row 0 contrated <<<<<"<< std::endl;
    if(dbg) Print(initPlaq);

    for (int row = 1; row < c.sizeN; row ++) {
        //First tensor of a new row
        tId = c.cToS.at(std::make_pair(0,row));
        siteT = prime(c.sites.at(tId),4);
        if(dbg) std::cout << tId;
        
        tempI = noprime( findtype(siteT.inds(), AUXLINK) );
        D = delta(tempI,prime(tempI,4));

        eocD = ( row == c.sizeN-1 ) ? prime(D,3) : ITensor(1.0);
        initPlaq *= (c.sites.at(tId)*D*eocD*siteT)
            * delta( prime(aI[0],3), prime(tempI,1)) 
            * delta( prime(aI[0],7), prime(tempI,5));
        aI[0] = tempI;

        if(dbg) Print(initPlaq);

        for (int col = 1; col < c.sizeM; col ++) {
            tId = c.cToS.at(std::make_pair(col,row));
            siteT = prime(c.sites.at(tId),4);
            if(dbg) std::cout << tId;

            tempI = noprime( findtype(siteT.inds(), AUXLINK) );
            D = delta(tempI,prime(tempI,4));

            eocD = ( row == c.sizeN-1 ) ? prime(D,3) : ITensor(1.0);
            eorD = ( col == c.sizeM-1 ) ? prime(D,2) : ITensor(1.0);
            initPlaq *= (c.sites.at(tId)*eocD*eorD*siteT)
                * delta( prime(aI[col],3), prime(tempI,1) )
                * delta( prime(aI[col],7), prime(tempI,5) )
                * delta( prime(aI[col-1],2), tempI)
                * delta( prime(aI[col-1],6), prime(tempI,4));
            aI[col] = tempI;
        }

        if(dbg) std::cout <<" >>>> row "<< row <<" contrated <<<<<"<< std::endl;
        if(dbg) Print(initPlaq);
    }

    initPlaq.mapprime(PHYS,4,1);

    if(dbg) std::cout <<">>>> contractCluster done <<<<<"<< std::endl;
    return initPlaq;
}

void absorbWeightsToSites(Cluster & c, bool dbg) {

    auto sqrtT = [](double r) { return std::sqrt(r); };
    auto quadT = [](double r) { return r*r; };

    // apply sqrtT to all wight tensors
    for ( auto & w : c.weights) w.second.apply(sqrtT);

    for ( auto & siteEntry : c.sites ) {
        auto sId = siteEntry.first;
        // contract each on-site tensor with its weights
        // and set back the original index
        for ( auto const& stw : c.siteToWeights.at(sId) ) {
            siteEntry.second *= c.weights.at(stw.wId);
            siteEntry.second *= delta(prime(c.aux[c.SI[sId]], stw.dirs[0]),
                prime(c.aux[c.SI[stw.sId[1]]], stw.dirs[1]) );
        }
    }

    // apply quadT to all weight tensors to recover original ones
    for ( auto & w : c.weights) w.second.apply(quadT);
}

std::ostream& operator<<(std::ostream& s, Cluster const& c) {
    s <<"Cluster( metaInfo: "<< c.metaInfo 
        << "sizeN: "<< c.sizeN <<", sizeM: "<< c.sizeM 
        <<", auxBondDim: "<< c.auxBondDim << std::endl;

    s <<"siteIds: [ ";
    for( const auto& siteId : c.siteIds ) {
        s << siteId <<" ";
    }
    s <<"]"<< std::endl;

    s <<"Indices phys AND aux: ["<< std::endl;
    for( unsigned int i=0; i<c.aux.size(); i++ ) {
        s << c.siteIds[i]<< ": " << c.phys[i] <<" -- "<< c.aux[i] << std::endl;
    }
    s <<"]"<< std::endl;

    s <<"clusterToSite: ["<< std::endl; 
    for( const auto& cToSEntry : c.cToS ) {
        s <<"("<< cToSEntry.first.first <<", "<< cToSEntry.first.second 
            <<") -> "<< cToSEntry.second <<" -> " << c.SI.at(cToSEntry.second) << std::endl;
    }
    s << "]" << std::endl;
    
    for( const auto& sitesEntry : c.sites ) {
        s << sitesEntry.first <<" = ";
        printfln("%f", sitesEntry.second);
    }

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
