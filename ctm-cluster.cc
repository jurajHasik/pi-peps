#include "ctm-cluster.h"

using namespace itensor; 

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

std::ostream& operator<<(std::ostream& s, Cluster const& c) {
    s <<"Cluster( metaInfo: "<< c.metaInfo 
        << "sizeN: "<< c.sizeN <<", sizeM: "<< c.sizeM 
        <<", auxBondDim: "<< c.auxBondDim << std::endl;

    s <<"siteIds: [ ";
    for( const auto& siteId : c.siteIds ) {
        s << siteId <<" ";
    }
    s <<"]"<< std::endl;

    s <<"clusterToSite: ["<< std::endl; 
    for( const auto& cToSEntry : c.cToS ) {
        s <<"("<< cToSEntry.first.first <<", "<< cToSEntry.first.second 
            <<") -> "<< cToSEntry.second << std::endl;
    }
    s << "]" << std::endl;
    
    for( const auto& sitesEntry : c.sites ) {
        s << sitesEntry.first <<" = ";
        printfln("%f", sitesEntry.second);  
    }

    return s;
}
