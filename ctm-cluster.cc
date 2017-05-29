#include "ctm-cluster.h"

using namespace itensor; 

ITensor contractCluster(Cluster const& c) {
    std::cout <<">>>> contractCluster called <<<<<"<< std::endl;

    // Contract cluster
    ITensor initPlaq, siteT;
    Index tempI;
    std::vector<Index> aI; 

    // First on-site tensor
    initPlaq = c.sites.at( c.cToS.at(std::make_pair(0,0)) );
    std::cout << c.cToS.at(std::make_pair(0,0));
    aI.push_back( noprime( 
        findtype(initPlaq.inds(), AUXLINK) ) );

    for (int col = 1; col < c.sizeM; col ++) {        
        siteT = c.sites.at( c.cToS.at(std::make_pair(col,0)) );
        std::cout << c.cToS.at(std::make_pair(col,0));
        aI.push_back( noprime( 
            findtype(siteT.inds(), AUXLINK) ) );
        initPlaq *= ( delta( 
            prime(aI[col-1],2), aI[col] ) * siteT );
    }

    std::cout <<" >>>> row 0 contrated <<<<<"<< std::endl;

    for (int row = 1; row < c.sizeN; row ++) {
        siteT = c.sites.at( c.cToS.at(std::make_pair(0,row)) );
        std::cout << c.cToS.at(std::make_pair(0,row));
        tempI = noprime( 
            findtype(siteT.inds(), AUXLINK) );
        initPlaq *= ( siteT * delta( 
            prime(aI[0],3), prime(tempI,1) ) );
        aI[0] = tempI;

        for (int col = 1; col < c.sizeM; col ++) {
            auto siteT = c.sites.at( c.cToS.at(std::make_pair(col,row)) );
            std::cout << c.cToS.at(std::make_pair(col,row));
            tempI = noprime( 
                findtype(siteT.inds(), AUXLINK) );
            initPlaq *= ( siteT 
                * delta( prime(aI[col],3), prime(tempI,1) )
                * delta( prime(aI[col-1],2), tempI) );
            aI[col] = tempI;
        }

        std::cout <<" >>>> row "<< row <<" contrated <<<<<"<< std::endl;
    }

    std::cout <<">>>> contractCluster done <<<<<"<< std::endl;
    return initPlaq;
}

std::ostream& operator<<(std::ostream& s, Cluster const& c) {
    s <<"Cluster( sizeN: "<< c.sizeN <<", sizeM: "<< c.sizeM 
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
