#include "ctm-cluster.h"

using namespace std; 

ostream& operator<<(ostream& s, Cluster const& c) {
    s <<"Cluster( sizeN: "<< c.sizeN <<", sizeM: "<< c.sizeM 
        <<", auxBondDim: "<< c.auxBondDim << endl;

    s <<"siteIds: [ ";
    for( const auto& siteId : c.siteIds ) {
        s << siteId <<" ";
    }
    s <<"]"<< endl;

    s <<"clusterToSite: ["<< endl; 
    for( const auto& cToSEntry : c.cToS ) {
        s <<"("<< cToSEntry.first.first <<", "<< cToSEntry.first.second 
            <<") -> "<< cToSEntry.second << endl;
    }
    s << "]" << endl;
    
    for( const auto& sitesEntry : c.sites ) {
        s << sitesEntry.first <<" = ";
        itensor::printfln("%f", sitesEntry.second);  
    }

    return s;
}
