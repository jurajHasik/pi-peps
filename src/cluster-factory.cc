#include "pi-peps/config.h"
#include "pi-peps/cluster-factory.h"
#include "pi-peps/ctm-cluster-basic.h"

ClusterFactory::ClusterFactory() {
	registerCluster("DEFAULT", &Cluster::create);
    registerCluster("1X1_A", &itensor::Cluster_1x1_A::create);
    registerCluster("2X2_ABBA", &itensor::Cluster_2x2_ABBA::create);
    registerCluster("2X2_ABCD", &itensor::Cluster_2x2_ABCD::create);
}

bool ClusterFactory::registerCluster(std::string const& name, 
    TCreateMethod funcCreate) {

    auto it = s_methods.find(name);
    if (it == s_methods.end()) { 
        s_methods[name] = funcCreate;
        return true;
    }
    return false;
}

std::unique_ptr<Cluster> ClusterFactory::create(nlohmann::json const& json_cluster) {
    
    std::string cluster_type = json_cluster.value("type","NOT_FOUND");

    auto it = s_methods.find(cluster_type);
    if (it != s_methods.end())
        return it->second(json_cluster); // call the "create" function

    std::string message = "[ClusterFactory] Invalid cluster type: "+ cluster_type;
    throw std::runtime_error(message);

    return nullptr;
}
