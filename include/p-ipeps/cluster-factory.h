#ifndef __CLUSTER_FACTORY_
#define __CLUSTER_FACTORY_

#include "json.hpp"
#include "p-ipeps/ctm-cluster.h"

class ClusterFactory {
	public:
		using TCreateMethod = std::function<std::unique_ptr<Cluster>(nlohmann::json const& json_cluster)>;

		ClusterFactory();
		virtual ~ClusterFactory() = default;

		bool registerCluster(std::string const& name, TCreateMethod funcCreate);

		std::unique_ptr<Cluster> create(nlohmann::json const& json_cluster);

	private:
		std::map<std::string, TCreateMethod> s_methods;
};

#endif