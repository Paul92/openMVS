//////////////////////////////////////////////////////////////////////
// MapMapInference.h
//
// Lightweight wrapper around the mapMAP solver for texture inference.
//
// Copyright 2025
//
#ifndef __MATH_MAPMAPINFERENCE_H__
#define __MATH_MAPMAPINFERENCE_H__

#include <cstddef>
#include <utility>
#include <vector>

namespace SEACAVE {

class MATH_API MapMapInference
{
public:
	typedef unsigned NodeID;
	typedef unsigned LabelID;
	typedef float EnergyType;
	typedef EnergyType (STCALL *FncSmoothCost)(NodeID, NodeID, LabelID, LabelID);

	enum { MaxEnergy = 100 };

	MapMapInference();
	~MapMapInference();

	void SetNumNodes(NodeID nNodes);
	inline NodeID GetNumNodes() const { return (NodeID)m_nodes.size(); }
	inline bool IsEmpty() const { return m_nodes.empty(); }

	void ReserveNeighbors(std::size_t nEdges);
	void ReserveDataCosts(NodeID nodeID, std::size_t nCosts);
	void SetNeighbors(NodeID nodeID1, NodeID nodeID2, EnergyType weight = EnergyType(1));
	void SetDataCost(LabelID label, NodeID nodeID, EnergyType cost);
	void SetSmoothCost(FncSmoothCost func);
	void SetSmoothScale(EnergyType scale);

	EnergyType Optimize();
	LabelID GetLabel(NodeID nodeID) const;

private:
	struct DataCost {
		LabelID label;
		EnergyType cost;
	};
	struct NodeCosts {
		std::vector<DataCost> entries;
	};
	struct Edge {
		NodeID nodeID1;
		NodeID nodeID2;
		EnergyType weight;
	};

	EnergyType AssignGreedyLabels();

private:
	std::vector<NodeCosts> m_nodes;
	std::vector<Edge> m_edges;
	std::vector<LabelID> m_solution;
	FncSmoothCost m_smoothFunc;
	EnergyType m_smoothPenalty;
	EnergyType m_smoothScale;
};

} // namespace SEACAVE

#endif // __MATH_MAPMAPINFERENCE_H__
