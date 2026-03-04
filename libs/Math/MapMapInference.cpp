#include "Common.h"
#include "MapMapInference.h"

#include <mapmap/full.h>

#include <algorithm>
#include <memory>

namespace SEACAVE {

namespace {

typedef MapMapInference::EnergyType CostType;
static constexpr NS_MAPMAP::uint_t kSimdWidth = NS_MAPMAP::sys_max_simd_width<CostType>();

typedef NS_MAPMAP::Graph<CostType> GraphType;
typedef NS_MAPMAP::LabelSet<CostType, kSimdWidth> LabelSetType;
typedef NS_MAPMAP::UnaryTable<CostType, kSimdWidth> UnaryTableType;
typedef NS_MAPMAP::PairwisePotts<CostType, kSimdWidth> PairwisePottsType;
typedef NS_MAPMAP::StopAfterIterations<CostType, kSimdWidth> TerminationType;
typedef NS_MAPMAP::mapMAP<CostType, kSimdWidth> SolverType;

inline MapMapInference::LabelID ExtractLabel(const LabelSetType& labelSet, MapMapInference::NodeID nodeID, const NS_MAPMAP::_iv_st<CostType, kSimdWidth>& offset) {
	return (MapMapInference::LabelID)labelSet.label_from_offset(nodeID, offset);
}

} // namespace

MapMapInference::MapMapInference()
	: m_smoothFunc(nullptr)
	, m_smoothPenalty(EnergyType(MaxEnergy))
	, m_smoothScale(EnergyType(1))
{
}

MapMapInference::~MapMapInference()
{
	// Clear function pointer to avoid potential issues during destruction
	m_smoothFunc = nullptr;
	// Clear all data to ensure proper cleanup order
	m_nodes.clear();
	m_edges.clear();
	m_solution.clear();
}

void MapMapInference::SetNumNodes(NodeID nNodes)
{
	m_nodes.clear();
	m_edges.clear();
	m_solution.clear();
	m_nodes.resize(nNodes);
}

void MapMapInference::ReserveNeighbors(std::size_t nEdges)
{
	m_edges.reserve(nEdges);
}

void MapMapInference::ReserveDataCosts(NodeID nodeID, std::size_t nCosts)
{
	ASSERT(nodeID < m_nodes.size());
	m_nodes[nodeID].entries.reserve(nCosts);
}

void MapMapInference::SetNeighbors(NodeID nodeID1, NodeID nodeID2, EnergyType weight)
{
	ASSERT(nodeID1 < m_nodes.size() && nodeID2 < m_nodes.size());
	m_edges.push_back({nodeID1, nodeID2, MAXF(weight, EnergyType(1e-3f))});
}

void MapMapInference::SetDataCost(LabelID label, NodeID nodeID, EnergyType cost)
{
	ASSERT(nodeID < m_nodes.size());
	m_nodes[nodeID].entries.push_back({label, cost});
}

void MapMapInference::SetSmoothCost(FncSmoothCost func)
{
	m_smoothFunc = func;
	if (m_smoothFunc) {
		const EnergyType pen = m_smoothFunc(0, 0, 0, 1);
		if (pen > EnergyType(0))
			m_smoothPenalty = pen;
	}
}

void MapMapInference::SetSmoothScale(EnergyType scale)
{
	m_smoothScale = MAXF(scale, EnergyType(1e-3f));
}

MapMapInference::LabelID MapMapInference::GetLabel(NodeID nodeID) const
{
	if (nodeID >= m_solution.size())
		return 0;
	return m_solution[nodeID];
}

MapMapInference::EnergyType MapMapInference::AssignGreedyLabels()
{
	m_solution.assign(m_nodes.size(), LabelID(0));
	EnergyType totalEnergy(0);
	for (NodeID nodeID = 0; nodeID < m_nodes.size(); ++nodeID) {
		const std::vector<DataCost>& entries = m_nodes[nodeID].entries;
		if (entries.empty())
			continue;
		LabelID bestLabel = entries.front().label;
		EnergyType bestCost = entries.front().cost;
		for (const DataCost& entry: entries) {
			if (entry.cost < bestCost) {
				bestCost = entry.cost;
				bestLabel = entry.label;
			}
		}
		m_solution[nodeID] = bestLabel;
		totalEnergy += bestCost;
	}
	return totalEnergy;
}

MapMapInference::EnergyType MapMapInference::Optimize()
{
	if (IsEmpty()) {
		m_solution.clear();
		return EnergyType(0);
	}

	for (const NodeCosts& node: m_nodes) {
		if (node.entries.empty())
			return AssignGreedyLabels();
	}

	if (m_edges.empty()) {
		return AssignGreedyLabels();
	}

	std::unique_ptr<GraphType> graph(new GraphType(m_nodes.size()));
	for (const Edge& edge: m_edges)
		graph->add_edge(edge.nodeID1, edge.nodeID2, static_cast<CostType>(edge.weight));
	graph->update_components();
	graph->sort_incidence_lists();

	std::unique_ptr<LabelSetType> labelSet(new LabelSetType(m_nodes.size(), false));
	std::vector<UnaryTableType> unaries;
	unaries.reserve(m_nodes.size());
	std::vector<NS_MAPMAP::_iv_st<CostType, kSimdWidth>> labels;
	std::vector<NS_MAPMAP::_s_t<CostType, kSimdWidth>> costs;

	for (NodeID nodeID = 0; nodeID < m_nodes.size(); ++nodeID) {
		std::vector<DataCost>& entries = m_nodes[nodeID].entries;
		std::sort(entries.begin(), entries.end(), [](const DataCost& a, const DataCost& b) {
			return a.label < b.label;
		});
		std::vector<DataCost> uniqueEntries;
		uniqueEntries.reserve(entries.size());
		for (const DataCost& entry: entries) {
			if (!uniqueEntries.empty() && uniqueEntries.back().label == entry.label) {
				if (entry.cost < uniqueEntries.back().cost)
					uniqueEntries.back().cost = entry.cost;
			} else {
				uniqueEntries.push_back(entry);
			}
		}
		entries.swap(uniqueEntries);

		labels.resize(entries.size());
		costs.resize(entries.size());
		for (size_t i = 0; i < entries.size(); ++i) {
			labels[i] = static_cast<NS_MAPMAP::_iv_st<CostType, kSimdWidth>>(entries[i].label);
			costs[i] = static_cast<NS_MAPMAP::_s_t<CostType, kSimdWidth>>(entries[i].cost);
		}
		labelSet->set_label_set_for_node(nodeID, labels);
		unaries.emplace_back(nodeID, labelSet.get());
		unaries.back().set_costs(costs);
	}

	std::unique_ptr<PairwisePottsType> pairwise(new PairwisePottsType(m_smoothPenalty * m_smoothScale));
	std::unique_ptr<TerminationType> termination(new TerminationType(20, true, true, true));
	SolverType solver;
	solver.set_graph(graph.get());
	solver.set_label_set(labelSet.get());
	for (NodeID nodeID = 0; nodeID < m_nodes.size(); ++nodeID)
		solver.set_unary(nodeID, &unaries[nodeID]);
	solver.set_pairwise(pairwise.get());
	solver.set_termination_criterion(termination.get());
	// Set empty callback to disable mapMAP logging output
	solver.set_logging_callback([](NS_MAPMAP::luint_t, NS_MAPMAP::_s_t<CostType, kSimdWidth>) {});

	std::vector<NS_MAPMAP::_iv_st<CostType, kSimdWidth>> solutionOffsets;
	CostType finalEnergy = CostType(0);
	try {
		finalEnergy = solver.optimize(solutionOffsets);
	} catch (const std::exception& e) {
		DEBUG("mapMAP inference failed: %s", e.what());
		return AssignGreedyLabels();
	}

	if (solutionOffsets.size() != m_nodes.size())
		return AssignGreedyLabels();

	m_solution.resize(m_nodes.size(), LabelID(0));
	for (NodeID nodeID = 0; nodeID < m_nodes.size(); ++nodeID) {
		const size_t labelSetSize = m_nodes[nodeID].entries.size();
		NS_MAPMAP::_iv_st<CostType, kSimdWidth> offset = solutionOffsets[nodeID];
		// Clamp offset to valid range: mapMAP returns per-node label set offset;
		// out-of-range can cause UB in label_from_offset and break openMVS texturing.
		if (labelSetSize == 0)
			m_solution[nodeID] = 0;
		else {
			if (static_cast<size_t>(offset) >= labelSetSize)
				offset = 0;
			m_solution[nodeID] = ExtractLabel(*labelSet, nodeID, offset);
		}
	}

	return static_cast<EnergyType>(finalEnergy);
}

} // namespace SEACAVE
