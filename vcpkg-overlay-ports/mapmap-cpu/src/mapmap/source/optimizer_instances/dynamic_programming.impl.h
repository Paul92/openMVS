/**
 * Copyright (C) 2016, Daniel Thuerck, Nick Heppert
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include <algorithm>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <iostream>
#include <numeric>
#include <atomic>

#include <mapmap/header/optimizer_instances/dynamic_programming.h>
#include <mapmap/header/optimizer_instances/dp_node.h>
#include <mapmap/header/optimizer_instances/dp_node_solver_factory.h>

NS_MAPMAP_BEGIN

/**
 * *****************************************************************************
 * *********************** DynamicProgrammingTableEntry ************************
 * *****************************************************************************
 */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
DynamicProgrammingTableEntry<COSTTYPE, SIMDWIDTH>::
DynamicProgrammingTableEntry(
    DPNode<COSTTYPE, SIMDWIDTH> * node,
    tbb_allocator_ptr<_s_t<COSTTYPE, SIMDWIDTH>> allocator)
: m_node(node),
  m_allocator(allocator)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
DynamicProgrammingTableEntry<COSTTYPE, SIMDWIDTH>::
~DynamicProgrammingTableEntry()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
const luint_t&
DynamicProgrammingTableEntry<COSTTYPE, SIMDWIDTH>::
node_id()
{
    return m_node->c_node.node_id;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
const luint_t&
DynamicProgrammingTableEntry<COSTTYPE, SIMDWIDTH>::
parent_id()
{
    return m_node->c_node.parent_id;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
DynamicProgrammingTableEntry<COSTTYPE, SIMDWIDTH>::
optimize_entry()
{
    /* first: retrieve solver for node - w.r.t to the cost function */
    DPNodeSolver_ptr<COSTTYPE, SIMDWIDTH> solver =
        DPNodeSolverFactory<COSTTYPE, SIMDWIDTH>::
        get_solver(m_node);

    /* if necessary, allocate scratch space for node */
    const luint_t scratch_bytes = solver->scratch_bytes_needed();
    const luint_t padded_scratch_bytes = DIV_UP(scratch_bytes,
        sizeof(_s_t<COSTTYPE, SIMDWIDTH>)) *
        sizeof(_s_t<COSTTYPE, SIMDWIDTH>);
    if(scratch_bytes > 0)
        m_node->c_scratch = m_allocator->allocate(padded_scratch_bytes);

    /* now optimize! */
    solver->optimize_node();

    /* free scratch space */
    if(scratch_bytes > 0)
        m_allocator->deallocate(m_node->c_scratch, padded_scratch_bytes);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
const _s_t<COSTTYPE, SIMDWIDTH> *
DynamicProgrammingTableEntry<COSTTYPE, SIMDWIDTH>::
optimal_value()
{
    return m_node->c_opt_values;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
const _iv_st<COSTTYPE, SIMDWIDTH> *
DynamicProgrammingTableEntry<COSTTYPE, SIMDWIDTH>::
optimal_labels()
{
    return m_node->c_opt_labels;
}

/**
 * *****************************************************************************
 * *********************** CombinatorialDynamicProgramming *********************
 * *****************************************************************************
 */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH>::
CombinatorialDynamicProgramming()
: m_level(0),
  m_level_size(0),
  m_value_allocator(new std::allocator<_s_t<COSTTYPE, SIMDWIDTH>>())
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH>::
~CombinatorialDynamicProgramming()
{
    /* free all node optimization spaces */
#if defined(BUILD_MEMORY_SAVE)
    for(const luint_t r : m_root_ids)
        m_value_allocator->deallocate(m_opt_value_nodes[r],
            m_opt_value_sizes[r]);
#endif
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_s_t<COSTTYPE, SIMDWIDTH>
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH>::
optimize(
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& solution)
{
    if(!this->data_complete())
        throw std::domain_error("Data for optimization problem "
            "incomplete!");

    /* search for leafs as initial nodes to optimize */
    this->discover_leaves();

    /* allocate required memory */
    this->allocate_memory();

    /* bottom-up pass: create table */
    bottom_up_opt();

    /* top-down pass: assign labels for minimum solution */
    top_down_opt(solution);

    return this->objective(solution);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH>::
discover_leaves()
{
    const luint_t num_nodes = this->m_tree->num_graph_nodes();

    /* find leaves: nodes without children that are in the tree */
    m_leaf_ids.clear();
    for(luint_t i = 0; i < num_nodes; ++i)
    {
        if(this->m_tree->node(i).is_in_tree &&
           this->m_tree->node(i).degree == 0)
            m_leaf_ids.push_back(i);
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH>::
allocate_memory()
{
    const luint_t num_nodes = this->m_tree->num_graph_nodes();

    /* create pointer table for nodes' value tables (and their sizes) */
    m_opt_value_nodes = std::vector<_s_t<COSTTYPE, SIMDWIDTH>*>(num_nodes);
    m_opt_value_sizes = std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>(num_nodes);

    /* compute label set sizes and offsets */
    std::vector<luint_t> lbl_set_sizes(num_nodes, (luint_t) 0);
    std::vector<luint_t> lbl_set_offsets(num_nodes, (luint_t) 0);

    #pragma omp parallel for schedule(static)
    for(std::ptrdiff_t i_p = 0; i_p < static_cast<std::ptrdiff_t>(num_nodes); ++i_p)
    {
        const luint_t i = static_cast<luint_t>(i_p);
        if(this->m_tree->node(i).parent_id != invalid_luint_t)
            lbl_set_sizes[i] = SIMDWIDTH * DIV_UP(
                this->m_label_set->label_set_size(
                this->m_tree->node(i).parent_id), SIMDWIDTH);
    }

    /* exclusive prefix sum for offsets */
    luint_t running_sum = 0;
    for(luint_t i = 0; i < num_nodes; ++i)
    {
        lbl_set_offsets[i] = running_sum;
        running_sum += lbl_set_sizes[i];
    }
    const luint_t total_mem_req = running_sum;

    /* index table for all nodes */
    m_opt_labels = std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>(total_mem_req,
        (uint_t) invalid_luint_t);

#ifndef BUILD_MEMORY_SAVE
    /* value table for all nodes */
    m_opt_values = std::vector<_s_t<COSTTYPE, SIMDWIDTH>>(total_mem_req);
#endif

    /* create pointer tables for indices for all nodes */
    m_opt_label_nodes = std::vector<_iv_st<COSTTYPE, SIMDWIDTH>*>(num_nodes);

    /* save offsets into tables for all nodes */
    #pragma omp parallel for schedule(static)
    for(std::ptrdiff_t i_p = 0; i_p < static_cast<std::ptrdiff_t>(num_nodes); ++i_p)
    {
        const luint_t i = static_cast<luint_t>(i_p);
        if(this->m_tree->node(i).parent_id != invalid_luint_t)
        {
            m_opt_label_nodes[i] = &m_opt_labels[0] + lbl_set_offsets[i];
#ifndef BUILD_MEMORY_SAVE
            m_opt_value_nodes[i] = &m_opt_values[0] + lbl_set_offsets[i];
#endif
        }
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH>::
node_memory_allocate(
    const luint_t node_id)
{
    /* determine number of needed storage spaces */
    const _iv_st<COSTTYPE, SIMDWIDTH> entry_size = SIMDWIDTH *
        DIV_UP(this->m_label_set->label_set_size(
        this->m_tree->node(node_id).parent_id), SIMDWIDTH);

    /* allocate memory */
    m_opt_value_nodes[node_id] = m_value_allocator->allocate(entry_size);
    m_opt_value_sizes[node_id] = entry_size;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH>::
node_memory_clean_children(
    const luint_t node_id)
{
    /* deallocate children's value tables */
    const TreeNode<COSTTYPE> tree_n = this->m_tree->node(node_id);
    const luint_t degree = tree_n.degree;
    const luint_t * children = tree_n.children_ids;

    for(luint_t d = 0; d < degree; ++d)
    {
        const luint_t c = children[d];
        m_value_allocator->deallocate(m_opt_value_nodes[c],
            m_opt_value_sizes[c]);
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH>::
bottom_up_opt()
{
    const luint_t num_nodes = this->m_tree->num_graph_nodes();

    /* count unprocessed children per node */
    std::vector<std::atomic<luint_t>> unproc_children(num_nodes);
    for(luint_t i = 0; i < num_nodes; ++i)
        unproc_children[i].store(this->m_tree->node(i).degree);

    /*
     * Level-wise parallel bottom-up traversal using OpenMP.
     * Process all ready nodes at the current level in parallel,
     * then collect the next level's ready nodes.
     */
    std::vector<luint_t> current_level(m_leaf_ids.begin(), m_leaf_ids.end());
    std::vector<luint_t> next_level;

    while(!current_level.empty())
    {
        next_level.clear();

        /* Each thread gets its own allocator for scratch memory */
        #pragma omp parallel
        {
            /* thread-local allocator avoids contention on std::allocator */
            std::allocator<_s_t<COSTTYPE, SIMDWIDTH>> local_alloc;
            auto local_alloc_ptr = tbb_allocator_ptr<_s_t<COSTTYPE, SIMDWIDTH>>(
                &local_alloc, [](std::allocator<_s_t<COSTTYPE, SIMDWIDTH>>*){});

            #pragma omp for schedule(dynamic, 64)
            for(std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(current_level.size()); ++idx)
            {
                const luint_t n = current_level[static_cast<size_t>(idx)];

#if defined(BUILD_MEMORY_SAVE)
                node_memory_allocate(n);
#endif

                /* create a bundle with all necessary information for DP */
                DPNode<COSTTYPE, SIMDWIDTH> dpb;
                dpb.c_node = this->m_tree->node(n);
                dpb.c_graph = this->m_graph;
                dpb.c_labels = this->m_label_set;
                dpb.c_unary = this->m_cbundle->get_unary_costs(n);
                dpb.c_pairwise = this->m_cbundle->get_pairwise_costs(
                    dpb.c_node.to_parent_edge_id);
                dpb.c_child_values = &m_opt_value_nodes;
                dpb.c_child_labels = &m_opt_label_nodes;
                dpb.c_assignment = &this->m_current_assignment;
                dpb.respect_dependencies = this->m_uses_dependencies;
                dpb.c_opt_values = m_opt_value_nodes[n];
                dpb.c_opt_labels = m_opt_label_nodes[n];

                /* retrieve cost functions for dependencies */
                dpb.c_dep_costs.resize(dpb.c_node.dependency_degree);
                for(luint_t i = 0; i < dpb.c_node.dependency_degree; ++i)
                    dpb.c_dep_costs[i] = this->m_cbundle->get_pairwise_costs(
                        dpb.c_node.dependency_edge_ids[i]);

                /* create one table entry per node */
                DynamicProgrammingTableEntry<COSTTYPE, SIMDWIDTH> dpe(&dpb,
                    local_alloc_ptr);

                /* delegate optimization! */
                dpe.optimize_entry();

#if defined(BUILD_MEMORY_SAVE)
                node_memory_clean_children(n);
#endif
            }
        } /* end omp parallel */

        /* Collect next level: check which parents are now ready (sequential) */
        for(luint_t idx = 0; idx < current_level.size(); ++idx)
        {
            const luint_t n = current_level[idx];
            const luint_t parent_id = this->m_tree->node(n).parent_id;

            if(parent_id == n)
            {
                /* root node */
                m_root_ids.push_back(n);
            }
            else
            {
                luint_t prev = unproc_children[parent_id].fetch_sub(1);
                if(prev == 1)
                {
                    /* all children done, parent is ready */
                    next_level.push_back(parent_id);
                }
            }
        }

        current_level.swap(next_level);
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH>::
top_down_opt(
    std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& solution)
{
    /* Collect root children as first level */
    std::vector<luint_t> current_level;

    for(luint_t i = 0; i < m_root_ids.size(); ++i)
    {
        const luint_t root = m_root_ids[i];
        const uint_t r_label_set_size = this->m_label_set->
            label_set_size(root);

        const _s_t<COSTTYPE, SIMDWIDTH> * my_costs =
            m_opt_value_nodes[root];
        const _iv_st<COSTTYPE, SIMDWIDTH> * my_labels =
            m_opt_label_nodes[root];

        _s_t<COSTTYPE, SIMDWIDTH> min_costs = my_costs[0];
        _iv_st<COSTTYPE, SIMDWIDTH> min_label = my_labels[0];
        for(uint_t l_i = 1; l_i < r_label_set_size; ++l_i)
        {
            if(my_costs[l_i] < min_costs)
            {
                min_costs = my_costs[l_i];
                min_label = my_labels[l_i];
            }
        }

        solution[root] = min_label;

        /* add children to first level */
        for(luint_t c = 0; c < this->m_tree->node(root).degree; ++c)
            current_level.push_back(this->m_tree->node(root).children_ids[c]);
    }

    /* Level-wise parallel top-down traversal */
    std::vector<luint_t> next_level;

    while(!current_level.empty())
    {
        next_level.clear();

        /* assign labels in parallel - each node reads only its parent's label */
        #pragma omp parallel for schedule(static)
        for(std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(current_level.size()); ++idx)
        {
            const luint_t n = current_level[static_cast<size_t>(idx)];
            const TreeNode<COSTTYPE>& node = this->m_tree->node(n);

            const _iv_st<COSTTYPE, SIMDWIDTH> p_label =
                solution[node.parent_id];

            const _iv_st<COSTTYPE, SIMDWIDTH> * my_labels =
                m_opt_label_nodes[n];
            solution[n] = my_labels[p_label];
        }

        /* collect next level's children (sequential) */
        for(luint_t idx = 0; idx < current_level.size(); ++idx)
        {
            const luint_t n = current_level[idx];
            const TreeNode<COSTTYPE>& node = this->m_tree->node(n);

            for(luint_t c = 0; c < node.degree; ++c)
                next_level.push_back(node.children_ids[c]);
        }

        current_level.swap(next_level);
    }
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
_iv_st<COSTTYPE, SIMDWIDTH>*
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH>::
index_ptr()
{
    return &m_opt_labels[0];
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
std::vector<luint_t> *
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH>::
current_level_queue_ptr()
{
    /* _a for even levels, _b else */
    return ((m_level & 0x1) ? &m_queue_b : &m_queue_a);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
std::vector<luint_t> *
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH>::
next_level_queue_ptr()
{
    /* _b for even levels, _a else */
    return ((m_level & 0x1) ? &m_queue_a : &m_queue_b);
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
FORCEINLINE
void
CombinatorialDynamicProgramming<COSTTYPE, SIMDWIDTH>::
next_level()
{
    ++m_level;
}

NS_MAPMAP_END
