/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */
#include <mapmap/header/tree_optimizer.h>

#include <cstddef>
#include <iostream>

NS_MAPMAP_BEGIN

template<typename COSTTYPE, uint_t SIMDWIDTH>
TreeOptimizer<COSTTYPE, SIMDWIDTH>::
TreeOptimizer()
: m_has_tree(false),
  m_has_label_set(false),
  m_has_costs(false),
  m_uses_dependencies(false)
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
TreeOptimizer<COSTTYPE, SIMDWIDTH>::
~TreeOptimizer()
{

}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
void
TreeOptimizer<COSTTYPE, SIMDWIDTH>::
set_graph(
    const Graph<COSTTYPE> * graph)
{
    m_graph = graph;
    m_has_graph = true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
void
TreeOptimizer<COSTTYPE, SIMDWIDTH>::
set_tree(
    const Tree<COSTTYPE> * tree)
{
    m_tree = tree;
    m_has_tree = true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
void
TreeOptimizer<COSTTYPE, SIMDWIDTH>::
set_label_set(
    const LabelSet<COSTTYPE, SIMDWIDTH> * label_set)
{
    m_label_set = label_set;
    m_has_label_set = true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
void
TreeOptimizer<COSTTYPE, SIMDWIDTH>::
set_costs(
    const CostBundle<COSTTYPE, SIMDWIDTH> * cbundle)
{
    m_cbundle = cbundle;
    m_has_costs = true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
void
TreeOptimizer<COSTTYPE, SIMDWIDTH>::
use_dependencies(
    const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& current_solution)
{
    m_current_assignment = current_solution;
    m_uses_dependencies = true;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
_s_t<COSTTYPE, SIMDWIDTH>
TreeOptimizer<COSTTYPE, SIMDWIDTH>::
objective(
    const std::vector<_iv_st<COSTTYPE, SIMDWIDTH>>& solution)
{
    _s_t<COSTTYPE, SIMDWIDTH> obj = (COSTTYPE) 0;
    const luint_t num_nodes = m_graph->num_nodes();
    const luint_t num_edges = m_graph->edges().size();

    /* unary costs - OpenMP parallel reduction */
    _s_t<COSTTYPE, SIMDWIDTH> unary_sum = (COSTTYPE) 0;
    #pragma omp parallel for reduction(+:unary_sum) schedule(static)
    for(std::ptrdiff_t n_p = 0; n_p < static_cast<std::ptrdiff_t>(num_nodes); ++n_p)
    {
        const luint_t n = static_cast<luint_t>(n_p);
        _s_t<COSTTYPE, SIMDWIDTH> tmp[SIMDWIDTH];
        const UnaryCosts<COSTTYPE, SIMDWIDTH> * ucosts =
            m_cbundle->get_unary_costs(n);

        const uint_t n_l = solution[n];

        _v_t<COSTTYPE, SIMDWIDTH> u_costs;
        if(ucosts->supports_enumerable_costs())
        {
            u_costs = ucosts->get_unary_costs_enum_offset(n_l);
        }
        else
        {
            const _iv_st<COSTTYPE, SIMDWIDTH> l =
                m_label_set->label_from_offset(n, n_l);
            u_costs = ucosts->
                get_unary_costs(iv_init<COSTTYPE, SIMDWIDTH>(l));
        }

        v_store<COSTTYPE, SIMDWIDTH>(u_costs, tmp);
        unary_sum += tmp[0];
    }
    obj += unary_sum;

    /* pairwise costs - OpenMP parallel reduction */
    _s_t<COSTTYPE, SIMDWIDTH> pairwise_sum = (COSTTYPE) 0;
    #pragma omp parallel for reduction(+:pairwise_sum) schedule(static)
    for(std::ptrdiff_t e_p = 0; e_p < static_cast<std::ptrdiff_t>(num_edges); ++e_p)
    {
        const luint_t e = static_cast<luint_t>(e_p);
        _s_t<COSTTYPE, SIMDWIDTH> tmp[SIMDWIDTH];
        const PairwiseCosts<COSTTYPE, SIMDWIDTH> * pcosts =
            m_cbundle->get_pairwise_costs(e);

        const luint_t n_a = m_graph->edges()[e].node_a;
        const luint_t n_b = m_graph->edges()[e].node_b;
        const _s_t<COSTTYPE, SIMDWIDTH> e_weight =
            m_graph->edges()[e].weight;

        const _iv_st<COSTTYPE, SIMDWIDTH> n_a_l_i = solution[n_a];
        const _iv_st<COSTTYPE, SIMDWIDTH> n_b_l_i = solution[n_b];

        const _iv_st<COSTTYPE, SIMDWIDTH> n_a_l =
            m_label_set->label_from_offset(n_a, n_a_l_i);
        const _iv_st<COSTTYPE, SIMDWIDTH> n_b_l =
            m_label_set->label_from_offset(n_b, n_b_l_i);

        _v_t<COSTTYPE, SIMDWIDTH> p_costs = pcosts->get_pairwise_costs(
            iv_init<COSTTYPE, SIMDWIDTH>(n_a_l),
            iv_init<COSTTYPE, SIMDWIDTH>(n_b_l));

        p_costs = v_mult<COSTTYPE, SIMDWIDTH>(p_costs,
            v_init<COSTTYPE, SIMDWIDTH>(e_weight));

        v_store<COSTTYPE, SIMDWIDTH>(p_costs, tmp);
        pairwise_sum += tmp[0];
    }
    obj += pairwise_sum;

    return obj;
}

/* ************************************************************************** */

template<typename COSTTYPE, uint_t SIMDWIDTH>
bool
TreeOptimizer<COSTTYPE, SIMDWIDTH>::
data_complete()
{
    return (m_has_graph && m_has_tree && m_has_label_set && m_has_costs);
}

NS_MAPMAP_END
