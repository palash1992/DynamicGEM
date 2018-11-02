#ifndef GRAPH_PYWRAPPER
#define GRAPH_PYWRAPPER

#include <Python.h>
#include <boost/python.hpp>
namespace py = boost::python;

#include "graph.h"
#include "exception.h"
#include "utils.h"
#include "nodemap.h"
#include "types.h"
#include <list>
#include <sstream>
#include <functional>
#include <type_traits>

// interface for a directed weighted graph
// TODO: register NodeSet etc for efficiency
// TODO: add support for undirected graph
template <typename NodeType, typename WeightType>
class GraphPyWrapper
{
public:
    typedef Graph<NodeType, WeightType, NodeMap> CGraph;

    GraphPyWrapper()
        : g(new CGraph())
    {
#ifdef DEBUG
        fprintf(stderr, "GraphPyWrapper dft ctor %p\n", this);
#endif
    }

    GraphPyWrapper(const GraphPyWrapper& g)
        : g(g.g)
    {
#ifdef DEBUG
        printf("GraphPyWrapper cpy ctor %p\n", this);
#endif
        // should never be copied with this ctor
        throw std::runtime_error("Ctor of GraphPyWrapper should never be invoked");
    }

    GraphPyWrapper(const GraphPyWrapper &&g)
        : g(g.g)
    {
#ifdef DEBUG
        printf("GraphPyWrapper mv ctor %p\n", this);
#endif
        g.g.reset();
    }

    GraphPyWrapper(CGraph *g)
        : g(g)
    {
#ifdef DEBUG
        fprintf(stderr, "GraphPyWrapper content ctor %p\n", this);
#endif
    }

    ~GraphPyWrapper()
    {
#ifdef DEBUG
        fprintf(stderr, "GraphPyWrapper dtor %p\n", this);
#endif
        //delete g;
    }

    // for boost::python constructors
    static GraphPyWrapper* makeGraph()
    {
        return new GraphPyWrapper();
    }

    void save_graph(const char *fn, bool binary = true)
    {
        g->save(fn, binary);
    }

    void load_graph(const char *fn)
    {
        g->load(fn);
    }

    std::string to_str() const
    {
        std::ostringstream oss;
        g->tostring(oss);
        return oss.str();
    }

    void parse_str(const std::string& str)
    {
        std::istringstream iss(str);
        g->fromstring(iss);
    }

    friend std::ostream& operator<<(std::ostream& os, const GraphPyWrapper& g)
    {
        g.g->tostring(os);
        return os;
    }

    void add_vertex(const typename CGraph::key_type& node)
    {
        g->newnode(node);
    }

    void add_edge(const typename CGraph::key_type& from, const typename CGraph::key_type& to, const typename CGraph::edgelist_type::mapped_type& val)
    {
        g->newedge(from, to, std::move(val));
    }

    void inc_edge(const typename CGraph::key_type& from, const typename CGraph::key_type& to, const typename CGraph::edgelist_type::mapped_type& inc)
    {
        // should we add such strict check?
        if(!g->exists(to))
            throw InvalidKeyException<typename CGraph::key_type>(__FUNCTION__, "invalid key to: ", to);
        std::pair<typename CGraph::mapped_type::iterator, bool> pr = g->newedge(from, to, inc, DUP_IGNORE);
        if(!pr.second)
            pr.first->second += inc;
    }

    size_t num_vertices() const
    {
        return g->size();
    }

    GraphPyWrapper *inverseEdge()
    {
        CGraph *sg = new CGraph(std::move(g->inverseEdge()));
        return new GraphPyWrapper(sg);
    }

    void merge(GraphPyWrapper *g, bool free_other = false)
    {
        this->g->merge_graph(*g->g, free_other);
    }

    py::list vertices()
    {
        py::list ret;
        NodeSet<typename CGraph::key_type> ns = std::move(g->nodes());
        for(auto& key : ns)
            ret.append(key);
        return ret;
    }

    py::list get(typename CGraph::key_type key) const
    {
        const typename CGraph::edgelist_type& ns = g->get_value(key);
        py::list ret;
        for(auto u : ns)
            ret.append(py::make_tuple(u.first, u.second));
        return ret;
    }

    WeightType edge(typename CGraph::key_type from, typename CGraph::key_type to) const
    {
        return g->edge_value(from, to);
    }

    bool exists(typename CGraph::key_type key) const
    {
        return g->exists(key);
    }

    bool exists(typename CGraph::key_type from, typename CGraph::key_type to) const
    {
        return g->exists(from, to);
    }

    py::list out_neighbours(typename CGraph::key_type key) const
    {
        const typename CGraph::mapped_type& v = g->get_value(key);
        if(&v == &CGraph::mapped_type::nullinst)
        {
            std::ostringstream oss;
            oss << __func__ << ": key " << key << " not found";
            throw InvalidKeyException<typename CGraph::key_type>(__FUNCTION__, oss.str(), key);
        }

        py::list ret;
        for(auto& u : v)
            ret.append(u.first);
        return ret;
    }
    
    size_t out_degree(typename CGraph::key_type key) const
    {
        return g->out_degree(key);
    }

    std::string node_type_name() const
    {
        return type2name<NodeType>::name;
    }

    std::string weight_type_name() const
    {
        return type2name<WeightType>::name;
    }

    const uintptr_t data() const
    {
        return (uintptr_t)g.get();
    }

private:
    std::shared_ptr<CGraph> g;
};

typedef GraphPyWrapper<int, float> Graph_Int32_Float;
typedef GraphPyWrapper<std::string, float> Graph_String_Float;
typedef GraphPyWrapper<int64_t, float> Graph_Int64_Float;

#endif // GRAPH_PYWRAPPER
