/**
  Code manipulating graph structure,
  actually we can implement this basing on boosting graph library...
*/

#ifndef GRAPH_H_INCLUDED
#define GRAPH_H_INCLUDED

#include <map>
#include <list>
#include <utility>
#include "nodemap.h"
#include "exception.h"
#include "defs.h"

#ifdef DEBUG
#include <iostream>
#endif

#define NOTIMPL(func) throw NotImplementedException("function " ## func ## " not implemented", func)

//template <typename node_type, typename rec_type,
//          template <typename, typename> typename impl_type>
//class EdgeList : public impl_type<node_type, rec_type>
//{
//public:
//    node_type rec2key(const rec_type& rec) { NOTIMPL(__FUNCTION__); }
//};

// when the concept of edge is introduced to NodeMap etc.
template <typename node_t, typename weight_t, template<typename, typename> class EdgeMapImpl>
class EdgeMap : public EdgeMapImpl<node_t, weight_t>
{
public:
    typedef node_t node_type;
    typedef weight_t weight_type;
    typedef EdgeMapImpl<node_t, weight_t> super;
    typedef typename super::value_type rec_type;

    static_assert(std::is_same<rec_type, std::pair<const node_type, weight_type> >::value,
                  "Invalid inner types for EdgeMap");

    EdgeMap() {}
    EdgeMap(EdgeMapImpl<node_t, weight_t>&& em)
        : super(std::move(em)) {}
    EdgeMap(EdgeMap&& em)
        : super(std::move(em)) {}
    EdgeMap& operator=(EdgeMap&& em)
    {
        super::operator=(std::move(em));
        return *this;
    }

    static node_type rec2key(const rec_type& rec) { return rec.first; }
    static rec_type invrec(const node_type& new_key, const rec_type& rec)
    {
        return std::make_pair(new_key, rec.second);
    }
    std::pair<typename super::iterator, bool> newrec(rec_type&& rec, DUPMODE dupmode = DUP_OVERWRITE)
    {
        return super::newnode(std::move(rec), dupmode);
    }
    std::pair<typename super::iterator, bool> newedge(const node_type& key, weight_type&& val, DUPMODE dupmode = DUP_OVERWRITE)
    {
        return super::newnode(key, std::move(val), dupmode);
    }
    bool exists(const node_type& key) const
    {
        return super::exists(key);
    }

    static const EdgeMap nullinst;

private:
    EdgeMap(const EdgeMap &em) = delete;
    EdgeMap& operator=(const EdgeMap& em) = delete;
};

template <typename node_t, typename weight_t, template<typename, typename> class EdgeMapImpl>
const EdgeMap<node_t, weight_t, EdgeMapImpl> EdgeMap<node_t, weight_t, EdgeMapImpl>::nullinst;

template <typename G>
class EdgeView
{
public:
    EdgeView(G *g) : g(g) {}

    class iterator
    {
    public:
        typedef typename G::iterator outeritr;
        typedef typename G::edgelist_type::iterator inneritr;

        std::pair<typename G::node_type, typename G::edgelist_type::value_type> operator*()
        {
            return std::make_pair(outer->first, *inner);
        }

        iterator& operator++()
        {
            inc();
            skip_invalid();
            return *this;
        }

        iterator operator++(int)
        {
            iterator ret = *this;
            operator++();
            return ret;
        }

        bool operator==(const iterator& itr) const
        {
            return (outer == outerend && itr.outer == outerend) ||
                    (inner == itr.inner && outer == itr.outer);
        }

        bool operator!=(const iterator& itr) const
        {
            return !operator==(itr);
        }

        friend class EdgeView;

    private:
        iterator(outeritr outer, outeritr outerend, inneritr inner)
            : outer(outer), outerend(outerend), inner(inner) {}

        iterator& inc()
        {
            if(outer == outerend) return *this;

            if(inner == outer->second.end())
            {
                ++outer;
                if(outer != outerend)
                    inner = outer->second.begin();
                else
                    inner = inneritr();
            }
            else
                ++inner;
            return *this;
        }

        iterator& skip_invalid()
        {
            while(outer != outerend && inner == outer->second.end())
                inc();
            return *this;
        }

        outeritr outer;
        outeritr outerend;
        inneritr inner;
    };

    iterator begin()
    {
        typename iterator::outeritr outer = g->begin();
        typename iterator::inneritr inner;
        if(outer != g->end())
        {
            inner = outer->second.begin();
        }
        return iterator(outer, g->end(), inner).skip_invalid();
    }
    iterator end()
    {
        typename iterator::outeritr outer = g->end();
        typename iterator::inneritr inner;
        return iterator(outer, outer, inner);
    }
    // TODO add const version here

private:
    G *g;
};

// TODO: add directed label to graph
template <typename node_t, typename weight_t, template <class, class> class EdgeMapImpl>
class Graph : public NodeMap<node_t, EdgeMap<node_t, weight_t, EdgeMapImpl> >
{
public:
//    static_assert(std::is_same<typename EdgeMap<mapped_t>::node_type, key_t>::value,
//                  "From and To nodes are not the same type");
    typedef EdgeMap<node_t, weight_t, EdgeMapImpl> edgelist_type;
    typedef NodeMap<node_t, edgelist_type> super;
    typedef node_t node_type;
    typedef weight_t weight_type;

//    enum
//    {
//        FMT_RAW = 0,
//        FMT_ADVANCED,  // with number of edges specified in each line
//    };

    Graph() {}
    Graph(Graph&& g) = default;
    Graph& operator=(Graph&& g) = default;
    //Graph(const char *fn, int fmt) { load(fn, fmt); }
    /** filter_out:
            true - ignore all those appeared in filter
            false - reserve only those appeared in filter
    */
    //bool load(const char *fn, int fmt, bool overwrite = false, const NodeSet& filter = NodeSet::empty_ns, bool filter_out = false);
    //bool load(FileNameInfo info, int fmt, bool overwrite = false, const NodeSet& filter = NodeSet::empty_ns, bool filter_out = false);
    //bool save(const char *fn, int fmt);
    //bool save(const FileNameInfo& info, int fmt);

    const edgelist_type& get_value(const node_type& key) const
    {
        return static_cast<const edgelist_type&>(super::get_value(key));
//        return super::get(key);
    }
    edgelist_type& get(const node_type& key)
    {
        return static_cast<edgelist_type&>(super::get(key));
    }
    
    size_t out_degree(const node_type& key) const
    {
        return get_value(key).size();
    }

    const weight_type& edge_value(const node_type& key1, const node_type& key2, const weight_type& dft = Defaults<weight_type>::get()) const
    {
        return get_value(key1).get_value(key2, dft);
    }

    weight_type& edge(const node_type& key1, const node_type& key2,
                      bool create_default = false, weight_type&& dft = weight_type())
    {
        return get(key1).get(key2, create_default, std::move(dft));
    }

    bool exists(const node_type& from, const node_type& to) const
    {
        return get_value(from).exists(to);
    }

    bool exists(const node_type& key) const { return super::exists(key); }

    // implement edge iterator as return type?
    std::pair<typename edgelist_type::iterator, bool> newedge(const node_type& from, const node_type& to, typename edgelist_type::weight_type&& val, DUPMODE dupmode = DUP_OVERWRITE)
    {
        edgelist_type& el = super::get(from);
        if(&el == &edgelist_type::nullinst)
            throw InvalidKeyException<node_type>(__FUNCTION__, "Invalid key from", from);
        return el.newedge(to, std::move(val), dupmode);
    }
    std::pair<typename edgelist_type::iterator, bool> newedge(const node_type& from, const node_type& to, const typename edgelist_type::weight_type& val, DUPMODE dupmode = DUP_OVERWRITE)
    {
        edgelist_type& el = super::get(from);
        if(&el == &edgelist_type::nullinst)
            throw InvalidKeyException<node_type>(__FUNCTION__, "Invalid key from", from);
        typename edgelist_type::weight_type cpy = val;
        return el.newedge(to, std::move(cpy), dupmode);
    }

    Graph inverseEdge() const
    {
        Graph ret;
        for(const auto& itr : *this)
        {
            if(ret.data.find(itr.first) == ret.data.end())
                ret.data.emplace(itr.first, std::move(typename Graph::edgelist_type()));

            for(const auto& rec : itr.second)
            {
                typename Graph::node_type key = Graph::edgelist_type::rec2key(rec);
                std::pair<typename Graph::iterator, bool> res = ret.newnode(key);
                res.first->second.newrec(std::move(Graph::edgelist_type::invrec(itr.first, rec)));
            }
        }
        return std::move(ret);
    }

    void merge_graph(Graph& g, bool free_other = true)
    {
        if(this == &g) free_other = false;

        Graph *gtgt = this, *gsrc = &g;
        if(free_other && gtgt->size() < gsrc->size())
            std::swap(gsrc, gtgt);

        for(auto srcitr = gsrc->begin(); srcitr != gsrc->end(); srcitr++)
        {
            auto tgtitr = gtgt->find(srcitr->first);
            if(tgtitr == gtgt->end())
            {
                if(free_other)
                    gtgt->newnode(srcitr->first, std::move(srcitr->second), DUP_WARN);
                else
                    gtgt->newnode(srcitr->first, SafeCopy<edgelist_type>::copy(srcitr->second), DUP_WARN);
            }
            else
            {
                for(auto insrcitr = srcitr->second.begin(); insrcitr != srcitr->second.end(); insrcitr++)
                {
                    tgtitr->second.get(insrcitr->first, true, weight_type()) += insrcitr->second;
                }
            }
        }

        if(free_other)
        {
            if(gtgt != this)
            {
                *this = std::move(*gtgt);
                gtgt->clear();
            }
            else
            {
                gsrc->clear();
            }
        }
    }

    // TODO: thie should be a const function, however, we haven't implemented
    // const edge view yet
    EdgeView<Graph<node_type, weight_type, EdgeMapImpl> > edges()
    {
        return EdgeView<Graph<node_type, weight_type, EdgeMapImpl> >(this);
    }

private:
    Graph(Graph& g) {}
    Graph& operator=(const Graph& g) { return *this; }
};

#endif // GRAPH_H_INCLUDED
