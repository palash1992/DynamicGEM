#ifndef NodeMap_H_INCLUDED
#define NodeMap_H_INCLUDED

#include <set>
#include <utility>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <fstream>
#include <limits>
#include <cmath>
#include <cassert>
#include <cstddef>
#include "exception.h"
#include "defs.h"

template <typename key_t, typename mapped_t>
class NodeMap;

// TODO: add serialization code, see nodemap.h
template <typename key_t>
class NodeMap<key_t, nullptr_t>
{
public:
    typedef key_t key_type;
    typedef nullptr_t mapped_type;
    typedef std::set<key_type> data_type;
    typedef typename data_type::iterator iterator;
    typedef typename data_type::const_iterator const_iterator;
    typedef typename data_type::size_type size_type;

    enum FORMAT
    {
        FMT_DICT = 0,
        FMT_FREQ = 1,
        FMT_LIST = 2,
    };

//    enum DUPMODE
//    {
//        DUP_IGNORE = 0,  // do not insert when duplicated
//        DUP_OVERWRITE,   // overwrite duplicated
//        DUP_WARN,        // throw if duplicated
//    };

    NodeMap() {}
    NodeMap(const data_type& data) : data(data) {}
    NodeMap(data_type&& data) : data(data) {}
    NodeMap(NodeMap&& ns) : data(move(ns.data)) {}
    NodeMap& operator=(NodeMap&& ns)
    {
        if(&ns != this) data = std::move(ns.data);
        return *this;
    }
    
    bool load(const char *fn, FORMAT fmt = FMT_LIST, DUPMODE dupmode = DUP_WARN)
    {
        std::string buf;
        std::ifstream fp(fn, std::ios::in);
        if(!fp.is_open() || fp.bad())
        {
            throw IOException(__FUNCTION__, ": Cannot load users file");
        }
        while(!std::getline(fp, buf).eof())
        {
            if(buf.empty()) break;

            std::istringstream iss(buf);
            key_type key;
            if(fmt == FMT_DICT)
            {
                iss.ignore(std::numeric_limits<size_t>::max(), ' ');
                assert(!iss.eof());

                iss >> key;
                if(iss.fail() || iss.bad())
                    throw IOException(__FUNCTION__, ": failed to parse key");

                newnode(key, dupmode);
            }
            else if(fmt == FMT_FREQ || fmt == FMT_LIST) // freq and list work in the same way
            {
                iss >> key;
                if(iss.fail() || iss.bad())
                    throw IOException(__FUNCTION__, ": failed to parse key");

                newnode(key, dupmode);
            }
        }
        fp.close();
        return true;
    }

    void save(const char *fn, int fmt);

    void merge(NodeMap& g, DUPMODE dupmode = DUP_OVERWRITE, bool free_other = true)
    {
        data_type *ptgt = &data, *psrc = &g.data;
        if(free_other && size() < g.size())
            std::swap(ptgt, psrc);

        for(auto& itr : *psrc)
        {
            insertex(*ptgt, move(itr), dupmode);
//            if(!overwrite)
//                assert(ptgt->find(itr) == ptgt->end());
//            ptgt->insert(itr);
        }
        if(free_other)
        {
            if(ptgt != &data)
            {
                data = move(*ptgt);
                ptgt->clear();
            }
            else
            {
                psrc->clear();
            }
        }
    }

    void clear() { data.clear(); }
    NodeMap copy() const { return std::move(NodeMap(data)); }

    std::vector<const_iterator> split(int parts) const
    {
        size_t sz = ceil(double(size()) / parts);
        std::vector<const_iterator> ret;
        size_t cnt = 0;
        for(auto itr = begin(); itr != end(); itr++, cnt++)
            if(cnt % sz == 0)
                ret.push_back(itr);
        ret.push_back(end());
        return move(ret);
    }

    iterator find(const key_type& key) { return data.find(key); }
    const_iterator find(const key_type& key) const { return data.find(key); }
    bool exists(const key_type& key) const { return data.find(key) != data.end(); }
    iterator begin() { return data.begin(); }
    const_iterator begin() const { return data.begin(); }
    iterator end() { return data.end(); }
    const_iterator end() const { return data.end(); }
    size_type size() const { return data.size(); }
    // TODO: it can be fancier if we may assign to this bool value
    bool operator[](const key_type& key) { return exists(key); }

    template <typename T>
    NodeMap subset(T begin, T end) const
    {
        NodeMap ret;
        for(auto i = begin; i != end; i++)
        {
            auto itr = find(*i);
            if(itr != end())
                ret.newnode(*itr, DUP_IGNORE);
        }
        return move(ret);
    }

    // this is an alias to exists in NodeSet case
    const bool get(const key_type& key) const
    {
        return exists(key);
    }
    
    // TODO: use a node view here 
    NodeMap nodes() const
    {
        return *this; 
    }

    std::pair<iterator, bool> newnode(const key_type& node, DUPMODE dupmode = DUP_OVERWRITE)
    {
        return insertex(node, dupmode);
    }

    template <typename T>
    std::pair<iterator, bool> newnode(T begin, T end, bool overwrite = true)
    {
        std::pair<iterator, bool> ret;
        for(auto itr = begin; itr != end; itr++)
        {
            ret = newnode(*itr, overwrite);
        }
        return ret;
    }

    static const NodeMap nullinst;

private:
    NodeMap(const NodeMap& ns) = delete;
    NodeMap& operator=(const NodeMap&) = delete;

    static std::pair<iterator, bool> insertex(data_type& data, const key_type& key, DUPMODE dupmode)
    {
        std::pair<iterator, bool> res;
        switch(dupmode)
        {
        case DUP_IGNORE:
        case DUP_OVERWRITE:
            return data.insert(key);
        case DUP_WARN:
            res = data.insert(key);
            if(!res.second)
            {
                std::ostringstream oss;
                oss << "Duplicated key: " << key;
                throw DuplicateKeyException<key_type>(__FUNCTION__, oss.str(), key);
            }
            return res;
        default:
            assert(0);
        }
    }

    std::pair<iterator, bool> insertex(const key_type& key, DUPMODE dupmode = DUP_OVERWRITE)
    {
        return insertex(this->data, key, dupmode);
    }

protected:
    data_type data;
};

template <typename key_type>
const NodeMap<key_type, nullptr_t> NodeMap<key_type, nullptr_t>::nullinst;

template <typename key_type>
using NodeSet = NodeMap<key_type, nullptr_t>;

#endif // NodeMap_H_INCLUDED
