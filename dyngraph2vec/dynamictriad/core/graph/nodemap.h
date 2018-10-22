#ifndef NODEMAP_H
#define NODEMAP_H

#include <map>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <cassert>
#include <utility>
#include <limits>
#include <typeinfo>
#include "nodeset.h"
#include "exception.h"
#include "utils.h"
#include "ioutils.h"

#ifdef DEBUG
#include <iostream>
#endif

// TODO: make the implementation a little milder,
// like using a shared_ptr for default copying
// for the sake of template compatibilities
template <typename key_t, typename mapped_t>
class NodeMap
{
public:
    typedef key_t key_type;
    typedef mapped_t mapped_type;
    typedef std::map<key_type, mapped_type> data_type;
    typedef typename data_type::value_type value_type;
    typedef typename data_type::iterator iterator;
    typedef typename data_type::const_iterator const_iterator;
    typedef typename data_type::size_type size_type;

    NodeMap() {}
    NodeMap(const data_type& data) : data(data) {}
    NodeMap(data_type&& data) : data(data) {}
    NodeMap(NodeMap<key_type, mapped_type>&& ns) : data(move(ns.data)) {}
    NodeMap& operator=(NodeMap<key_type, mapped_type>&& ns)
    {
        if(&ns != this) data = std::move(ns.data);
        return *this;
    }

    bool save(const char *fn, bool binary = true)
    {
        std::ofstream out(fn, std::ios::out);
        if(!out.is_open() || out.bad())
            throw IOException(__FUNCTION__, ": Cannot open file for writing");
        bool res = save(out, binary);
        out.close();
        return res;
    }

    bool save(std::ostream& out, bool binary = true)
    {
        if(binary)
        {
            out << "bin";
            OBinArchive(out) << *this;
            return true;
        }
        else
        {
            out << "txt" << std::endl;
            // use tostring rather than default serialize impl
            tostring(out);
            return true;
        }
        return false;
    }

    bool load(const char *fn)
    {
        std::ifstream in(fn, std::ios::in);
        if(!in || in.bad())
            throw IOException(__FUNCTION__, ": Cannot open file for reading");

        bool res = load(in);
        in.close();
        return res;
    }

    bool load(std::istream& in)
    {
        char magic[4] = {};
        in.read(magic, 3);
        if(strcmp(magic, "bin") == 0)
        {
            IBinArchive(in) >> *this;
            return true;
        }
        else if(strcmp(magic, "txt") == 0)
        {
            fromstring(in);
            return true;
        }
        else
            throw InvalidFormatException(__FUNCTION__, "Unrecognized magic bytes found in saved graph");
        return false;
    }

    // NOTE: that we do not clear data in load functions
    bool fromstring(std::istream& in, DUPMODE dupmode = DUP_WARN)
    {
        size_t sz;
        in >> sz;
        for(size_t i = 0; i < sz; i++)
        {
            key_type key;
            mapped_type val;
            ITxtArchive(in) >> key;

            if(in.fail() || in.bad())
                throw IOException(__FUNCTION__, ": failed to parse key");
            ITxtArchive(in) >> val;
            if(in.fail() || in.bad())
                throw IOException(__FUNCTION__, ": failed to parse value");
            newnode(key, std::move(val), dupmode);
        }
        return true;
    }

    // *T must be of type string
    template <typename T>
    bool fromstring(T fnbegin, T fnend, DUPMODE dupmode = DUP_WARN)
    {
        bool res = true;
        for(auto itr = fnbegin; itr != fnend; ++itr)
            res &= fromstring((*itr).c_str(), dupmode);
        return res;
    }

    // TODO: this is not ideal, what if the class is nested three times?
    bool tostring(std::ostream& out) const
    {
        OTxtArchive ar(out);
        ar << size() << OTxtArchive::SetDelim('\n', true);
        for(const auto& itr : *this)
        {
            ar << itr.first << OTxtArchive::SetDelim('\n', true) << itr.second << OTxtArchive::SetDelim('\n', true);
        }
        return true;
    }

    template <class Archive>
    void serialize(Archive&& ar) const
    {
        ar << data;
    }

    template <class Archive>
    void deserialize(Archive&& ar)
    {
        ar >> data;
    }

    void merge(NodeMap<key_type, mapped_type>& g, DUPMODE dupmode = DUP_OVERWRITE, bool free_other = true)
    {
        if(this == &g)
        {
            if(dupmode == DUP_WARN)
                throw DuplicateKeyException<key_type>(__FUNCTION__,
                    "Trying to merge a graph with itself", key_type());
            else
                return;
        }

        data_type *ptgt = &data, *psrc = &g.data;
        if(free_other && size() < g.size())
            std::swap(ptgt, psrc);

        for(auto& itr : *psrc)
        {
            if(free_other)
                insertex(*ptgt, move(itr), dupmode);
            else
                insertex(*ptgt, SafeCopy<value_type>::copy(itr), dupmode);
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
        return std::move(ret);
    }
    iterator find(const key_type& key) { return data.find(key); }
    const_iterator find(const key_type& key) const { return data.find(key); }
    bool exists(const key_type& key) const { return data.find(key) != data.end(); }
    iterator begin() { return data.begin(); }
    const_iterator begin() const { return data.begin(); }
    iterator end() { return data.end(); }
    const_iterator end() const { return data.end(); }
    size_type size() const { return data.size(); }
    mapped_type& operator[](const key_type& key) { return data[key]; }
    
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

    mapped_type& get(const key_type& key, bool create_default = false, mapped_type&& dft = mapped_type())
    {
        iterator itr = find(key);
        if(itr != end())
        {
            return itr->second;
        }
        else // note that nullinst must exist in mapped_type if dft is not given
        {
            if(create_default)
            {
                auto itr = newnode(key, std::move(dft), DUP_WARN).first;
                return itr->second;
            }
            else
            {
                const void *ptr = Has_nullinst<mapped_type>::safeget();
                if(ptr) return *(mapped_type*)ptr;
                else
                    throw NoSuchFieldException(__FUNCTION__, std::string("No such field in type ") + typeid(mapped_type).name(), "nullinst");
            }
        }
    }

    const mapped_type& get_value(const key_type& key, const mapped_type& dft = Defaults<mapped_type>::get()) const
    {
        const_iterator itr = find(key);
        if(itr != end())
        {
            return itr->second;
        }
        else
            return dft;
    }

    // TODO: use a node view here 
    NodeSet<key_type> nodes() const
    {
        NodeSet<key_type> ns;
        for(auto& itr : data)
            ns.newnode(itr.first);
        return std::move(ns);
    }

    // think more carefully on whether to use l/r-value reference on
    // value_type, key_type and mapped_type
    std::pair<iterator, bool> newnode(value_type&& val, DUPMODE dupmode = DUP_OVERWRITE)
    {
        return insertex(move(val), dupmode);
    }

    std::pair<iterator, bool> newnode(const key_type& node, mapped_type&& val, DUPMODE dupmode = DUP_OVERWRITE)
    {
        return insertex(std::make_pair(node, std::move(val)), dupmode);
    }

    std::pair<iterator, bool> newnode(const key_type& node, DUPMODE dupmode = DUP_OVERWRITE)
    {
        return insertex(std::make_pair(node, mapped_type()), dupmode);
    }

    template <typename T>
    std::pair<iterator, bool> newnode(T begin, T end, DUPMODE dupmode = DUP_OVERWRITE)
    {
        std::pair<iterator, bool> ret;
        for(auto itr = begin; itr != end; itr++)
        {
            ret = newnode(*itr, dupmode);
        }
        return ret;
    }

    static const NodeMap nullinst;
private:
    NodeMap(const NodeMap& ns) = delete;
    NodeMap& operator=(const NodeMap&) = delete;
    //void loadgraph(const char *fn, const NodeMap& filter, bool filter_out)

    // insert to data while handling duplicated keys
    static std::pair<iterator, bool> insertex(data_type& data, value_type&& val, DUPMODE dupmode)
    {
        std::pair<iterator, bool> res;
        switch(dupmode)
        {
        case DUP_IGNORE:
            return data.insert(move(val));
        case DUP_OVERWRITE:
            res = data.insert(move(val));
            if(!res.second)
            {
                res.first->second = std::move(val.second);
                res.second = true;
            }
            return res;
        case DUP_WARN:
            res = data.insert(move(val));
            if(!res.second)
            {
                std::ostringstream oss;
                oss << "Duplicated key: " << val.first;
                throw DuplicateKeyException<key_type>(__FUNCTION__, oss.str(), val.first);
            }
            return res;
        default:
            assert(0);
        }
    }

    inline std::pair<iterator, bool> insertex(value_type&& val, DUPMODE dupmode)
    {
        return insertex(this->data, std::move(val), dupmode);
    }

protected:
    data_type data;
};

template <typename key_t, typename mapped_t>
const NodeMap<key_t, mapped_t> NodeMap<key_t, mapped_t>::nullinst;

#endif // NODEMAP_H
