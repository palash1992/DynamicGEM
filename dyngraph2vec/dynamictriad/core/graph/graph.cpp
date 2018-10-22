#include "graph.h"

//#include <list>
//#include <map>
//#include <utility>
//#include <cassert>
//#include <sstream>
//#include "exception.h"
//using namespace std;

//const Graph Graph::empty_graph;
//const list<Graph::key_type> Graph::empty_val;

//bool Graph::load(const char *fn, int fmt, bool overwrite, const NodeSet& filter, bool filter_out)
//{
//    // in this case, we filter_out an empty graph
//    if(&filter == &NodeSet::empty_ns)
//        filter_out = true;

//    FILE *fp = fopen(fn, "r");
//    if(!fp)
//    {
//        char buf[128];
//        sprintf(buf, "Cannot open graph file for reading: %s", fn);
//        perror(buf);
//        return false;
//    }

//    while(1)
//    {
//        Graph::key_type user = 0;
//        list<Graph::key_type> arr;
//        if(fmt == Graph::FMT_RAW)
//        {
//            user = readline(fp, &arr);
//            if(user == -1) break;
//        }
//        else
//        {
//            size_t ecnt;
//            if(fscanf(fp, WUID_IOFMT "%lu", &user, &ecnt) == EOF)
//                break;

//            for(size_t i = 0; i < ecnt; i++)
//            {
//                wuid_t tmp;
//                fscanf(fp, WUID_IOFMT, &tmp);
//                arr.push_back(tmp);
//            }
//        }

//        //assert(!exists(user));
//        if(filter_out ^ filter.exists(user))
//        {
//            if(!overwrite && exists(user))
//            {
//                ostringstream oss;
//                oss << "Duplicated key: " << user;
//                throw DuplicateKeyException<key_type>(oss.str(), user);
//            }
//            data[user] = move(arr);
//        }
//    }

//    fclose(fp);
//    return true;
//}

//bool Graph::load(FileNameInfo info, int fmt, bool overwrite, const NodeSet& filter, bool filter_out)
//{
//    // we use a separate graph to load because we do not consider overwrite when loading multiple files
//    Graph ng;
//    for(auto itr : info)
//    {
//        printf("loading %s\n", itr.c_str());
//        ng.load(itr.c_str(), fmt, false, filter, filter_out);
//    }
//    merge(ng, overwrite);
//    return true;
//}

//void Graph::merge(Graph& g, bool overwrite)
//{
//    if(size() < g.size())  // for efficiency
//        data.swap(g.data);
//    for(const auto& itr : g.data)
//    {
//        if(!overwrite && exists(itr.first))
//        {
//            ostringstream oss;
//            oss << "Duplicated key: " << itr.first;
//            throw DuplicateKeyException<key_type>(oss.str(), itr.first);
//        }
//        data[itr.first] = move(itr.second);
//    }
//    g.clear();
//}

//const list<Graph::key_type>& Graph::get(Graph::key_type key) const
//{
//    if(exists(key))
//        return find(key)->second;
//    else
//        return empty_val;
//}

//// TODO: return a view instead of creating a new nodeset
//NodeSet Graph::nodes() const
//{
//    NodeSet ret;
//    for(const auto& itr : data)
//    {
//        ret.newnode(itr.first);
//    }
//    return move(ret);
//}

//Graph Graph::subset(const NodeSet& ns)
//{
//    Graph ret;
//    for(const auto& itr : data)
//    {
//        if(ns.exists(itr.first))
//            ret[itr.first] = itr.second;
//    }
//    return move(ret);
//}

//bool Graph::save(const char *fn, int fmt)
//{
//    FILE *fp = fopen(fn, "w");
//    if(!fp)
//    {
//        perror("Cannot open network file for writing");
//        return false;
//    }

//    for(const auto& u : data)
//    {
//        fprintf(fp, WUID_IOFMT, u.first);
//        if(fmt == FMT_ADVANCED)
//            fprintf(fp, " %lu", u.second.size());
//        for(auto fu : u.second)
//            fprintf(fp, " " WUID_IOFMT, fu);
//        fprintf(fp, "\n");
//    }
//    fclose(fp);
//    return true;
//}

//bool Graph::save(const FileNameInfo& info, int fmt)
//{
//    if(info.rr != info.lr)
//        throw runtime_error("saving graph to multiple files is not supported");
//    for(auto itr : info)
//    {
//        return save(itr.c_str(), fmt);
//    }
//    return true;
//}

//Graph Graph::inverseEdge() const
//{

//}
