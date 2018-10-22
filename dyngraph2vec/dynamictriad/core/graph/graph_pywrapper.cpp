#include <Python.h>
#include <boost/python.hpp>
namespace py = boost::python;

#include "graph_pywrapper.h"

using namespace std;

#ifdef DEBUG
#include <cstdio>
#endif
#include <cassert>

template <typename T>
struct GraphPickleSuite : py::pickle_suite
{
    static py::tuple getinitargs(const T& t)
    {
        return py::tuple();
    }

    static py::tuple getstate(const T& t)
    {
        std::string str = t.to_str();
        return py::make_tuple<std::string>(str);
    }

    static void setstate(T& t, py::tuple state)
    {
        py::extract<std::string> ext(state[0]);
        std::string str = ext();
        t.parse_str(str);
    }
};



#define DECL_GRAPH_OVERLOAD(cls) \
    BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(cls ## _adjacency_overloads, cls::adjacency, 0, 1) \
    BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(cls ## _exists_overloads, cls::exists, 1, 2)

/*
.def("__init__", py::make_constructor( \
         static_cast<cls* (*)(const char *, int, int)>(&cls::makeGraph))) \
*/

#define DECL_GRAPH_METHODS(cls) \
    py::class_<cls, boost::noncopyable >(#cls, py::no_init) \
    .def("__init__", py::make_constructor( \
             static_cast<cls* (*)()>(&cls::makeGraph))) \
    .def("save_graph", &cls::save_graph) \
    .def("load_graph", &cls::load_graph) \
    .def(str(py::self)) \
    .def("parse_str", &cls::parse_str) \
    .def("to_str", &cls::to_str) \
    .def("add_vertex", &cls::add_vertex) \
    .def("add_edge", &cls::add_edge) \
    .def("inc_edge", &cls::inc_edge) \
    .def("num_vertices", &cls::num_vertices) \
    .def("vertices", &cls::vertices) \
    .def("inverse_edge", &cls::inverseEdge, \
         py::return_value_policy<py::manage_new_object>()) \
    .def("merge", &cls::merge, (py::arg("g"), py::args("free_other")=false)) \
    .def("get", &cls::get) \
    .def("edge", &cls::edge) \
    .def("exists", (bool (cls::*)(typename cls::CGraph::key_type, typename cls::CGraph::key_type))0, cls ## _exists_overloads()) \
    .def("out_neighbours", &cls::out_neighbours) \
    .def("out_degree", &cls::out_degree) \
    .def("node_type", &cls::node_type_name) \
    .def("weight_type", &cls::weight_type_name) \
    .def("data", &cls::data) \
    .def_pickle(GraphPickleSuite<cls>())
//        .def("subgraph", &cls::subgraph,
//             py::return_value_policy<py::manage_new_object>())
    //.def("adjacency", &cls::adjacency)
//        .def("adjacency", (py::list (cls::*)(py::list))0, adjacency_overloads())

template <typename T>
T* translate(uintptr_t ptr)
{
    return reinterpret_cast<T*>(ptr);
}

template <typename T>
py::object transfer_object(T *g)
{
    py::object expose = py::make_function(&translate<T>, py::return_value_policy<py::manage_new_object>());
    return expose(reinterpret_cast<uintptr_t>(g));
}

void _throw_unsupported(const string& ntype, const string& wtype)
{
    ostringstream oss;
    oss << "Unsupported node type " << ntype << " with weight type " << wtype;
    throw runtime_error(oss.str());
}

// NOTE: currently we ignore other versions of makeGraph since they are usually
//       not necessary
// TODO: any better way to do this?
py::object makeGraph(const string& ntype, const string& wtype)
{
    if(ntype == "int" || ntype == "int32")
    {
        if(wtype == "float")
        {
            py::object ret = transfer_object<Graph_Int32_Float>(Graph_Int32_Float::makeGraph());
            //py::object ret = py::object(Graph_Int32_Float::makeGraph());
            return ret;
        }
        else
            _throw_unsupported(ntype, wtype);
    }
    else if(ntype == "long" || ntype == "int64")
    {
        if(wtype == "float")
        {
            py::object ret = transfer_object<Graph_Int64_Float>(Graph_Int64_Float::makeGraph());
            return ret;
        }
        else
            _throw_unsupported(ntype, wtype);
    }
    else if(ntype == "string")
    {
        if(wtype == "float")
        {
            return transfer_object<Graph_String_Float>(Graph_String_Float::makeGraph());
            //return py::object(Graph_String_Float::makeGraph());
        }
        else
            _throw_unsupported(ntype, wtype);
    }
    else
        _throw_unsupported(ntype, wtype);

    return py::object(0);
//    return (Graph_Int_Float*)0;
}

DECL_GRAPH_OVERLOAD(Graph_Int32_Float)
DECL_GRAPH_OVERLOAD(Graph_String_Float)
DECL_GRAPH_OVERLOAD(Graph_Int64_Float)

BOOST_PYTHON_MODULE(mygraph)
{
    DECL_GRAPH_METHODS(Graph_Int32_Float);
    DECL_GRAPH_METHODS(Graph_String_Float);
    DECL_GRAPH_METHODS(Graph_Int64_Float);
    py::scope().attr("Graph_Int_Float") = py::scope().attr("Graph_Int32_Float");  // add an alias
//    py::enum_<CGraph::format>("format")
//        .value("FMT_RAW", CGraph::FMT_RAW)
//        .value("FMT_ADVANCED", CGraph::FMT_ADVANCED)
//    ;
    py::def("Graph", &makeGraph);
    py::enum_<DUPMODE>("DUPMODE")
        .value("DUP_IGNORE", DUP_IGNORE)
        .value("DUP_OVERWRITE", DUP_OVERWRITE)
        .value("DUP_WARN", DUP_WARN)
    ;
}


//template <typename NodeType, typename WeightType>
//class GraphPyWrapper
//{
//public:


//    GraphPyWrapper(const char *dir, int lb, int ub)
//        : g(new CGraph())
//    {
//#ifdef DEBUG
//        fprintf(stderr, "GraphPyWrapper load ctor %p\n", this);
//#endif
//        FileList lst{dir, lb, ub};
//        g->load(lst.begin(), lst.end());
//    }



//    static GraphPyWrapper* makeGraph(const char *dir, int lb, int ub)
//    {
//        return new GraphPyWrapper(dir, lb, ub);
//    }



//    GraphPyWrapper *subgraph(py::list& users) const
//    {
//        NodeSet ns;
//        for(int i = 0; i < len(users); i++)
//        {
//            py::extract<CGraph::key_type> extractor(users[i]);
//            if(extractor.check())
//                ns.newnode(extractor());
//        }
//        CGraph *sg = new CGraph(move(g->subgraph(ns, true)));
//        GraphPyWrapper *ret = new GraphPyWrapper(sg);
//        return ret;
//    }



//    py::list adjacency() const
//    {
//        CGraph::sptype sp = g->adjacency();
//        py::list ret, ridx, cidx;
//        for(int r = 0; r < sp.outerSize(); r++)
//            for(CGraph::sptype::InnerIterator itr(sp, r); itr; ++itr)
//            {
//                ridx.append(itr.row());
//                cidx.append(itr.col());
//            }
//        ret.append(ridx);
//        ret.append(cidx);
//        return ret;
//    }

//    // this is an ugly fix which is neither safe nor elegant
//    // should be removed as soon as order is implemented
//    py::list adjacency(py::list order) const
//    {
//        py::stl_input_iterator<CGraph::key_type> begin(order), end;
//        // wth is this stl_input_iterator shares a common state?
//        list<CGraph::key_type> arr(begin, end);
//        CGraph::sptype sp = g->adjacencyOrdered(arr.begin(), arr.end());
//        py::list ret, ridx, cidx;
//        for(int r = 0; r < sp.outerSize(); r++)
//            for(CGraph::sptype::InnerIterator itr(sp, r); itr; ++itr)
//            {
//                ridx.append(itr.row());
//                cidx.append(itr.col());
//            }
//        ret.append(ridx);
//        ret.append(cidx);
//        return ret;
//    }


//};
