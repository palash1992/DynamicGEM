#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

#include <boost/python.hpp>
namespace py = boost::python;

#include <Eigen/Core>
namespace ei = Eigen;

#include <map>
#include <vector>
#include <string>
#include <exception>
#include <set>
#include <map>
#include <cmath>
#include <iostream>
using namespace std;

#include <omp.h>

#include "graph/graph_pywrapper.h"
#include "boost_python_omp.h"

struct Record
{
    int i, j, k;
    int tm, lb;
    float wtv1, wtv2;
};

using Tensor1D = ei::Map<ei::Array<float, 1, ei::Dynamic, ei::RowMajor>, ei::RowMajor>;
using Tensor1D_Managed = ei::Array<float, 1, ei::Dynamic, ei::RowMajor>;

using Tensor2D = ei::Map<ei::Array<float, ei::Dynamic, ei::Dynamic, ei::RowMajor>, ei::RowMajor>;
using Tensor2D_Managed = ei::Array<float, ei::Dynamic, ei::Dynamic, ei::RowMajor>;

using Tensor3D = PyArrayObject*;

template <typename graph_t> using cgraph_type = const typename graph_t::CGraph*;
template <typename graph_t> using node_type = typename graph_t::CGraph::node_type;

template <typename graph_t>
Tensor1D_Managed X(int a, int b, int c, cgraph_type<graph_t> g, Tensor2D emb, const vector<node_type<graph_t>>& nodenames)
{
    float w1 = g->edge_value(nodenames[a], nodenames[c]);
    float w2 = g->edge_value(nodenames[b], nodenames[c]);
    
    if(w1 < 1e-6 || w2 < 1e-6)  // save computation
        if(!g->exists(nodenames[a], nodenames[c]) || !g->exists(nodenames[b], nodenames[c]))
            throw runtime_error("invalid open triangle");

    return (emb.row(c) - emb.row(a)) * w1 + (emb.row(c) - emb.row(b)) * w2;
}

template <typename graph_t>
float P(int a, int b, int c, cgraph_type<graph_t> g, Tensor2D emb, Tensor1D theta, const vector<node_type<graph_t>>& nodenames)
{
    Tensor1D_Managed x = X<graph_t>(a, b, c, g, emb, nodenames);
    float power = theta.segment(0, theta.size() - 1).cwiseProduct(x).sum();
    power = -(power + theta(0, theta.size() - 1));
 
    if(power > 100.0f)
        return 0.0f;
    else
        return 1.0f / (1 + exp(power));
}

template <typename graph_t>
void translate_input(const py::list py_graph, const py::list py_nodenames, vector<cgraph_type<graph_t>> *graphs, vector<node_type<graph_t>> *nodenames)
{
    // graph
    for(int i = 0; i < py::len(py_graph); i++)
    {
        cgraph_type<graph_t> g = (cgraph_type<graph_t>)py::extract<uintptr_t>(py_graph[i].attr("data")())(); 
        graphs->push_back(g);
    }

    // nodenames
    for(int i = 0; i < py::len(py_nodenames); i++)
    {
        py::extract<node_type<graph_t>> ext(py_nodenames[i]);
        if(!ext.check())
            throw runtime_error("Type check failed for nodename convertion");
        nodenames->push_back(ext());
    }
}

// this is required because eigen::map REQUIRES proper init in debug mode,
// so we HAVE TO directly return it rather than passing a pointer
Tensor1D translate_1darray(py::object arr)
{
    PyArrayObject *obj = (PyArrayObject*)arr.ptr();
    int sz = PyArray_DIM(obj, 0);
    // assert float type
    if(PyArray_DESCR(obj)->kind != 'f')
        throw logic_error("dtype of ndarray is not float32!");
    return Tensor1D((float*)PyArray_DATA(obj), 1, sz);
}

Tensor3D translate_3darray(py::object arr)
{
    return (Tensor3D)arr.ptr();
}

template <typename T>
T extract(py::object obj)
{
    py::extract<T> ext(obj);
    if(!ext.check())
    {
        ostringstream oss;
        oss << "Type check failed for type " << typeid(T).name();
        throw runtime_error(oss.str());
    }
    return ext();
}

void extract_record(py::object rec, Record *out)
{
    py::extract<py::list> ext(rec);
    if(!ext.check())
        throw runtime_error("Type check failed for data record, expecting py::list");
    py::list lst = ext();
    out->tm = extract<int>(lst[0]);
    out->k = extract<int>(lst[1]);  // center node
    out->i = extract<int>(lst[2]);
    out->j = extract<int>(lst[3]);
    out->lb = extract<int>(lst[4]);
    out->wtv1 = extract<float>(lst[5]);
    out->wtv2 = extract<float>(lst[6]);
}

Tensor2D slice_tensor3d(Tensor3D t, int idx)
{
    void *data = PyArray_GETPTR1(t, idx);
    // assert float type
    if(PyArray_DESCR(t)->kind != 'f')
        throw logic_error("dtype of ndarray is not float32!");
    return Tensor2D((float*)data, PyArray_DIM(t, 1), PyArray_DIM(t, 2));
}

template <typename graph_t>
py::list _emcoef(py::list data, py::object py_emb, py::object py_theta, py::list py_graphs, py::list py_nodenames, int localstep)
{
    vector<cgraph_type<graph_t>> graphs;
    vector<node_type<graph_t>> nodenames;

    py::list ret;
    ret.append(0);
    ret *= py::len(data);

    translate_input<graph_t>(py_graphs, py_nodenames, &graphs, &nodenames);
    Tensor3D emb = translate_3darray(py_emb);
    Tensor1D theta = translate_1darray(py_theta);

    // build name2idx
    map<node_type<graph_t>, int> name2idx;
    int idx_cnt = 0;
    for(const auto& name : nodenames)
        name2idx[name] = idx_cnt++;

    double eps = 1e-6;
    int datalen = py::len(data);
    int pardeg = 120;
    int num_threads = omp_get_num_procs();
    //int num_threads = 1;  // for debug

    GILRelease gilrelease;

    OMP_INIT_FOR(datalen, pardeg);
#ifdef DEBUG
    cout << "step size " << __omp_step_size << ' ' << __omp_sz << ' ' << __omp_deg << endl;
#endif
#pragma omp parallel for shared(data, localstep, graphs, emb, theta, nodenames, ret) num_threads(num_threads) schedule(dynamic, 1)
    OMP_BEGIN_FOR(lb, ub);

#ifdef DEBUG 
    cout << "thread " << omp_get_thread_num() << ": from " << lb << " to " << ub << endl;
#endif

    Record currec[ub - lb];
    double curC[ub - lb];

    { GILAcquire gil;
    for(int i = lb; i < ub; i++)
        extract_record(data[i], &currec[i - lb]);
    }

    for(int i = lb; i < ub; i++)
    {
        double C, C0, C1;
        Record rec = currec[i - lb];

        int tm0based = rec.tm - localstep;
        if(tm0based < 0)
            throw runtime_error("trying to access graph before the first time step");

        const cgraph_type<graph_t> g = graphs[tm0based];
        Tensor2D curemb = slice_tensor3d(emb, tm0based);

        if(rec.lb == 0)
        {
            C = 1.0;
        }
        else
        {
            C0 = P<graph_t>(rec.i, rec.j, rec.k, g, curemb, theta, nodenames);
            const auto& inbr = g->get_value(nodenames[rec.i]);
            set<node_type<graph_t>> cmnbr;
            for(const auto& itr : g->get_value(nodenames[rec.j]))
                if(inbr.exists(itr.first))
                    cmnbr.insert(itr.first);

            C1 = 1;
            for(const auto& nbr : cmnbr)
            {
                C1 *= (1 - P<graph_t>(rec.i, rec.j, name2idx[nbr], g, curemb, theta, nodenames));
            }
            C1 = 1.0 - C1;

            C = 1.0 - C0 / (C1 + eps);

            if(!isfinite(C))
            {
                cerr << C0 << ' ' << C1 << ' ' << C << endl;
                cerr << rec.i << ' ' << rec.j << ' ' << rec.k << endl;
                cerr << g->exists(nodenames[rec.i], nodenames[rec.k]) << ' ' << g->exists(nodenames[rec.j], nodenames[rec.k]) << endl;
                for(const auto& nbr : g->get_value(nodenames[rec.i]))
                    cerr << name2idx[nbr.first] << ' ';
                cerr << endl;
                for(const auto& nbr : g->get_value(nodenames[rec.j]))
                    cerr << name2idx[nbr.first] << ' ';
                cerr << endl;
                throw runtime_error("inf or nan detected when calculating em coefficients");
            }
        }
        curC[i - lb] = float(C);
    }
    { GILAcquire gil;
    for(int i = lb; i < ub; i++)
    {
        Record rec = currec[i - lb];
        ret[i] = py::make_tuple(py::list(py::make_tuple(rec.tm, rec.k, rec.i, rec.j)), py::list(py::make_tuple(curC[i - lb], rec.wtv1, rec.wtv2)));
    } }

    OMP_END_FOR();
    return ret;
}

py::list emcoef(py::list data, py::object py_emb, py::object py_theta, py::list py_graphs, py::list py_nodenames, int localstep)
{
    string cls = py::extract<string>(py::object(py_graphs[0]).attr("__class__").attr("__name__"));
    if(cls == "Graph_Int32_Float")
    {
        return _emcoef<Graph_Int32_Float>(data, py_emb, py_theta, py_graphs, py_nodenames, localstep);
    }
    else if(cls == "Graph_String_Float")
    {
        return _emcoef<Graph_String_Float>(data, py_emb, py_theta, py_graphs, py_nodenames, localstep);
    }
    else
    {
        throw runtime_error(string("Unknown graph type ") + cls);
    }
}

BOOST_PYTHON_MODULE(dynamic_triad_cimpl)
{
    PyEval_InitThreads();
    py::def("emcoef", &emcoef);
}
