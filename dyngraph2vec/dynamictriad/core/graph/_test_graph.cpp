#include <string>
#include <iostream>
using namespace std;

#include "nodemap.h"
#include "nodeset.h"
#include "graph.h"

void test_nodemap()
{
    NodeMap<int, string> *nm = new NodeMap<int, string>();
    cout << "test default value" << endl;
    cout << nm->get_value(0) << endl;
    nm->newnode(100, string("hello"));
    nm->newnode(200, string("world"));
    cout << (*nm)[100] << endl;
    assert((*nm)[100] == "hello");
    NodeMap<int, string> *nm1 = new NodeMap<int, string>(move(*nm));
    cout << (*nm1)[100] << endl;
    assert((*nm1)[100] == "hello");
    try
    {
        nm1->newnode(100, string("olleh"), DUP_WARN);
        assert(0);
    }
    catch(...)
    {
    }
    nm1->newnode(100, string("olleh"));
    cout << (*nm1)[100] << endl;
    assert((*nm1)[100] == "olleh");
    nm1->newnode(200, string("dlrow"), DUP_IGNORE);
    cout << (*nm1)[200] << endl;
    assert((*nm1)[200] == "world");
    cout << nm1->exists(100) << endl;
    assert(nm1->exists(100));
    nm->newnode(400, string("abcd"));
    nm->newnode(200, string("dcba"), DUP_OVERWRITE);
    nm1->merge(*nm);
    int keys[] = {100, 200, 400};
    string values[] = {"olleh", "dcba", "abcd"};
    int pt = 0;
    for(auto& itr : *nm1)
    {
        cout << itr.first << itr.second << endl;
        assert(itr.first == keys[pt]);
        assert(itr.second == values[pt]);
        pt++;
    }
    NodeMap<int, int> nm2;
    try
    {
        nm2.get(10);
    }
    catch(NoSuchFieldException& e)
    {
        cout << e.what() << endl;
    }
}

void test_nodeset()
{
    NodeSet<string> *nm = new NodeSet<string>();
    nm->newnode(string("hello"));
    nm->newnode(string("world"));
    cout << (*nm)[string("hello")] << endl;
    assert((*nm)[string("hello")]);
    NodeSet<string> *nm1 = new NodeSet<string>(move(*nm));
    cout << (*nm1)[string("hello")] << endl;
    assert((*nm1)[string("hello")]);
    try
    {
        nm1->newnode(string("hello"), DUP_WARN);
        assert(0);
    }
    catch(...)
    {
    }
    nm1->newnode(string("dlrow"), DUP_IGNORE);
    cout << (*nm1)[string("dlrow")] << endl;
    assert((*nm1)[string("dlrow")]);
    cout << nm1->exists(string("dlrow")) << endl;
    assert(nm1->exists(string("dlrow")));
    nm->newnode(string("abcd"));
    nm->newnode(string("dcba"), DUP_OVERWRITE);
    nm1->merge(*nm);
    string values[] = {"abcd", "dcba", "dlrow", "hello", "world"};
    int pt = 0;
    for(auto& itr : *nm1)
    {
        cout << itr << endl;
        assert(itr == values[pt]);
        pt++;
    }
}

struct A { };

void test_graph()
{
    Graph<string, float, NodeMap> g0;

    cout << "test default" << endl;
    cout << g0.edge_value("10", "20") << endl;
    g0.newnode("0");
    cout << g0.edge_value("0", "100") << endl;
    
    Graph<int, int, NodeMap> g;

    for(int i = 0; i < 10; i++)
    {
        g.newnode(i);
        g[i].newnode(i + 1, i + 1);
    }

    cout << "test tostring" << endl;
    g.tostring(cout);
    cout << endl;

    Graph<int, int, NodeMap> g1 = g.inverseEdge();
    for(int i = 1; i <= 10; i++)
    {
        cout << g1.exists(i) << endl;
        assert(g1.exists(i));
        cout << g1[i].exists(i - 1) << endl;
        assert(g1[i].exists(i - 1));
        assert(!g1[i].exists(i) && !g1[i].exists(i + 1));
        cout << g1[i][i - 1] << endl;
        assert(g1[i][i - 1] == i);
    }
    auto ev = g1.edges();
    assert(ev.end() == ev.end());
    for(auto itr = ev.begin(); itr != ev.end(); itr++)
    {
        cout << (*itr).first << ' ' << (*itr).second.first << ' ' << (*itr).second.second << endl;
    }

    cout << "testing save&load" << endl;
    g1.save("_test_graph.graph");

    Graph<int, int, NodeMap> g2;
    g2.load("_test_graph.graph");

    cout << "edges loaded" << endl;
    ev = g2.edges();
    for(auto itr = ev.begin(); itr != ev.end(); itr++)
    {
        cout << (*itr).first << ' ' << (*itr).second.first << ' ' << (*itr).second.second << endl;
    }

    cout << "compare with original" << endl;
    ev = g1.edges();
    for(auto itr = ev.begin(); itr != ev.end(); itr++)
    {
        cout << (*itr).first << ' ' << (*itr).second.first << ' ' << (*itr).second.second << endl;
        assert(g2.exists((*itr).first, (*itr).second.first));
    }

    g1.save("_test_graph_txt.graph", false);

    cout << "A has nullinst " << Has_nullinst<A>::value << endl;
    assert(Has_nullinst<A>::value == 0);
    cout << "edgemap has nullinst " << Has_nullinst<EdgeMap<int, int, NodeMap> >::value << endl;
    assert((Has_nullinst<EdgeMap<int, int, NodeMap> >::value) == 1);
}

void test_functions()
{
    Graph<string, float, NodeMap> g, g1;
    g.newnode("hello");
    g.newnode("world");
    g.newnode("WORLD");
    g.newnode("abcde");
    g1.newnode("hello");
    g1.newnode("world");
    g1.newnode("HELLO");
    g1.newnode("abcde");
    g.newedge("hello", "world", 1);
    g.newedge("WORLD", "world", 2);
    g.newedge("abcde", "WORLD", 1.2);
    g.newedge("abcde", "hello", 2.34);
    g1.newedge("hello", "world", 3);
    g1.newedge("hello", "HELLO", 4);
    g1.newedge("abcde", "hello", 2.66);
    g.merge_graph(g1);

    cout << "expected output (order does not matter):" << endl;
    cout << "hello world 4" << endl
         << "WOLRD world 2" << endl
         << "abcde WORLD 1.2" << endl
         << "abcde hello 5" << endl
         << "hello HELLO 4" << endl;

    cout << "real output: " << endl;

    auto ev = g.edges();
    for(const auto& itr : ev)
    {
        cout << itr.first << ' ' << itr.second.first << ' ' << itr.second.second << endl;
    }

    ev = g1.edges();
    for(const auto& itr : ev)
    {
        cout << itr.first << ' ' << itr.second.first << ' ' << itr.second.second << endl;
    }

    cout << "what if merge with itself" << endl;

    g.merge_graph(g);
    ev = g.edges();
    for(const auto& itr : ev)
    {
        cout << itr.first << ' ' << itr.second.first << ' ' << itr.second.second << endl;
    }
}

int main()
{
    test_nodemap();
    cout << "#################################" << endl;
    test_nodeset();
    cout << "#################################" << endl;
    test_graph();
    cout << "#################################" << endl;
    test_functions();
    return 0;
}
