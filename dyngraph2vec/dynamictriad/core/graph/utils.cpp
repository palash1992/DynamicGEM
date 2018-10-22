#include "utils.h"
//#include <list>
//#include <cstdio>
//#include <cstdlib>
//#include <cerrno>
//#include <cassert>
//#include <cstring>
//#include <string>
//using namespace std;

//wuid_t readline(FILE *fp, list<wuid_t> *out)
//{
//    thread_local static char buf[4096];
//    char *pt = buf;

//    if(!fgets(buf, sizeof(buf), fp))
//        return -1;  // eof reached
//    out->clear();
//    wuid_t curuser = str2wuid(pt, &pt, 10);

//    assert(errno != ERANGE);
//    assert(pt != buf);  // the first user id must exist
//    assert(pt < buf + sizeof(buf) && *pt != 0);  // impossible for such a long user id!!!

//    while(1)
//    {
//        while(pt < buf + sizeof(buf) && ISSPACE(*pt)) pt++;
//        assert(pt < buf + sizeof(buf));  // this is true as long as trailing \0 exists
//        if(*pt == '\n') break;

//        char *nextpt;
//        wuid_t user = str2wuid(pt, &nextpt, 10);
//        assert(errno != ERANGE);
//        assert(nextpt != pt || *pt == 0); // the only way to fail a parse is \0 reached

//        if(*nextpt == 0) // load next
//        {
//            int cpychrs = nextpt - pt;    // bytes_left - 1(\0)
//            strncpy(buf, pt, cpychrs);    // what if dst and src overlapped???
//            if(!fgets(buf + cpychrs, sizeof(buf) - cpychrs, fp))
//            {
//                *(buf + cpychrs) = '\n';
//                *(buf + cpychrs + 1) = 0;
//            }

//            pt = buf;
//        }
//        else
//        {
//            out->push_back(user);
//            pt = nextpt;
//        }

//    }
//    return curuser;
//}

//void _test_read_line(const char *samplefn)
//{
//    list<wuid_t> ln;
//    FILE *fp = fopen(samplefn, "r");
//    FILE *fp1 = fopen(samplefn, "r");
//    while(1)
//    {
//        wuid_t u = readline(fp, &ln);
//        wuid_t u1;
//        if(u == -1) break;
//        fscanf(fp1, "%ld", &u1);
//        assert(u == u1);
//        for(auto itr : ln)
//        {
//            fscanf(fp1, "%ld", &u1);
//            assert(itr == u1);
//        }
//    }
//    assert(fscanf(fp, "%*d") == EOF);
//    printf("All test passed\n");
//    fclose(fp1);
//    fclose(fp);
//}

//string wuid2str(wuid_t u)
//{
//    char buf[32];  // should be long enough for int64
//    sprintf(buf, WUID_IOFMT, u);
//    return string(buf);
//}
