#ifndef OMP_UTILS_H
#define OMP_UTILS_H

#include <Python.h>
#include <cmath>

#define OMP_INIT_FOR(sz, deg) \
    do { \
        int __omp_step_size = int(ceil(float(sz) / deg) + 0.5f); \
        int __omp_deg = deg; \
        int __omp_sz = sz;

#define OMP_BEGIN_FOR(lbvar, ubvar) \
        for(int __omp_par = 0; __omp_par < __omp_deg; __omp_par++) { \
            int lbvar = __omp_par * __omp_step_size; \
            int ubvar = (__omp_par + 1) * __omp_step_size; \
            ubvar = (ubvar > __omp_sz ? __omp_sz : ubvar); \
            lbvar = (lbvar > __omp_sz ? __omp_sz : lbvar);

#define OMP_END_FOR() \
        } \
    } while(0);


class GILRelease {
    public:
        inline GILRelease() { m_thread_state = PyEval_SaveThread(); }
        inline ~GILRelease() { PyEval_RestoreThread(m_thread_state); m_thread_state = NULL; }
    private:
        PyThreadState* m_thread_state;
};

struct GILAcquire{
    GILAcquire() {
        state = PyGILState_Ensure();
    }

    ~GILAcquire() {
        PyGILState_Release(state);
    }
private:
    PyGILState_STATE state;
};

#endif // OMP_UTILS_H
