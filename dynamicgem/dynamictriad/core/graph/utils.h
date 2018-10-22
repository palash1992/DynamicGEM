#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <list>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <type_traits>
#include <typeinfo>
#include "exception.h"

#define remove_qualifiers(x) typename std::remove_cv<typename std::remove_reference<x>::type>::type
#define is_primitive(x) std::is_scalar<remove_qualifiers(x)>::value
#define ifelse(c, x, y) typename std::conditional<c, x, y>::type

 // this macro detects any unambiguous, non-private member in a type 
 // TODO: should be able to return non-const members
#define make_hasx(x) \
template <typename T> struct Has_ ## x \
{ \
private: \
    template <typename T1> struct int_ { typedef int type; }; \
    template <typename T1, typename TEST = int> \
    struct TestHas { \
        static const void *safeget(const T1& obj, bool croak = false) { \
            if(croak) \
                throw NoSuchFieldException("", "No such field", #x); \
            else \
                return 0; \
        } \
        static const void *safeget(bool croak = false) {  \
            if(croak) \
                throw NoSuchFieldException("", "No such field", #x); \
            else \
                return 0; \
        } \
        const static int value = 0; \
    }; \
    template <typename T1> \
    struct TestHas<T1, typename int_<decltype(T1:: x)>::type> { \
        static const void *safeget(T1& obj, bool croak = false) { return &(obj. x); } \
        static const void *safeget(bool croak = false) { return &(T1:: x); } \
        const static int value = 1; \
    }; \
public: \
    const static int value = TestHas<T>::value; \
    static const void *safeget(const T& obj) { return TestHas<T>::safeget(obj); } \
    static const void *safeget() { return TestHas<T>::safeget(); } \
}

make_hasx(nullinst);

struct DftPrimitive {};
struct DftNullinst {};

template <typename T>
using DefaultCond = ifelse(is_primitive(T), DftPrimitive, \
    ifelse(Has_nullinst<T>::value, DftNullinst, void));

template <typename T, class = DefaultCond<T> >
struct Defaults
{
    static const T dft;
    static const T& get() { return dft; }
    //static const T& get()
    //{
    //    std::ostringstream oss("No default value defined for type ");
    //    oss << typeid(T).name();
    //    throw NoSuchFieldException(__FUNCTION__, oss.str(), "default");
    //}
};

template <typename T, class dummy>
const T Defaults<T, dummy>::dft;

template <typename T>
struct Defaults<T, DftPrimitive>
{
    static const T dft;
    static const T& get() { return dft; }
};

template <typename T>
const T Defaults<T, DftPrimitive>::dft = T();

template <typename T>
struct Defaults<T, DftNullinst>
{
    static const T& get() { return T::nullinst; }
};

struct CopyConstructable {};
struct CopyAssignable {};

template <typename T>
using CopyCond = ifelse(std::is_copy_constructible<T>::value, CopyConstructable, \
    ifelse(std::is_copy_assignable<T>::value, CopyAssignable, void));

template <typename T, typename = CopyCond<T> >
struct SafeCopy
{
    static T copy(const T& from)
    {
        return std::move(from.copy());
    }
};

template <typename T>
struct SafeCopy<T, CopyConstructable>
{
    static T copy(const T& from)
    {
        T to(from);
        return std::move(to);
    }
};

template <typename T>
struct SafeCopy<T, CopyAssignable>
{
    static T copy(const T& from)
    {
        T to;
        to = from;
        return std::move(to);
    }
};

struct FileList
{
    FileList(const char *pat, int lr, int rr)
        : pattern(pat), lr(lr), rr(rr) {}

    class iterator
    {
    public:
        iterator(const FileList *parent, int pos) : parent(parent), pos(pos) {}
        std::string operator*() const
        {
            if(pos > parent->rr)
                return "";

            char buf[128];  // hope this is enough, not safe though
            sprintf(buf, parent->pattern, pos);
            return std::string(buf);
        }

        iterator& operator++()
        {
            pos++;
            return *this;
        }

        iterator operator++(int)
        {
            iterator ret = *this;
            pos++;
            return ret;
        }

        bool operator==(const iterator& itr) const { return parent == itr.parent && pos == itr.pos; }

        bool operator!=(const iterator& itr) const { return !operator==(itr); }

    private:
        const FileList *parent;
        int pos;
    };

    iterator begin() const
    {
        return iterator(this, lr);
    }

    iterator end() const
    {
        return iterator(this, rr + 1);
    }

    const char *pattern;
    int lr, rr;
};

#endif // UTILS_H_INCLUDED
