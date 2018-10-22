#ifndef TYPES_H
#define TYPES_H

#include <string>

template <typename T>
struct type2name
{};

template <>
struct type2name<std::string>
{
    static const std::string name;
};

template <>
struct type2name<int>
{
    static const std::string name;
};

template <>
struct type2name<float>
{
    static const std::string name;
};

template <>
struct type2name<int64_t>
{
    static const std::string name;
};

#endif // TYPES_H
