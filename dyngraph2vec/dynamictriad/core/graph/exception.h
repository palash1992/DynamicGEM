#ifndef EXCEPTION_H_INCLUDE
#define EXCEPTION_H_INCLUDE

#include <stdexcept>
#include <exception>
#include <cerrno>
#include <cstring>
#include <sstream>

template <typename T>
class DuplicateKeyException : public std::runtime_error
{
public:
    DuplicateKeyException(const std::string& where, const std::string& msg, T key)
        : std::runtime_error(where + ": " + msg + " " + static_cast<std::ostringstream&>(std::ostringstream() << key).str()), 
            key(key), msg(msg), where(where) {}
    T key;
    std::string msg;
    std::string where;
};

template <typename T>
class InvalidKeyException : public std::runtime_error
{
public:
    InvalidKeyException(const std::string& where, const std::string& msg, T key)
        : std::runtime_error(where + ": " + msg + " " + static_cast<std::ostringstream&>(std::ostringstream() << key).str()),
            key(key), msg(msg), where(where) {}

    T key;
    std::string msg;
    std::string where;
};

// this class is not thread safe due to usage of errno
class IOException : public std::runtime_error
{
public:
    IOException(const std::string& where, const std::string& msg)
        : std::runtime_error(where + ": " + msg + ": " + strerr),
          msg(msg), where(where), strerr(strerror(errno)) {}
    std::string msg;
    std::string where;
    std::string strerr;
};

class InvalidFormatException : public std::runtime_error
{
public:
    InvalidFormatException(const std::string& where, const std::string& msg)
        : std::runtime_error(where + ": " + msg + ": " + strerr),
          msg(msg), where(where), strerr(strerror(errno)) {}
    std::string msg;
    std::string where;
    std::string strerr;
};

class NotImplementedException : public std::logic_error
{
public:
    NotImplementedException(const std::string& where, const std::string& msg)
        : std::logic_error(funcname + ": " + msg), msg(msg), funcname(where) {}
    std::string msg;
    std::string funcname;
};

class NoSuchFieldException : public std::logic_error
{
public:
    NoSuchFieldException(const std::string& where, const std::string& msg, const std::string fieldname)
        : std::logic_error(where + ": " + msg + ": " + fieldname),
          msg(msg), where(where), fieldname(fieldname)
    {}
    std::string msg;
    std::string where;
    std::string fieldname;
};

#endif // EXCEPTION_H_INCLUDE
