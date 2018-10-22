#ifndef IOUTILS_H
#define IOUTILS_H

#include <iostream>
#include <map>
#include <type_traits>
#include <string>
#include <cstring>
#include <limits>
#include "utils.h"


class BaseArchive {};

class IArchive : public BaseArchive
{
public:
    static const bool mode_out = false;
};

class OArchive : public BaseArchive
{
public:
    static const bool mode_out = true;
};

class TxtArchive : public BaseArchive
{
public:
    static const bool mode_binary = false;
};

class BinArchive : public BaseArchive
{
public:
    static const bool mode_binary = true;
};

class ITxtArchive : public IArchive, TxtArchive
{
public:
    ITxtArchive(std::istream& is) : is(&is) {}

    template <typename T, class = typename std::enable_if<is_primitive(T)>::type>
    ITxtArchive& operator>>(T& t)
    {
        is->operator>>(t);
        return *this;
    }

//    template <typename T>
//    ITxtArchive& operator&(T& t)
//    {
//        return operator>>(t);
//    }

    ITxtArchive& operator>>(char *s)
    {
        size_t sz;
        (*is) >> sz;
        is->get();  // consume @
        is->read(s, sz);
        is->get();  // consume separator
        return *this;
    }

    ITxtArchive& operator>>(std::string& s)
    {
        size_t sz;
        (*is) >> sz;
        is->get();

        s.resize(sz);

        is->read((char*)s.data(), sz);
        is->get();
        return *this;
    }

private:
    ITxtArchive(ITxtArchive&) = delete;
    ITxtArchive& operator=(ITxtArchive&) = delete;

    std::istream *is;
};

class IBinArchive : public IArchive, BinArchive
{
public:
    IBinArchive(std::istream& is) : is(&is) {}

    template <typename T, class = typename std::enable_if<is_primitive(T)>::type>
    IBinArchive& operator>>(T& t)
    {
        is->read((char*)&t, sizeof(t));
        return *this;
    }

    IBinArchive& operator>>(char *s)
    {
        is->getline(s, std::numeric_limits<size_t>::max(), 0);
        return *this;
    }

    IBinArchive& operator>>(std::string& s)
    {
        std::getline(*is, s, '\0');
        return *this;
    }

//    template <typename T>
//    IBinArchive& operator&(T& t)
//    {
//        return operator>>(t);
//    }

private:
    IBinArchive(IBinArchive&) = delete;
    IBinArchive& operator=(IBinArchive&) = delete;
    std::istream *is;
};

class OTxtArchive : public OArchive, TxtArchive
{
public:
    // currently delim should only by space characters, for the sake of easy reading in ITxtArchive
    struct SetDelim
    {
        SetDelim(char delim, bool restore = true)
            : delim(delim), restore(restore) {}
        char delim;
        bool restore;
    };

    OTxtArchive(std::ostream& os, char delim = ' ')
        : os(&os), delim(delim), require_delim(false), old_delim(0) {}

    ~OTxtArchive()
    {
        (*os) << "\n";  // always add new line as the last delim
    }

    template <typename T, class = typename std::enable_if<is_primitive(T)>::type>
    OTxtArchive& operator<<(const T& t)
    {
        output_with_delim(t);
        return *this;
    }

    OTxtArchive& operator<<(const char *s)
    {
        std::ostringstream oss;
        oss << strlen(s) << '@' << s;
        output_with_delim(oss.str());
        return *this;
    }

    OTxtArchive& operator<<(const std::string& s)
    {
        std::ostringstream oss;
        oss << s.size() << '@' << s;
        output_with_delim(oss.str());
        return *this;
    }

    OTxtArchive& operator<<(SetDelim dlm)
    {
        if(dlm.restore)
            old_delim = delim;
        delim = dlm.delim;
        return *this;
    }

    char get_delim()
    {
        char ret = delim;
        if(old_delim)  // need restore
        {
            delim = old_delim;
            old_delim = 0;
        }
        return ret;
    }

//    template <typename T>
//    OTxtArchive& operator&(T& t)
//    {
//        return operator<<(t);
//    }

private:
    OTxtArchive(OTxtArchive&) = delete;
    OTxtArchive& operator=(OTxtArchive&) = delete;

    template <typename T>
    void output_with_delim(T t)
    {
        if(require_delim)
        {
            (*os) << get_delim();
            require_delim = false;
        }

        (*os) << t;

        require_delim = true;
    }

    std::ostream *os;

    char delim;
    bool require_delim;
    char old_delim;
};

class OBinArchive : public OArchive, BinArchive
{
public:
    OBinArchive(std::ostream& os) : os(&os) {}

    template <typename T, class = typename std::enable_if<is_primitive(T)>::type>
    OBinArchive& operator<<(const T& t)
    {
        os->write((const char*)&t, sizeof(t));
        return *this;
    }

    OBinArchive& operator<<(const char *s)
    {
        os->write(s, strlen(s));
        os->write("\0", 1);
        return *this;
    }

    OBinArchive& operator<<(const std::string& s)
    {
        os->write(s.data(), s.size());
        os->write("\0", 1);
        return *this;
    }

//    template <typename T>
//    OBinArchive& operator&(const T& t)
//    {
//        return operator<<(t);
//    }

private:
    OBinArchive(OBinArchive&) = delete;
    OBinArchive& operator=(OBinArchive&) = delete;
    std::ostream *os;
};

template <class Archive, class T, class = typename std::enable_if<std::is_base_of<OArchive, typename std::remove_reference<Archive>::type>::value && !is_primitive(T), bool>::type>
Archive&& operator<<(Archive&& ar, const T& t)
{
    t.serialize(std::forward<Archive>(ar));
    return std::forward<Archive>(ar);
}

template <class Archive, class T, class = typename std::enable_if<std::is_base_of<IArchive, typename std::remove_reference<Archive>::type>::value && !is_primitive(T), bool>::type>
Archive&& operator>>(Archive&& ar, T& t)
{
    t.deserialize(std::forward<Archive>(ar));
    return std::forward<Archive>(ar);
}

/* implementation for containers */
//template <class Archive, class = typename std::enable_if<std::is_base_of<IArchive, typename std::remove_reference<Archive>::type>::value, bool>::type>
//Archive&& operator>> (Archive&& ar, std::string& s)
//{
//    return std::forward<Archive>(ar >> s);
//}

//template <class Archive, class = typename std::enable_if<std::is_base_of<OArchive, typename std::remove_reference<Archive>::type>::value, bool>::type>
//Archive&& operator<< (Archive&& ar, const std::string& s)
//{
//    return std::forward<Archive>(ar << s.c_str());
//}

template <class Archive, typename key_t, typename mapped_t, class = typename std::enable_if<std::is_base_of<IArchive, typename std::remove_reference<Archive>::type>::value, bool>::type>
Archive&& operator>> (Archive&& ar, std::map<key_t, mapped_t>& t)
{
    size_t sz;
    ar >> sz;
    for(size_t i = 0; i < sz; i++)
    {
        key_t key;
        mapped_t value;
        ar >> key >> value;
        t.emplace(std::move(key), std::move(value));
    }
    return std::forward<Archive>(ar);
}

template <class Archive, typename key_t, typename mapped_t, class = typename std::enable_if<std::is_base_of<OArchive, typename std::remove_reference<Archive>::type>::value, bool>::type>
Archive&& operator<< (Archive&& ar, const std::map<key_t, mapped_t>& t)
{
    ar << t.size();
    for(const auto& itr : t)
    {
        ar << itr.first << itr.second;
    }
    return std::forward<Archive>(ar);
}



#endif // IOUTILS_H
