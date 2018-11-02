#ifndef DEFS_H
#define DEFS_H

enum DUPMODE
{
    DUP_IGNORE = 0,  // do not insert when duplicated
    DUP_OVERWRITE,   // overwrite duplicated
    DUP_WARN,        // throw if duplicated
};

#endif // DEFS_H
