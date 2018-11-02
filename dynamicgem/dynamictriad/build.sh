pushd () {
    command pushd "$@" > /dev/null
}

popd () {
    command popd "$@" > /dev/null
}

set -e

filedir=$(dirname $(readlink -f $0))
echo entering $filedir
pushd $filedir

ask() {
    local ret
    echo -n "$1 (default: $2, use a space ' ' to leave it empty) " 1>&2
    read ret
    if [ -z "$ret" ]; then
        ret=$2
    elif [ "$ret" == " " ]; then
        ret=""
    fi
    echo $ret
}

echo "You may need to specify some environments before building"

pylib=$(python -c "from distutils.sysconfig import get_config_var; print('{}/{}'.format(get_config_var('LIBDIR'), get_config_var('INSTSONAME')))")
pylib=$(ask "PYTHON_LIBRARY?" $pylib)
export PYTHON_LIBRARY=$pylib

pyinc=$(python -c "from distutils.sysconfig import get_config_var; print(get_config_var('INCLUDEPY'))")
pyinc=$(ask "PYTHON_INCLUDE_DIR?" $pyinc)
export PYTHON_INCLUDE_DIR=$pyinc

eigeninc=$(ask "EIGEN3_INCLUDE_DIR?" /usr/include)
export EIGEN3_INCLUDE_DIR=$eigeninc

boostroot=$(ask "BOOST_ROOT?" "")
export BOOST_ROOT=$boostroot

boost_pylib=$(ask "name for boost_python library? (useful when boost_python cannot be detected by cmake)" "boost_python")
export BOOST_PYTHON_LIBNAME=$boost_pylib

echo building mygraph module ...
rm -rf core/mygraph-build
mkdir -p core/mygraph-build
pushd core/mygraph-build
cmake ../graph
make && ln -sf mygraph-build/mygraph.so ../mygraph.so 
popd

echo building c extensions for dynamic triad ...
pushd core/algorithm
rm -rf build && mkdir build && cd build
cmake ..
make && make install
popd

popd
