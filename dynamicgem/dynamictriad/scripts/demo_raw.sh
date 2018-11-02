pushd () {
    command pushd "$@" > /dev/null
}

popd () {
    command popd "$@" > /dev/null
}

set -e

MAINPATH=$(dirname $(readlink -f $0))/..
export PYTHONPATH=$MAINPATH
echo setting \$PYTHONPATH to $PYTHONPATH
echo entering directory $MAINPATH
pushd $MAINPATH

if [ -e output ]; then
    if [ -f output/.dynamic_triad ]; then
        rm -rf output
    else
        echo file/directory $MAINPATH/output already exists, please remove it before running the demo 1>&2
        popd
        exit 1
    fi
fi

mkdir -p output
touch output/.dynamic_triad
python . -I 10 -d data/academic_toy.pickle -n 15 -K 48 -l 4 -s 2 -o output --beta-smooth 1 --beta-triad 1 --datasetmod core.dataset.citation -m 1980 --cachefn /tmp/academic_raw -b 5000
# we have to use a different cache file because the file name and indexing are different between data/academic_toy and data/academic_toy.pickle,
# though they are actually the same dataset
python scripts/stdtests.py -f output -d data/academic_toy.pickle -m 1980 -s 4 -l 2 -n 15 -t all --datasetmod core.dataset.citation --cachefn /tmp/academic_raw

popd
