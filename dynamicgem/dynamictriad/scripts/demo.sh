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
rm -rf data/academic_toy
mkdir -p data/academic_toy
python scripts/academic2adjlist.py data/academic_toy.pickle data/academic_toy
python . -I 10 -d data/academic_toy -n 15 -K 48 -l 4 -s 2 -o output --beta-smooth 1 --beta-triad 1 --cachefn /tmp/academic -b 5000
#python . -I 10 -d data/academic_toy -n 15 -K 48 -l 1 -s 1 -o output --beta-smooth 1 --beta-triad 1 --cachefn /tmp/academic -b 5000
# we have to use a different cache file because the file name and indexing are different between data/academic_toy and data/academic_toy.pickle,
# though they are actually the same dataset
python scripts/stdtests.py -f output -d data/academic_toy.pickle -m 1980 -s 4 -l 2 -n 15 -t all --datasetmod core.dataset.citation --cachefn /tmp/academic_raw

popd
