from __future__ import print_function

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


def main():
    import sys
    from six.moves import cPickle
    import argparse
    import importlib
    import time
    from os.path import isfile
    import dataset.dataset_utils as du
    import algorithm.embutils as eu

    # random.seed(977)  # for reproducability
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-I', '--niters', type=int, help="number of optimization iterations", default=10)
    parser.add_argument('-m', '--starttime', type=str, help=argparse.SUPPRESS, default=0)
    parser.add_argument('-d', '--datafile', type=str, required=True, help='input directory name')
    parser.add_argument('-b', '--batchsize', type=int, help="batchsize for training", default=5000)
    parser.add_argument('-n', '--nsteps', type=int, help="number of time steps", required=True)
    parser.add_argument('-K', '--embdim', type=int, help="number of embedding dimensions", default=48)
    parser.add_argument('-l', '--stepsize', type=int, help="size of of a time steps", default=1)
    parser.add_argument('-s', '--stepstride', type=int, help="interval between two time steps", default=1)
    parser.add_argument('-o', '--outdir', type=str, required=True, help="output directory name")
    parser.add_argument('--cachefn', type=str, help="prefix for data cache files", default=None)
    parser.add_argument('--lr', type=float, help="initial learning rate", default=0.1)
    parser.add_argument('--beta-smooth', type=float, default=0.1, help="coefficients for smooth component")
    parser.add_argument('--beta-triad', type=float, default=0.1, help="coefficients for triad component")
    parser.add_argument('--negdup', type=int, help="neg/pos ratio during sampling", default=1)
    parser.add_argument('--datasetmod', type=str, help=argparse.SUPPRESS, default='core.dataset.adjlist',
                        # help='module name for dataset loading',
                        )
    # parser.add_argument('--dataname', type=str, default=None, help='name for the current data file')
    parser.add_argument('--validation', type=str, default='link_reconstruction',
                        help=', '.join(list(sorted(set(du.TestSampler.tasks) & set(eu.Validator.tasks)))))
    args = parser.parse_args()
    args.beta = [args.beta_smooth, args.beta_triad]
    # some fixed arguments in published code
    args.pretrain_size = args.nsteps
    args.trainmod = 'core.algorithm.dynamic_triad'
    args.sampling_args = {}

    if args.validation not in du.TestSampler.tasks:
        raise NotImplementedError("Validation task {} not supported in TestSampler".format(args.validation))
    if args.validation not in eu.Validator.tasks:
        raise NotImplementedError("Validation task {} not supported in Validator".format(args.validation))

    print("running with options: ", args.__dict__)

    def load_trainmod(modname):
        mod = importlib.import_module(modname)
        return getattr(mod, 'Model')

    def load_datamod(modname):
        mod = importlib.import_module(modname)
        return getattr(mod, 'Dataset')

    def load_or_update_cache(ds, cachefn):
        if cachefn is None:
            return
        cachefn += '.cache' 
        if isfile(cachefn + '.args'):
            args = cPickle.load(open(cachefn + '.args', 'r'))
            try:
                ds.load_cache(args, lambda: cPickle.load(open(cachefn, 'r')))
                print("Data loaded from cache file {}".format(cachefn))
                return
            except (ValueError, EOFError) as e:
                print("Failed to load cache file {}: {}".format(cachefn, e.message))

        # update cache
        print("updating cache file for prefix {}".format(cachefn))
        ar, args = ds.cache()
        cPickle.dump(args, open(cachefn + '.args', 'w'))
        cPickle.dump(ar, open(cachefn, 'w'))
        print("cache file {} updated".format(cachefn))

    def export(vertices, data, outdir):
        for i in range(len(data)):
            assert len(vertices) == len(data[i]), (len(vertices), len(data[i]))
            fn = "{}/{}.out".format(outdir, i)
            fh = open(fn, 'w')
            for j in range(len(vertices)):
                print("{} {}".format(vertices[j], ' '.join(["{:.3f}".format(d) for d in data[i][j]])), file=fh)
            fh.close()

    TrainModel = load_trainmod(args.trainmod)
    Dataset = load_datamod(args.datasetmod)

    ds = Dataset(args.datafile, args.starttime, args.nsteps, stepsize=args.stepsize, stepstride=args.stepstride)
    load_or_update_cache(ds, args.cachefn)
    # dsargs = {'datafile': args.datafile, 'starttime': args.starttime, 'nsteps': args.nsteps,
    #           'stepsize': args.stepsize, 'stepstride': args.stepstride, 'datasetmod': args.datasetmod}
    tm = TrainModel(ds, pretrain_size=args.pretrain_size, embdim=args.embdim, beta=args.beta,
                    lr=args.lr, batchsize=args.batchsize, sampling_args=args.sampling_args)

    edgecnt = [g.num_edges() for g in ds.gtgraphs]
    k_edgecnt = sum(edgecnt[:args.pretrain_size])
    print("{} edges in pretraining graphs".format(k_edgecnt))

    if args.pretrain_size > 0:
        initstep = ds.time2step(args.starttime)
        tm.pretrain_begin(initstep, initstep + args.pretrain_size)

        print("generating validation set")
        validargs = tm.dataset.sample_test_data(args.validation, initstep, initstep + args.pretrain_size, size=10000)
        #print(validargs)
        print("{} validation samples generated".format(len(validargs[0])))

        max_val, max_idx, maxmodel = -1, 0, None

        # for early stopping
        start_time = time.time()
        scores = []
        for i in range(args.niters):
            tm.pretrain_begin_iteration()

            epoch_loss = 0
            for batidx, bat in enumerate(tm.batches(args.batchsize)):
                inputs = tm.make_pretrain_input(bat)
                l = tm.pretrain['lossfunc'](inputs)
                if isinstance(l, (list, tuple)):
                    l = l[0]
                epoch_loss += l
                print("\repoch {}: {:.0%} completed, cur loss: {:.3f}".format(i, float(batidx * args.batchsize)
                                                                              / tm.sample_size(), l.flat[0]), end='')
                sys.stdout.flush()
            tm.pretrain_end_iteration()

            print(" training completed, total loss {}".format(epoch_loss), end='')

            # without validation, the model exists only after I iterations
            if args.validation != 'none':
                val_score = tm.validate(args.validation, *validargs)

                if val_score > max_val:
                    max_val = val_score
                    max_idx = i
                    maxmodel = tm.save_model()
                print(", validation score {:.3f}".format(val_score))
            else:
                max_idx, max_val = i, epoch_loss
                # maxmodel is not saved here in order to save time
                print("")

            # checkpoint disabled
            # if i % 5 == 0:
            #     lastmodel = tm.save_model()
            #     if args.validation == 'none':
            #         maxmodel = lastmodel
            #
            #     tm.restore_model(maxmodel)  # restore parameters while preserving other info
            #     cPickle.dump([tm.archive(), dsargs, lastmodel], open(args.outdir, 'w'))
            #     tm.restore_model(lastmodel)

            if args.validation != 'none':
                scores.append(val_score)
                if max_val > 0 and i - max_idx > 5:
                    break

        print("best validation score at itr {}: {}".format(max_idx, max_val))
        print("{} seconds elapsed for pretraining".format(time.time() - start_time))
        #lastmodel = tm.save_model()  # for debug
        print("saving output to {}".format(args.outdir))
        tm.restore_model(maxmodel)
        tm.pretrain_end()
        export(tm.dataset.mygraphs['any'].vertices(), tm.export(), args.outdir)

    # online training disabled
    startstep = tm.dataset.time2step(args.starttime)
    for y in range(startstep + args.pretrain_size, startstep + args.nsteps):
        raise NotImplementedError()
