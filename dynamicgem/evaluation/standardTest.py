from __future__ import print_function

import sys
import os

try:
    import core
except ImportError:
    mainpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Error while importing core modules, try adding {} to python path".format(mainpath))
    sys.path.append(mainpath)

import numpy as np
import argparse
import random
try:
    from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
except ImportError:
    from sklearn.cross_validation import cross_val_score, KFold, StratifiedKFold
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from six.moves import cPickle
import importlib
from sklearn.linear_model import LogisticRegression
from os.path import isfile


class standardTest(object):
    def __init__(self, ds, emb, **kwargs):
        self.scale = kwargs.get('scale', None)
        self.debug = kwargs.get('debug', False)
        self.clname = kwargs.get('classifier', 'lr')
        self.ds = ds
        self.emb = emb
        
        self.clf = self.__make_classifier()

    def __make_classifier(self):
        class_weight = 'balanced'
        
        if self.clname == 'svm':
            return svm.SVC(kernel='linear', class_weight=class_weight)
        elif self.clname == 'lr':
            return LogisticRegression(class_weight=class_weight)
        else:
            raise NotImplementedError()

    def __classify(self, feat, lbs):
        sm = None

        poscnt, negcnt = np.sum(lbs == 1), np.sum(lbs == -1)
        print("classifying with pos:neg = {}:{}".format(poscnt, negcnt))

        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True)
            parts = cv.split(feat, lbs)
        except TypeError:
            cv = StratifiedKFold(lbs, n_folds=5, shuffle=True)
            parts = cv

        f1, prec, rec, acc = [], [], [], []
        for tr, te in parts:
            if sm is not None:
                x, y = sm.fit_sample(feat[tr], lbs[tr])
                # x, y = feat[tr], lbs[tr]
            else:
                x, y = feat[tr], lbs[tr]
            model = self.clf.fit(x, y)
            p = model.predict(feat[te])
            if self.debug:
                print("results:", p, lbs[te])
            f1.append(f1_score(lbs[te], p))
            prec.append(precision_score(lbs[te], p))
            rec.append(recall_score(lbs[te], p))
            acc.append(accuracy_score(lbs[te], p))
        return prec, rec, f1, acc

    # each element in feat is a list of feature vectors
    def __classify_multifeat(self, feat, lbs, decision_func=None):
        if decision_func is None:
            # vote
            def decision_func(model, feat):
                res = []
                for f in feat:
                    # TODO: note that max because in most cases, we identify a positive sample
                    # TODO: as long as it is identified positive in any time step
                    res.append(max(model.decision_function(f)))
                return res

        sm = None

        poscnt, negcnt = np.sum(lbs == 1), np.sum(lbs == -1)
        print("classifying with pos:neg = {}:{}".format(poscnt, negcnt))

        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True)
            parts = cv.split(feat, lbs)
        except TypeError:
            cv = StratifiedKFold(lbs, n_folds=5, shuffle=True)
            parts = cv

        f1, prec, rec, acc = [], [], [], []
        for tr, te in parts:
            x, y = [], []
            for f, l in zip(feat[tr], lbs[tr]):
                x.extend(f)
                y.extend([l] * len(f))
 
            if sm is not None:
                x, y = sm.fit_sample(x, y)
                # x, y = feat[tr], lbs[tr]
            
            model = self.clf.fit(x, y)
            score = decision_func(model, feat[te])
            p = np.sign(score)
            # p = model.predict(feat[te])
            if self.debug:
                print("results:", p, lbs[te])
            f1.append(f1_score(lbs[te], p))
            prec.append(precision_score(lbs[te], p))
            rec.append(recall_score(lbs[te], p))
            acc.append(accuracy_score(lbs[te], p))

        return prec, rec, f1, acc

    class ResultPresenter(object):
        def __init__(self):
            self.f1, self.prec, self.rec, self.acc = [], [], [], []

        def add_result(self, res):
            self.prec.extend(res[0])
            self.rec.extend(res[1])
            self.f1.extend(res[2])
            self.acc.extend(res[3])

        def show_result(self):
            print("precision mean: {} std: {}".format(np.mean(self.prec), np.std(self.prec)))
            print("recall mean: {} std: {}".format(np.mean(self.rec), np.std(self.rec)))
            print("f1 mean: {} std: {}".format(np.mean(self.f1), np.std(self.f1)))
            print("accuracy mean: {} std: {}".format(np.mean(self.acc), np.std(self.acc)))

    # we predict time t from features at time t-intv
    def node_classify(self, ds, emb, scale=None, intv=0, repeat=1):
        samp, lbs = ds.sample_test_data('node_classify', ds.localstep + intv, ds.localstep + ds.nsteps)
        if len(samp) == 0:
            raise RuntimeError("No valid sample found in dataset {}".format(ds.name))
        posidx = np.nonzero(lbs == 1)[0]
        negidx = np.nonzero(lbs == -1)[0]

        rp = StdTests.ResultPresenter()
        for time in range(repeat):
            # generate random neg sample
            if scale is None:
                curnegidx = negidx
            else:
                curnegidx = random.sample(negidx, scale * len(posidx))
            cursamp = np.concatenate((samp[posidx], samp[curnegidx]), axis=0)
            curlb = np.concatenate((lbs[posidx], lbs[curnegidx]), axis=0)
            # predict current
            feat = emb[cursamp[:, 0] - intv - ds.localstep, cursamp[:, 1]]

            res = self.__classify(feat, curlb)
            rp.add_result(res)
        rp.show_result()

    def all_link_predict(self, ds, emb, intv=0, repeat=1):
        samp, lbs = ds.sample_test_data('link_reconstruction', ds.localstep + intv, ds.localstep + ds.nsteps)

        # TODO: different feature generation method might be used here
        feat = np.fabs(emb[samp[:, 0] - intv - ds.localstep, samp[:, 1]] - emb[samp[:, 0] - intv - ds.localstep, samp[:, 2]])
        print("feature shape {}".format(feat.shape))
        rp = StdTests.ResultPresenter()
        for i in range(repeat):
            res = self.__classify(feat, lbs)
            rp.add_result(res)
        rp.show_result()

    def changed_link_predict(self, ds, emb, intv=0, repeat=1):
        samp, lbs = ds.sample_test_data('changed_link_classify', ds.localstep + intv, ds.localstep + ds.nsteps)

        feat = np.fabs(emb[samp[:, 0] - intv - ds.localstep, samp[:, 1]] - emb[samp[:, 0] - intv - ds.localstep, samp[:, 2]])
        print("feature shape {}".format(feat.shape))
        rp = StdTests.ResultPresenter()
        for i in range(repeat):
            res = self.__classify(feat, lbs)
            rp.add_result(res)
        rp.show_result()

    def run_tests(self, tests, repeat=1):
        if 'node_classify' in tests or 'node_classify_all' in tests or 'all' in tests:
            try:
                print("task: node label predict from current time step")
                self.node_classify(self.ds, self.emb, scale=self.scale, intv=0, repeat=repeat)
            except (AttributeError, NotImplementedError, RuntimeError) as e:
                print("skipping node_classify task", e.message)

        if 'node_predict' in tests or 'node_classify_all' in tests or 'all' in tests:
            try:
                print("task: node label predict from previous time step")
                self.node_classify(self.ds, self.emb, scale=self.scale, intv=1, repeat=repeat)
            except (AttributeError, NotImplementedError, RuntimeError) as e:
                print("skipping node_predict task", e.message)

        if 'link_classify' in tests or 'link_classify_all' in tests or 'all' in tests:
            try:
                print("task: link prediction from current time step")
                self.all_link_predict(self.ds, self.emb, intv=0, repeat=repeat)
            except (AttributeError, NotImplementedError, RuntimeError) as e:
                print("skipping link_classify task", e.message)
        
        if 'link_predict' in tests or 'link_classify_all' in tests or 'all' in tests:
            try:
                print("task: link prediction from previous time step")
                self.all_link_predict(self.ds, self.emb, intv=1, repeat=repeat)
            except (AttributeError, NotImplementedError, RuntimeError) as e:
                print("skipping link_predict task", e.message) 

        if 'changed_link_classify' in tests or 'changed_link_classify_all' in tests or 'all' in tests:
            try:
                print("task: changed link prediction from current time step")
                self.changed_link_predict(self.ds, self.emb, intv=0, repeat=repeat)
            except (AttributeError, NotImplementedError) as e:
                print("skipping changed_link_classify task", e.message)
        
        if 'changed_link_predict' in tests or 'changed_link_classify_all' in tests or 'all' in tests:
            try:
                print("task: changed link prediction from previous time step")
                self.changed_link_predict(self.ds, self.emb, intv=1, repeat=repeat)
            except (AttributeError, NotImplementedError) as e:
                print("skipping changed_link_predict task", e.message)


if __name__ == '__main__':
    def load_datamod(modname):
        mod = importlib.import_module(modname)
        return getattr(mod, 'Dataset')


    def load_or_update_cache(ds, cacheprefix):
        if isfile(cacheprefix + '.cache.args'):
            args = cPickle.load(open(cacheprefix + '.cache.args', 'r'))
            try:
                ds.load_cache(args, lambda: cPickle.load(open(cacheprefix + '.cache', 'r')))
                print("Data loaded from cache file {}".format(cacheprefix + '.cache'))
                return
            except (ValueError, EOFError) as e:
                print("Failed to load cache file {}: {}".format(cacheprefix + '.cache', e.message))

        # update cache when load failed
        print("updating cache for prefix {}".format(cacheprefix))
        ar, args = ds.cache()
        cPickle.dump(args, open(cacheprefix + '.cache.args', 'w'))
        cPickle.dump(ar, open(cacheprefix + '.cache', 'w'))
        print("cache file {} updated".format(cacheprefix))


    def load_embedding(fn, vs):
        data = open(fn, 'r').read().rstrip('\n').split('\n')
        emb = {}
        for line in data:
            fields = line.split()
            emb[fields[0]] = [float(e) for e in fields[1:]]
        # it is possible that the output order differs from :param vs: given different node_type,
        # so we have to reorder the embedding according to :param vs:
        emb = [emb[str(v)] for v in vs]

        return np.vstack(emb)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--embdir', type=str, required=True, help='directory storing embedding outputs')
    parser.add_argument('-t', '--test', type=str, nargs='+', default='all',
                        help='type of test, (node_classify, node_predict, link_classify, link_predict, '
                             'changed_link_classify, changed_link_predict, all)')
    parser.add_argument('-m', '--starttime', type=str, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--datasetmod', type=str, default='core.dataset.edgelist', help=argparse.SUPPRESS)
    parser.add_argument('-d', '--datafn', type=str, required=True, help='data file path')
    parser.add_argument('-s', '--stepsize', type=int, required=True, help='step size')
    parser.add_argument('-l', '--stepstride', type=int, required=True, help='step stride')
    parser.add_argument('-n', '--nsteps', type=int, required=True, help='#steps to test')
    parser.add_argument('--classifier', type=str, default='lr', help='lr, svm')
    parser.add_argument('--repeat', type=int, default=1, help='number of times to repeat experiment')
    parser.add_argument('--cachefn', type=str, default=None, help='name of dataset cache file')
    # parser.add_argument('--dataname', type=str, default=None, help='name for the current data file')
    args = parser.parse_args()

    args.debug = False
    args.scale = 1

    print("running with options: ", args.__dict__)

    Dataset = load_datamod(args.datasetmod)

    # although arguments can be found in cache files, we require them to be specified explicitly
    # so that it may not be a problem to leave the old cache not removed
    ds = Dataset(args.datafn, args.starttime, args.nsteps,
                 stepsize=args.stepsize, stepstride=args.stepstride)
    if args.cachefn is not None:
        load_or_update_cache(ds, args.cachefn)

    emb = []
    for i in range(ds.localstep, ds.localstep + args.nsteps):
        fn = "{}/{}.out".format(args.embdir, i - ds.localstep)
        emb.append(load_embedding(fn, ds.mygraphs['any'].vertices()))
    emb = np.stack(emb, axis=0)
    print("embedding shape is {}".format(emb[0].shape))

    tester = StdTests(ds, emb, scale=args.scale, classifier=args.classifier, debug=args.debug)
    tester.run_tests(args.test, repeat=args.repeat)
