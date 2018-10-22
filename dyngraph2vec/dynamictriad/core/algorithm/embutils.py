from __future__ import print_function

import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from copy import deepcopy
try:
    from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
except ImportError:
    from sklearn.cross_validation import cross_val_score, KFold, StratifiedKFold
from core import utils
from core import gconfig as gconf


class WithData(object):
    @property
    def dataset(self):
        raise NotImplementedError()


class TrainFlow(utils.Archivable, WithData):
    def __init__(self, **flowargs):
        self.__arg_names = ['embdim', 'beta', 'trainmod', 'datasetmod']
        self.__args = {k: flowargs.get(k, None) for k in self.__arg_names}
        # self.__args = {'embdim': embdim, 'beta': beta, 'trainmod': trainmod, 'datasetmod': datasetmod}

        self._history = []

        self._sequence = utils.OffsetList(0, 0, [], managed=True)  # to make the sequence managed by OffsetList
        self._tagged = {}

        self.__training = False
        self.__curbegin = None
        self.__curend = None

    def __add_history(self, hist):
        if len(self._history) > 0:
            if self._history[-1][1] != hist[0]:
                raise RuntimeError("Expected to train from step {}, got {} instead"
                                   .format(self._history[-1][1], hist[1]))
            dsrg = [self.dataset.localstep, self.dataset.localstep + self.dataset.nsteps]
            if self._history[-1][1] == dsrg[0]:
                self._history.append(hist)
            else:
                if hist[0] >= dsrg[0] and hist[1] <= dsrg[1]:
                    self._history[-1][1] = hist[1]  # merge
                else:
                    raise RuntimeError("trying to train from {} to {} (excluded), out of dataset range [{}, {})"
                                       .format(hist[0], hist[1], dsrg[0], dsrg[1]))
        else:
            self._history.append(hist)

    def start_training(self, begin, end):
        self.__training = True
        self.__curbegin, self.__curend = begin, end
        if len(self._history) == 0:  # pretrain
            self._sequence = utils.OffsetList(begin, 0, [], managed=True)
        self.__add_history([begin, end])

    def stop_training(self):
        self.__training = False
        self.__curbegin, self.__curend = None, None

    @property
    def cur_train_begin(self):
        if not self.__training:
            raise RuntimeError("TrainFlow not in training mode")
        return self.__curbegin

    @property
    def cur_train_end(self):
        if not self.__training:
            raise RuntimeError("TrainFlow not in training mode")
        return self.__curend
    
    @property
    def init_train_begin(self):
        if len(self._history) == 0:
            raise RuntimeError("no training records found")
        return self._history[0][0]
        
    @property
    def init_train_end(self):
        if len(self._history) == 0:
            raise RuntimeError("no training records found")
        return self._history[0][1]

    @property
    def last_train_begin(self):
        if len(self._history) == 0:
            raise RuntimeError("no training records found")
        return self._history[-1][0]

    @property
    def last_train_end(self):
        if len(self._history) == 0:
            raise RuntimeError("no training records found")
        return self._history[-1][1]

    @property
    def flowargs(self):
        return self.__args

    @property
    def is_training(self):
        return self.__training

    def embeddings_at(self, step, allow_missing=False, default=None):
        try:
            return self._sequence[step]
        except KeyError:
            if allow_missing:
                return default
            else:
                raise RuntimeError("trying to access missing embedding at step {}".format(step))

    # samples: [(time, node1, node2, ...), ...]
    def make_features(self, samples):
        feat = []
        emb_cache = utils.KeyDefaultDict(lambda x: self.embeddings_at(x))
        for s in samples:
            feat.append(emb_cache[s[0]][s[1:]])
        feat = np.array(feat)
        if gconf.debug:
            print("features shape: {}".format(feat.shape))  # (sample_size, node_cnt, feat_dim)
        return feat

    def clear(self):
        self._history = []
        self._sequence = utils.OffsetList(0, 0, [], managed=True) 
        self._tagged = {}
        self.__training = False

    def slim_storage(self, keep_size):
        startstep = min(self._history[-1][1] - keep_size, self._history[-1][0])
        startstep = max(startstep, self._sequence.offset)
        self._sequence = utils.OffsetList(startstep, self._history[-1][1] - startstep, self._sequence[startstep:],
                                          copy=False, managed=True)

    def export(self):
        """
        exports the embedding vectors
        :return:
        """
        return list(self._sequence)

    def archive(self, name=None, copy=True):
        if name is None:
            prefix = 'TrainFlow'
        else:
            prefix = '{}_TrainFlow'.format(name)

        ar = super(TrainFlow, self).archive(name)
        ar['{}_args'.format(prefix)] = self.__args
        ar['{}_history'.format(prefix)] = self._history
        ar['{}_sequence'.format(prefix)] = [self._sequence.offset, self._sequence.length, list(self._sequence)]
        ar['{}_tagged'.format(prefix)] = self._tagged

        if copy:
            ar = deepcopy(ar)

        return ar

    def load_archive(self, ar, copy=True, name=None):
        if self.__training:
            raise RuntimeError("archive should be loaded before training starts")

        if name is None:
            prefix = 'TrainFlow'
        else:
            prefix = '{}_TrainFlow'.format(name)

        super(TrainFlow, self).load_archive(ar, copy=copy, name=name)
        self._sequence, self._tagged = utils.OffsetList(*ar['{}_sequence'.format(prefix)], copy=copy, managed=True), \
                                       ar['{}_tagged'.format(prefix)]

        self._history = ar['{}_history'.format(prefix)]
        print("[debug] train history: {}".format(self._history))

        self.__check_flowargs(ar['{}_args'.format(prefix)])

        if copy:
            self._history = deepcopy(self._history)

    def __check_flowargs(self, old_args):
        for n in old_args:
            if self.__args.get(n, None) is not None and self.__args[n] != old_args[n]:
                raise RuntimeError("Argument mismatch {}: {}(old) vs. {}(cur)".format(n, old_args[n], self.__args[n]))
            self.__args[n] = old_args[n]

    # checkpoints
    # this is different from archive system that only params related to latest train are considered
    def save_model(self, copy=True):
        model = [[], self._tagged]
        for i in range(self.cur_train_begin, self.cur_train_end):
            model[0].append(self.embeddings_at(i))
        if copy:
            model = deepcopy(model)
        return model

    def restore_model(self, model, begin=None, end=None, copy=True):
        if begin is None:
            begin = self.cur_train_begin
        if end is None:
            end = self.cur_train_end

        if len(model[0]) != end - begin:
            raise RuntimeError("trying to restore invalid model with length {} to range [{}, {})"
                               .format(len(model[0]), begin, end))
        if copy:
            model = deepcopy(model)

        self._tagged = model[1]
        for i in range(begin, end):
            self._sequence[i] = model[0][i - begin]


class TrainFlowView(TrainFlow, WithData):
    def __init__(self, **flowargs):
        super(TrainFlowView, self).__init__(**flowargs)

    def start_training(self, begin, end):
        raise NotImplementedError()

    def stop_training(self):
        raise NotImplementedError()

    @property
    def dataset(self):
        return None


# the sub class must implement make_features,
# TODO: this is almost the same as class StdTests, with slight differences such as this class
# TODO: focuses only on f1 score, etc, consider merging these two classes in the future
class Validator(object):
    tasks = 'link_prediction', 'link_reconstruction', 'node_classify', 'node_predict', 'none'

    @property
    def __task_handler(self):
        ret = defaultdict(lambda: self.__unknown)
        ret.update({'link_prediction': self._validate_link_reconstruction,
                    'link_reconstruction': self._validate_link_reconstruction,
                    'node_classify': self._validate_node_classify,
                    'node_predict': self._validate_node_classify,
                    'none': self.__none})
        return ret

    def _validate_link_reconstruction(self, samples, lbs):
        # cache = utils.KeyDefaultDict(lambda x: self.embeddings_at(x))
        # feat = []
        # for v in samples:
        #     emb = cache[v[0] - 1]
        #     # feat.append(np.concatenate((emb[v[1]], emb[v[2]]), axis=0))
        #     feat.append(np.abs(emb[v[1]] - emb[v[2]]))
        # feat = np.vstack(feat)
        feat = self.make_features(samples)
        feat = np.abs(feat[:, 0] - feat[:, 1])

        clf = LogisticRegression()
        try:
            cv = StratifiedKFold(lbs, n_folds=2, shuffle=True)
            parts = cv
        except TypeError:
            cv = StratifiedKFold(n_splits=2, shuffle=True)
            parts = cv.split(feat, lbs)

        val_score = []
        for tr, te in parts:
            model = clf.fit(feat[tr], lbs[tr])
            p = model.predict(feat[te])
            val_score.append(f1_score(lbs[te], p))
        return np.mean(val_score)

    def _validate_node_classify(self, samples, lbs):
        # note that the 1-st dimension of feat is for each node in each sample (time, node1, node2, ...)
        feat = self.make_features(samples)[:, 0]
        assert len(feat) == len(lbs)

        clf = LogisticRegression(class_weight='balanced')
        try:
            cv = StratifiedKFold(lbs, n_folds=2, shuffle=True)
            parts = cv
        except TypeError as e:
            cv = StratifiedKFold(n_splits=2, shuffle=True)
            parts = cv.split(feat, lbs)

        val_score = []
        for tr, te in parts:
            model = clf.fit(feat[tr], lbs[tr])
            p = model.predict(feat[te])
            val_score.append(f1_score(lbs[te], p))
        return np.mean(val_score)

    def __none(self, samples, lbs):
        return 0

    def __unknown(self, samples, lbs):
        raise NotImplementedError()

    def validate(self, task, samples, lbs):
        return self.__task_handler[task](samples, lbs)

# class Validation(Validator, ValidationSampler):
#     @property
#     def tasks(self):
#         return list(set(Validator.tasks.fget(self)).intersection(set(ValidationSampler.tasks.fget(self))))
