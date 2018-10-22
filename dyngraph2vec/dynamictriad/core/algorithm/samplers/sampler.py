from __future__ import print_function


class Sampler(object):
    def pretrain_begin(self, begin, end):
        pass

    def pretrain_end(self):
        pass

    def pretrain_begin_iteration(self):
        pass

    def pretrain_end_iteration(self):
        pass

    def online_begin(self, begin, end):
        pass

    def online_end(self):
        pass

    def online_begin_iteration(self):
        pass

    def online_end_iteration(self):
        pass

    def make_pretrain_input(self, batch):
        pass

    def make_online_input(self, batch):
        pass

    def shuffle_sample(self):
        pass

    def batches(self, batchsize):
        raise NotImplementedError()

    def sample_size(self):
        raise NotImplementedError()
