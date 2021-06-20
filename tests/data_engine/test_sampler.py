from data_engine.sampler import HighLossSampler


class DataSource:
    def __init__(self, len):
        self._len = len

    def __len__(self):
        return self._len


def test_full_pass():
    sampler = HighLossSampler(":memory:", allowed_stale=None, cutoff=None, robust_std=None)
    sampler.datasource = DataSource(50)
    assert all(v1 == v2 for v1, v2 in zip(range(50), sampler))


def test_non_inference_stale_selection():
    sampler = HighLossSampler(":memory:", allowed_stale=3, cutoff=1, robust_std=None)
    sampler.datasource = DataSource(50)

    sampler.inference_mode = False

    tmp = 0
    for _ in sampler:
        tmp += 1
    assert tmp == 50

    tmp = 0
    for _ in sampler:
        tmp += 1
    assert tmp == 0

    tmp = 0
    for _ in sampler:
        tmp += 1
    assert tmp == 0

    tmp = 0
    for _ in sampler:
        tmp += 1
    assert tmp == 50


def test_inference_stale_selection():
    sampler = HighLossSampler(":memory:", allowed_stale=3, cutoff=1, robust_std=None)
    sampler.datasource = DataSource(50)

    sampler.inference_mode = False
    for _ in sampler:
        pass
    sampler.update_losses(list(range(50)))

    sampler.inference_mode = True

    tmp = 0
    for _ in sampler:
        tmp += 1
    assert tmp == 50


    tmp = 0
    for _ in sampler:
        tmp += 1
    assert tmp == 50


def test_non_inference_traditional_learning():
    sampler = HighLossSampler(":memory:", allowed_stale=None, cutoff=None, robust_std=None)
    sampler.datasource = DataSource(50)

    sampler.inference_mode = False

    tmp = 0
    for _ in sampler:
        tmp += 1
    assert tmp == 50

    tmp = 0
    for _ in sampler:
        tmp += 1
    assert tmp == 50
