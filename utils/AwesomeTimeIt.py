import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        print (f'---- {method.__name__} is about to start ----')
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print (f'---- {method.__name__} is done in {te-ts:.2f} seconds ----')
        return result
    return timed