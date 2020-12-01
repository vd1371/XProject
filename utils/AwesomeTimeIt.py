import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        print (f'---- {method.__name__} is about to start ----')
        result = method(*args, **kw)
        te = time.time()
        elapsed_time = ts - te
        try:
            # args[0] is supposed to be self
            args[0].log.info(f'---- {method.__name__} is done in {te-ts:.2f} seconds ----')
        except:
            pass
        print (f'---- {method.__name__} is done in {te-ts:.2f} seconds ----')

        return result
    return timed