from threading import Thread
from queue import Queue
# 定义线程生成器
class ThreadedGenerator(object):
    #初始化
    def __init__(self, iterator,#遍历器
                 sentinel=object(),# 哨兵 确保传的对象是有效的
                 queue_maxsize=0,#队列最大的默认值
                 daemon=False):
        self._iterator = iterator
        self._sentinel = sentinel
        self._queue = Queue(maxsize=queue_maxsize)
        self._thread = Thread(
            name=repr(iterator),
            target=self._run
        )
        self._thread.daemon = daemon
        self._started = False

    def __repr__(self):
        return 'ThreadedGenerator({!r})'.format(self._iterator)

    def _run(self):
        try:
            for value in self._iterator:
                if not self._started:
                    return
                self._queue.put(value)
        finally:
            self._queue.put(self._sentinel)

    def close(self):
        self._started = False
        try:
            while True:
                self._queue.get(timeout=30)
        #         捕获键盘输入的异常
        except KeyboardInterrupt as e:
            raise e
        except:
            pass

    def __iter__(self):
        self._started = True
        self._thread.start()
        for value in iter(self._queue.get, self._sentinel):
            yield value
        self._thread.join()
        self._started = False

    def __next__(self):
        if not self._started:
            self._started = True
            self._thread.start()
        value = self._queue.get(timeout=30)
        if value == self._sentinel:
            raise StopIteration()
        return value

def test():

    def gene():
        i = 0
        while True:
            yield i
            i += 1

    t = gene()
    test = ThreadedGenerator(t)

    for _ in range(10):
        print(next(test))

    test.close()

if __name__ == '__main__':
    test()

