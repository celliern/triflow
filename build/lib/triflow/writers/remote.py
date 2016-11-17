#!/usr/bin/env python
# coding=utf8

import logging
from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager

from triflow.displays import simple_display_with_id
from triflow.misc import coroutine
from triflow.writers.datreant import get_datreant_conf, datreant_init


@coroutine
def remote_writer(simul):
    class QueueManager(BaseManager):
        pass
    path_data, simul_name, simul.pars = get_datreant_conf(simul)

    server = simul.conf.get('remote.path', 'localhost')
    port = simul.conf.get('remote.port', 50000)
    QueueManager.register('get_queue')
    distant_manager = QueueManager(address=(server,
                                            port),
                                   authkey=b'triflow')
    distant_manager.connect()
    queue = distant_manager.get_queue()
    queue.put(('init', simul.id,
               [path_data, simul_name, simul.pars]))
    display = simple_display_with_id(simul)
    while True:
        simul = yield
        queue.put(('run', simul.id, display.send(simul)))


def datreant_server_writer(port=50000):
    class QueueManager(BaseManager):
        pass

    class Worker(Process):
        def __init__(self, q):
            self.q = q
            self.working = True
            self.cache = {}
            super(Worker, self).__init__()

        def run(self):
            while self.working:
                msg = self.q.get()
                print(msg)
    queue = Queue()

    w = Worker(queue)
    w.start()

    QueueManager.register('get_queue', callable=lambda: queue)
    local_manager = QueueManager(address=('', port),
                                 authkey=b'triflow')
    server = local_manager.get_server()
    server.serve_forever()
