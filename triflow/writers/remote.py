#!/usr/bin/env python
# coding=utf8

import logging
from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager

from triflow.displays import simple_display, full_display
from triflow.misc import coroutine
from triflow.writers.datreant import (datreant_init, datreant_save,
                                      get_datreant_conf)


@coroutine
def remote_step_writer(simul):
    class QueueManager(BaseManager):
        pass
    path_data, simul_name, compressed = get_datreant_conf(simul)

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
    display = simple_display(simul)
    while True:
        simul = yield
        t, field = display.send(simul)
        tosave = {name: field[name]
                  for name
                  in simul.solver.fields}
        tosave['x'] = simul.x
        queue.put(('run', simul.id, simul.i, simul.t, tosave))


@coroutine
def remote_steps_writer(simul):
    class QueueManager(BaseManager):
        pass
    path_data, simul_name, compressed = get_datreant_conf(simul)

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
    display = full_display(simul)
    while True:
        simul = yield
        t, field = display.send(simul)
        tosave = {name: field[name]
                  for name
                  in simul.solver.fields}
        tosave['t'] = t
        tosave['x'] = simul.x
        queue.put(('run', simul.id, simul.i, simul.t, tosave))


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
                logging.debug(msg)
                if msg[0] == 'init':
                    simul_id, [path_data, simul_name, pars] = msg[1:]
                    treant = datreant_init(path_data, simul_name, pars)
                    self.cache[simul_id] = treant
                if msg[0] == 'run':
                    simul_id, i, t, tosave = msg[1:]
                    treant = self.cache[simul_id]
                    datreant_save(treant, i, t, tosave)
    queue = Queue()

    QueueManager.register('get_queue', callable=lambda: queue)
    logging.info('QueueManager registered')
    w = Worker(queue)
    w.start()
    logging.info('Worker initialized')
    local_manager = QueueManager(address=('', port),
                                 authkey=b'triflow')
    server = local_manager.get_server()
    logging.info('starting server...')
    server.serve_forever()


remote_step_writer.writer_type = 'remote'
remote_steps_writer.writer_type = 'remote'
