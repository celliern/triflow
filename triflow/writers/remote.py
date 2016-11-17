#!/usr/bin/env python
# coding=utf8

import logging
from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager

from triflow.displays import simple_display, full_display
from triflow.misc import coroutine
from triflow.writers.datreant import (datreant_init, datreant_save,
                                      get_datreant_conf)

try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

logging.getLogger(__name__).addHandler(NullHandler())


@coroutine
def remote_step_writer(simul):
    class QueueManager(BaseManager):
        pass
    path_data, simul_name, compressed = get_datreant_conf(simul)

    server = simul.conf.get('remote.server', 'localhost')
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

    server = simul.conf.get('remote.server', 'localhost')
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
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

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
                logger.debug(msg)
                if msg[0] == 'init':
                    simul_id, [path_data, simul_name, pars] = msg[1:]

                    treant = datreant_init(path_data, simul_name, pars)
                    logger.info("initialize %s, writer set at %s" %
                                (simul_id,
                                 treant.abspath))
                    self.cache[simul_id] = treant
                if msg[0] == 'run':
                    simul_id, i, t, tosave = msg[1:]
                    logger.info("save %s, iter %i, time %f in %s" %
                                (simul_id,
                                 i, t, treant.abspath))
                    treant = self.cache[simul_id]
                    datreant_save(treant, i, t, tosave)
    queue = Queue()

    QueueManager.register('get_queue', callable=lambda: queue)
    logger.info('QueueManager registered')
    w = Worker(queue)
    w.start()
    logger.info('Worker initialized')
    local_manager = QueueManager(address=('', port),
                                 authkey=b'triflow')
    server = local_manager.get_server()
    logger.info('starting server...')
    server.serve_forever()


remote_step_writer.writer_type = 'remote'
remote_steps_writer.writer_type = 'remote'
