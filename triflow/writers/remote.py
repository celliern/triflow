#!/usr/bin/env python
# coding=utf8

import logging
from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager
from threading import Thread

import click

from triflow.displays import full_display, simple_display
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
def queue_async(queue):
    while True:
        data = yield
        Thread(target=queue.put, args=(data,)).start()


def init_remote(simul):
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
    send_remote = queue_async(queue)
    send_remote.send(('init', simul.id, [path_data, simul_name, simul.pars]))
    return queue, send_remote


def remote_step_writer(simul):
    remote_queue, send_remote = init_remote(simul)
    display = simple_display(simul)
    for t, field in display:
        t, field = next(display)
        tosave = {name: field[name]
                  for name
                  in simul.solver.fields}
        tosave['x'] = simul.x
        send_remote.send(('run', simul.id, simul.i, simul.t, tosave))
        yield


def remote_steps_writer(simul):
    remote_queue, send_remote = init_remote(simul)
    display = full_display(simul)
    for t, field in display:
        tosave = {name: field[name]
                  for name
                  in simul.solver.fields}
        tosave['t'] = t
        tosave['x'] = simul.x
        send_remote.send(('run', simul.id, simul.i, simul.t, tosave))
        yield


remote_step_writer.writer_type = 'remote'
remote_steps_writer.writer_type = 'remote'


@click.command()
@click.option('-p', '--port', default=50000, help='port number.')
@click.option('--debug-level', 'debug',
              default='INFO', help='verbosity level.')
def datreant_server_writer(port, debug):
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(debug)

    class QueueManager(BaseManager):
        pass

    class Worker(Process):
        def __init__(self, q, cache):
            self.q = q
            self.working = True
            self.cache = cache
            super(Worker, self).__init__()

        def run(self):
            while self.working:
                try:
                    msg = self.q.get()
                    logging.debug(self.cache)
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
                except KeyError:
                    logging.error('KEYERROR')
    queue = Queue()

    QueueManager.register('get_queue', callable=lambda: queue)
    QueueManager.register('register_writer', callable=None)
    logger.info('Manager registered')
    w = Worker(queue)
    w.start()
    logger.info('Worker initialized')
    local_manager = QueueManager(address=('', port),
                                 authkey=b'triflow')
    server = local_manager.get_server()
    logger.info('starting server...')
    server.serve_forever()
