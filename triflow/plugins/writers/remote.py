#!/usr/bin/env python
# coding=utf8

import functools as ft
import logging
from multiprocessing import current_process
from multiprocessing.managers import BaseManager
from threading import Thread

import click
import numpy as np
from triflow.misc.misc import coroutine
from triflow.plugins.displays import simple_display
from triflow.plugins.writers.datreant import (datreant_append, datreant_init,
                                              datreant_save, get_datreant_conf)

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
    current_process().authkey = b'triflow'

    class QueueManager(BaseManager):
        pass
    path_data, simul_name, compressed = get_datreant_conf(simul)

    server = simul.conf.get('remote.server', 'localhost')
    port = simul.conf.get('remote.port', 50000)
    QueueManager.register('datreant_init')
    QueueManager.register('datreant_save')
    QueueManager.register('datreant_append')
    distant_manager = QueueManager(address=(server,
                                            port))
    distant_manager.connect()
    datreant_init = distant_manager.datreant_init
    datreant_save = distant_manager.datreant_save
    datreant_append = distant_manager.datreant_append
    return datreant_init, datreant_save, datreant_append


def remote_step_writer(simul):
    path_data, simul_name, compressed = get_datreant_conf(simul)
    datreant_init, datreant_save = init_remote(simul)[:2]
    datreant_init(path_data, simul_name, simul.pars)
    display = simple_display(simul)
    for t, field in display:
        tosave = {name: field[name]
                  for name
                  in simul.solver.fields}
        tosave['x'] = simul.x
        simul.simuloop.run_in_executor(simul.executor,
                                       ft.partial(
                                           datreant_save,
                                           path_data,
                                           simul_name,
                                           simul.i,
                                           t,
                                           tosave,
                                           compressed,
                                           simul.datalock))
        yield


def remote_steps_writer(simul):
    path_data, simul_name, compressed = get_datreant_conf(simul)
    datreant_init, datreant_save, datreant_append = init_remote(simul)
    datreant_init(path_data, simul_name, simul.pars)
    display = simple_display(simul)
    for t, field in display:
        tosave = {name: field[name]
                  for name
                  in simul.solver.fields}
        tosave['t'] = np.array([t])
        tosave['x'] = simul.x
        simul.simuloop.run_in_executor(simul.executor,
                                       ft.partial(
                                           datreant_append,
                                           path_data,
                                           simul_name,
                                           simul.i,
                                           t,
                                           tosave,
                                           compressed,
                                           simul.datalock))
        yield


remote_step_writer.writer_type = 'remote'
remote_steps_writer.writer_type = 'remote'


@click.command()
@click.option('-p', '--port', default=50000, help='port number.')
@click.option('--debug-level', 'debug',
              default='DEBUG', help='verbosity level.')
def datreant_server_writer(port, debug):
    current_process().authkey = b'triflow'
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(debug)

    class QueueManager(BaseManager):
        pass

    QueueManager.register('datreant_init', callable=datreant_init)
    QueueManager.register('datreant_save', callable=datreant_save)
    QueueManager.register('datreant_append', callable=datreant_append)
    logger.info('Manager registered')
    local_manager = QueueManager(address=('', port))
    server = local_manager.get_server()
    logger.info('starting server...')
    server.serve_forever()
