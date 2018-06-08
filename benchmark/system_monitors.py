"""
| **@created on:** 20/07/2017,
| **@author:** Prathyush SP,
| **@version:** v0.0.1
|
| **Description:**
| class to handle the CPU and GPU system monitoring
|
| Sphinx Documentation Status:**
|
..todo::
"""

# from py3nvml.py3nvml import *
import time
import psutil
import logging
from threading import Thread
from typeguard import typechecked
from benchmark.gpustat import get_gpu_stats
from abc import abstractmethod
from collections import OrderedDict
from typing import Union

logger = logging.getLogger(__name__)


class MONITORS:
    class TYPE:
        CPU_MONITOR = 'cpu_monitor'
        GPU_MONITOR = 'gpu_monitor'
        MEMORY_MONITOR = 'memory_monitor'

    class CODE:
        RUNNING = 'running'
        ERROR = 'error'
        INITIALIZING = 'initializing'
        DISABLED = 'disabled'
        COMPLETED = 'completed'


class BenchmarkMonitor(Thread):
    """
    | **@author:** Prathyush SP
    |
    | Benchmark Monitors
    """

    @typechecked
    def __init__(self, pid: int, interval: Union[int, float], monitor_type: str):
        self.monitor_type = monitor_type
        self.monitor_disabled = False
        self.pid = pid
        self.interval = interval
        self.process_status = None
        Thread.__init__(self)
        self.psutil_process = None
        self.validate()

    def _check_process_status(self):
        """
        | **@author:** Prathyush SP
        |
        | Check status of pid which is monitored
        :return: True / False based on conditions
        """
        self.process_status = self.psutil_process.status()
        return psutil.pid_exists(self.pid) and self.psutil_process.is_running() \
               and not (self.psutil_process.status() == psutil.STATUS_ZOMBIE
                        or self.psutil_process.status() == psutil.STATUS_DEAD
                        or self.psutil_process.status() == psutil.STATUS_STOPPED) and not self.monitor_disabled

    def run(self):
        """
        | **@author:** Prathyush SP
        |
        | Thread.start() implementation.
        """
        if self.monitor_disabled:
            self.process_status = MONITORS.CODE.DISABLED
            logger.error('Monitor disabled due to error . . .')
        else:
            try:
                logger.debug('Starting {} . . .'.format(self.monitor_type))
                while self._check_process_status():
                    logger.debug('Running {} . . .'.format(self.monitor_type))
                    self.monitor_running()
                    time.sleep(self.interval)
                self.monitor_stopped()
                self.process_status = MONITORS.CODE.COMPLETED if self.psutil_process.wait() == 0 else MONITORS.CODE.ERROR
                logger.debug('Process Exited. Stopping {}  . . .'.format(self.monitor_type))
            except Exception as e:
                self.monitor_stopped()
                self.process_status = MONITORS.CODE.ERROR
                logger.debug('Encountered {}. Stopping {}'.format(e, self.monitor_type))

    @abstractmethod
    def monitor_running(self):
        """
        | **@author:** Prathyush SP
        |
        | Initialize / Update monitor variables during runtime
        """
        pass  # pragma: no cover

    @abstractmethod
    def monitor_stopped(self):
        """
        | **@author:** Prathyush SP
        |
        | Initialize / Update monitor variables on stop
        """
        pass  # pragma: no cover

    @abstractmethod
    def monitor_stats(self):
        """
        | **@author:** Prathyush SP
        |
        :return - Monitor Statistics
        """
        pass  # pragma: no cover

    def validate(self):
        """
        | **@author:** Prathyush SP
        |
        | BenchmarkMonitor 
        :return: 
        """
        if self.interval < 1:
            self.interval = 1
            logger.debug('Minimum supported interval is 1 seconds')
        if psutil.pid_exists(self.pid):
            self.psutil_process = psutil.Process(self.pid)
            self.process_status = MONITORS.CODE.INITIALIZING
        else:
            logger.debug('Process does not exist')
            self.monitor_disabled = True
            self.process_status = 'ERROR'


class MemoryMonitor(BenchmarkMonitor):
    """
    | **@author:** Prathyush SP
    |
    | Memory Monitor
    """

    def __init__(self, pid: int, interval: Union[int, float] = 1):
        BenchmarkMonitor.__init__(self, pid=pid, interval=interval, monitor_type=MONITORS.TYPE.MEMORY_MONITOR)
        self.memory_usage_per_second = []
        self.max_memory_usage = None

    def monitor_running(self):
        """
        | **@author:** Prathyush SP
        |
        | Initialize / Update monitor variables during runtime
        """
        self.memory_usage_per_second.append(self.psutil_process.memory_info().rss / 2 ** 20)

    def monitor_stopped(self):
        """
        | **@author:** Prathyush SP
        |
        | Initialize / Update monitor variables on stop
        """
        self.max_memory_usage = float(max(self.memory_usage_per_second)) if self.memory_usage_per_second else 0

    def monitor_stats(self):
        """
        | **@author:** Prathyush SP
        |
        :return - Monitor Statistics
        """
        return OrderedDict([
            ('total_memory (GB)', psutil.virtual_memory().total / 2 ** 30),
            ('memory_usage_per_second (MBps)', self.memory_usage_per_second),
            ('max_memory_usage (MB)', self.max_memory_usage)
        ])


class CPUMonitor(BenchmarkMonitor):
    """
    | **@author:** Prathyush SP
    |
    | CPU Monitor
    """

    def __init__(self, pid: int, interval: Union[int, float] = 1):
        BenchmarkMonitor.__init__(self, pid=pid, interval=interval, monitor_type=MONITORS.TYPE.CPU_MONITOR)
        self.cpu_usage_per_second = []
        self.max_cpu_usage = None

    def monitor_running(self):
        """
        | **@author:** Prathyush SP
        |
        | Initialize / Update monitor variables during runtime
        """
        self.cpu_usage_per_second.append(self.psutil_process.cpu_percent() / psutil.cpu_count())

    def monitor_stopped(self):
        """
        | **@author:** Prathyush SP
        |
        | Initialize / Update monitor variables on stop
        """
        self.max_cpu_usage = float(max(self.cpu_usage_per_second)) if self.cpu_usage_per_second else 0

    def monitor_stats(self):
        """
        | **@author:** Prathyush SP
        |
        :return - Monitor Statistics
        """
        return OrderedDict([
            ('cpu_cores', psutil.cpu_count(logical=False)),
            ('cpu_threads', psutil.cpu_count()),
            ('cpu_usage_per_second (%/s)', self.cpu_usage_per_second),
            ('max_cpu_usage (%)', self.max_cpu_usage)
        ])


class GPUMonitor(BenchmarkMonitor):
    """
    | **@author:** Prathyush SP
    |
    | Description: This class is for monitoring the GPU usage
    """

    def __init__(self, pid: Union[int, float], interval: Union[int, float] = 1):
        """
        | **@author:** Prathyush SP
        |
        | GPUMonitor initializer function
        :param pid: Process ID of the process that needs to be monitored
        :param interval: Interval at which the process needs to be monitored
        """
        BenchmarkMonitor.__init__(self, pid=pid, interval=interval, monitor_type=MONITORS.TYPE.GPU_MONITOR)

        # Get the stats using nvidia-smi python wrapper
        stats = get_gpu_stats()
        self.monitor_disabled = True if stats == 'ERROR' else self.monitor_disabled
        if not self.monitor_disabled:
            self.number_gpus = len(stats)
            self.gpus = list(stats.keys())

            self.gpu_total_memory = {g: stats[g]['TotalMemory'] for g in self.gpus}
            self.gpu_power_limit = {g: stats[g]['PowerLimit'] for g in self.gpus}

            self.gpu_memory_usage_per_interval = {g: [] for g in self.gpus}
            self.gpu_utilization_per_interval = {g: [] for g in self.gpus}
            self.gpu_power_drawn_per_interval = {g: [] for g in self.gpus}
            self.gpu_graphics_clock_per_interval = {g: [] for g in self.gpus}
            self.gpu_sm_clock_per_interval = {g: [] for g in self.gpus}
            self.gpu_memory_clock_per_interval = {g: [] for g in self.gpus}
            self.gpu_temperature_per_interval = {g: [] for g in self.gpus}

            self.gpu_max_memory_usage = None
            self.gpu_max_utilization = None
            self.gpu_max_power_drawn = None
            self.gpu_max_graphics_clock = None
            self.gpu_max_sm_clock = None
            self.gpu_max_memory_clock = None
            self.gpu_max_temperature = None

    def monitor_running(self):
        """
        | **@author:** Prathyush SP
        |
        | Fetch statistics while monitoring the process
        :return:
        """
        stats = get_gpu_stats()
        for k, v in stats.items():
            assert k in self.gpus
            self.gpu_memory_usage_per_interval[k].append(v['UsedMemory'])
            self.gpu_utilization_per_interval[k].append(v['GPUUtilization'])
            self.gpu_power_drawn_per_interval[k].append(v['PowerDrawn'])
            self.gpu_graphics_clock_per_interval[k].append(v['GraphicsClock'])
            self.gpu_sm_clock_per_interval[k].append(v['SMClock'])
            self.gpu_memory_clock_per_interval[k].append(v['MemoryClock'])
            self.gpu_temperature_per_interval[k].append(v['GPUTemperature'])

    def monitor_stopped(self):
        """
        |**@author:** Prathyush SP
        |
        | Create the summary from the monitored statistics
        :return:
        """
        self.gpu_max_memory_usage = self._calc_max(stats=self.gpu_memory_usage_per_interval)
        self.gpu_max_utilization = self._calc_max(stats=self.gpu_utilization_per_interval)
        self.gpu_max_power_drawn = self._calc_max(stats=self.gpu_power_drawn_per_interval)
        self.gpu_max_graphics_clock = self._calc_max(stats=self.gpu_graphics_clock_per_interval)
        self.gpu_max_sm_clock = self._calc_max(stats=self.gpu_sm_clock_per_interval)
        self.gpu_max_memory_clock = self._calc_max(stats=self.gpu_memory_clock_per_interval)
        self.gpu_max_temperature = self._calc_max(stats=self.gpu_temperature_per_interval)

    def _calc_max(self, stats: dict):
        """
        | **@author:** Prathyush SP
        |
        | Calculates the max value of every list in a given dict
        :param stats: A dict containing lists of usage for all the GPUs
        :return:
        """
        max_dict = {}
        for g in self.gpus:
            max_dict[g] = None
            if len(stats[g]) > 0:
                max_dict[g] = max(stats[g])
        return max_dict

    def monitor_stats(self):
        """
        | **@author:** Prathyush SP
        |
        | Return the collected statistics in an ordered dict
        :return:
        """
        if self.monitor_disabled:
            return OrderedDict([('error', 'Monitor was disabled')])
        return OrderedDict([
            ('gpu_count', self.number_gpus),
            ('gpu_devices', self.gpus),
            ('gpu_total_memory (in MiB)', self.gpu_total_memory),
            ('gpu_power_limit (in Watt)', self.gpu_power_limit),
            ('gpu_memory_usage_per_interval (in MiB)', self.gpu_memory_usage_per_interval),
            ('gpu_utilization_per_interval (in %)', self.gpu_utilization_per_interval),
            ('gpu_power_drawn_per_interval (in Watt)', self.gpu_power_drawn_per_interval),
            ('gpu_graphics_clock_per_interval (in MHz)', self.gpu_graphics_clock_per_interval),
            ('gpu_sm_clock_per_interval (in MHz)', self.gpu_sm_clock_per_interval),
            ('gpu_memory_clock_per_interval (in MHz)', self.gpu_memory_clock_per_interval),
            ('gpu_temperature_per_interval (in degree C)', self.gpu_temperature_per_interval),
            ('gpu_max_memory_usage (in MiB)', self.gpu_max_memory_usage),
            ('gpu_max_utilization (in %)', self.gpu_max_utilization),
            ('gpu_max_power_drawn (in Watt)', self.gpu_max_power_drawn),
            ('gpu_max_graphics_clock (in MHz)', self.gpu_max_graphics_clock),
            ('gpu_max_sm_clock (in MHz)', self.gpu_max_sm_clock),
            ('gpu_max_memory_clock (in MHz)', self.gpu_max_memory_clock),
            ('gpu_max_temperature (in degree C)', self.gpu_max_temperature),
        ])
