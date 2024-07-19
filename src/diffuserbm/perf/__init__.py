import json
import time

from collections import defaultdict


_LATENCY_TABLE_HEADER_FORMAT = \
    '| {:25} | {:17} | {:5} | {:23} |'.format('name', 'total', 'count', 'each')
_LATENCY_TABLE_ROW_SPLITTER = \
    '+{}+{}+{}+{}+'.format('-' * 27, '-' * 19, '-' * 7, '-' * 25)
_LATENCY_TABLE_ROW_FORMAT = \
    '| {name:25} | {total:-12,.2f} msec | {count:-5} | {each:-12,.2f} msec/{unit:5} |'
_LATENCY_CSV_HEADER_FORMAT = \
    'name,total,count,each,unit'
_LATENCY_CSV_ROW_FORMAT = \
    '{name},{total:.2f},{count},{each:.2f},msec/{unit}'


class _Latency:
    def __init__(self, name: str, unit: str, monitor):
        self.__name = name
        self.__unit = unit
        self.__monitor = monitor

        self.__tick = None
        self.__tock = None

    @property
    def name(self) -> str:
        return self.__name

    @property
    def unit(self) -> str:
        return self.__unit

    @property
    def latency(self) -> float:
        return (self.__tock - self.__tick) * 1000

    @property
    def key(self) -> str:
        return f'{self.__name}/{self.__unit}'

    def __enter__(self):
        self.__tick = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__tock = time.perf_counter()
        self.__monitor.append(self)


class _LatencyAggregator:
    def __init__(self, name='', unit='', value=0.0, count=0):
        self.__name = name
        self.__unit = unit
        self.__value = value
        self.__count = count

        self.__group = self.__name.split(' ')[0]

    def __add__(self, other: _Latency):
        return _LatencyAggregator(
            name=other.name,
            unit=other.unit,
            value=self.__value + other.latency,
            count=self.__count + 1,
        )

    @property
    def name(self):
        return self.__name

    @property
    def group(self):
        return self.__group

    @property
    def __dict__(self):
        return {
            'name': self.__name,
            'total': {
                'value': round(self.__value, 3),
                'unit': 'msec'
            },
            'each': {
                'value': round(self.__value / self.__count, 3),
                'unit': f'msec/{self.__unit}'
            },
        }

    def __str__(self):
        return '{name}: {total} msec, {each} msec/{unit}'.format(
            name=self.__name,
            total=self.__value,
            each=self.__value / self.__count,
            unit=self.__unit,
        )

    @property
    def __table_row__(self):
        return _LATENCY_TABLE_ROW_FORMAT.format(
            name=self.__name,
            total=self.__value,
            count=self.__count,
            each=self.__value / self.__count,
            unit=self.__unit,
        )

    @property
    def __csv_row__(self):
        return _LATENCY_CSV_ROW_FORMAT.format(
            name=self.__name,
            total=self.__value,
            count=self.__count,
            each=self.__value / self.__count,
            unit=self.__unit,
        )


class PerfMon:
    def __init__(self):
        self.__latencies = []

    def measure_latency(self, name, unit: str = ""):
        return _Latency(name, unit, self)

    def append(self, latency: _Latency):
        self.__latencies.append(latency)

    @property
    def __aggregated_latency__(self):
        results = defaultdict(_LatencyAggregator)

        for metric in self.__latencies:
            results[metric.key] += metric

        return results.values()

    @property
    def __dict__(self):
        return [v.__dict__ for v in self.__aggregated_latency__]

    def __str__(self):
        return '\n'.join([str(v) for v in self.__aggregated_latency__])

    @property
    def json(self):
        return json.dumps(self.__dict__, indent=4)

    @property
    def text(self):
        return str(self)

    @property
    def table(self):
        buffer = [_LATENCY_TABLE_ROW_SPLITTER, _LATENCY_TABLE_HEADER_FORMAT]
        group = ''

        for latency in self.__aggregated_latency__:
            if group != latency.group:
                group = latency.group
                buffer.append(_LATENCY_TABLE_ROW_SPLITTER)

            buffer.append(latency.__table_row__)

        buffer.append(_LATENCY_TABLE_ROW_SPLITTER)

        return '\n'.join(buffer)

    @property
    def csv(self):
        return '\n'.join(
            [_LATENCY_CSV_HEADER_FORMAT] + [v.__csv_row__ for v in self.__aggregated_latency__]
        )

    def report(self, format_type):
        return self.__getattribute__(format_type)
