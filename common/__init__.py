"""common — общая инфраструктура проекта G1 EDU.

Содержит:
- :mod:`common.transport` — обёртка над ZeroMQ для pub/sub и req/rep.
- :mod:`common.config` — конфигурация через YAML (``config/default.yaml``).
- :mod:`common.logger` — настроенный rich-логгер с файловым выводом.
- :mod:`common.state` — общие структуры данных (pydantic).
- :mod:`common.mock_hardware` — симулятор железа ``MockG1``.
- :mod:`common.monitor` — CLI-утилита для просмотра ZeroMQ-топиков.
"""
