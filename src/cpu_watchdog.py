"""CPU watchdog: self-monitors process CPU and kills itself if stuck in a busy loop.

The ccwrapper uvicorn process can enter an epoll busy-loop when a dangling fd
from claude agent SDK subprocesses keeps firing EPOLLIN with no reader.  When
that happens, CPU pins to ~100% doing nothing useful.  This watchdog detects
that condition and exits so systemd Restart=always can bring us back clean.
"""

import asyncio
import logging
import os
import signal
import time

logger=logging.getLogger(__name__)

# Thresholds
_CHECK_INTERVAL_S=int(os.getenv("WATCHDOG_INTERVAL", "30"))
_CPU_THRESHOLD=float(os.getenv("WATCHDOG_CPU_THRESHOLD", "80.0"))
_STRIKES_TO_KILL=int(os.getenv("WATCHDOG_STRIKES", "3"))


def _get_own_cpu_percent(interval: float=1.0) -> float:
    """Measure own CPU% over `interval` seconds using /proc/self/stat.

    Works on Linux without psutil.  Returns 0-N_CORES*100 scale,
    normalised to 0-100 single-core equivalent.
    """
    try:
        clk_tck=os.sysconf("SC_CLK_TCK")

        def _read_cpu_ticks():
            with open("/proc/self/stat") as f:
                parts=f.read().split(")")[-1].split()
            # utime=index 11, stime=index 12 (0-indexed after the ')' split)
            utime=int(parts[11])
            stime=int(parts[12])
            return utime+stime

        t0=_read_cpu_ticks()
        wall0=time.monotonic()
        time.sleep(interval)
        t1=_read_cpu_ticks()
        wall1=time.monotonic()

        delta_ticks=t1-t0
        delta_wall=wall1-wall0
        if delta_wall<=0:
            return 0.0

        cpu_pct=(delta_ticks/clk_tck)/delta_wall*100.0
        return cpu_pct
    except Exception as e:
        logger.warning(f"CPU measurement failed: {e}")
        return 0.0


class CPUWatchdog:
    def __init__(self):
        self._task: asyncio.Task|None=None
        self._strikes=0

    async def _loop(self):
        logger.info(
            f"CPU watchdog started: interval={_CHECK_INTERVAL_S}s, "
            f"threshold={_CPU_THRESHOLD}%, strikes={_STRIKES_TO_KILL}"
        )
        while True:
            await asyncio.sleep(_CHECK_INTERVAL_S)
            # Run blocking CPU measurement in executor to not starve the loop
            loop=asyncio.get_running_loop()
            cpu=await loop.run_in_executor(None, _get_own_cpu_percent, 2.0)

            if cpu>=_CPU_THRESHOLD:
                self._strikes+=1
                logger.warning(
                    f"CPU watchdog: {cpu:.1f}% >= {_CPU_THRESHOLD}% "
                    f"(strike {self._strikes}/{_STRIKES_TO_KILL})"
                )
                if self._strikes>=_STRIKES_TO_KILL:
                    logger.critical(
                        f"CPU watchdog: {_STRIKES_TO_KILL} consecutive strikes — "
                        f"epoll busy-loop detected, forcing exit for systemd restart"
                    )
                    # Give journald a moment to flush
                    await asyncio.sleep(0.5)
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
            else:
                if self._strikes>0:
                    logger.info(f"CPU watchdog: {cpu:.1f}% — back to normal, resetting strikes")
                self._strikes=0

    def start(self):
        if self._task is None or self._task.done():
            self._task=asyncio.get_event_loop().create_task(self._loop())

    def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()


cpu_watchdog=CPUWatchdog()
