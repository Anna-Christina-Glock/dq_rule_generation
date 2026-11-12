"""
Adaptive concurrency control for the parallel LLM batch runner
(main_server_paralell.py).

Motivation
----------
With reasoning models active, requests run long and fill the vLLM KV cache.
Once every slot is occupied, new requests queue on the server: their TTFT
(time to first token) spikes and - because the client keeps firing requests
at a fixed rate - the queue grows instead of draining and everything slows
down.

This module limits the number of in-flight requests to a configurable
ceiling, lowers that ceiling automatically as soon as recent TTFT samples
show the server is falling over, and stops sending requests completely if
the situation gets bad enough. Once the server recovers, the ceiling is
raised again, but only slowly.

Building blocks
---------------
AdaptiveGate
    Thread-safe "at most N requests in flight" limiter whose limit can be
    changed at runtime and that can be paused completely ("break").
AdaptiveController
    Watches a rolling window of TTFT samples of finished requests and
    adjusts the gate's limit:
        * median(last window) > ttft_high  ->  limit *= decrease_factor  (fast)
        * median(last window) > ttft_break ->  break: send nothing new, wait
          for the server to drain, then resume with a few probe requests
        * full window, all < ttft_low      ->  limit += increase_step    (slow)

    Only requests with a usable TTFT (streaming openai/vLLM path) feed the
    controller; samples without a TTFT (failed request, ollama path) are
    ignored.

Note on the break: the check that decides whether the server has recovered
consists of one local integer comparison and one console log line. It does
NOT put any additional traffic on the vLLM server. The only requests the
server sees because of the break are the small number of probe requests
used to confirm that it has recovered.
"""

import statistics
import threading
import time
from collections import deque
from datetime import datetime


def _ts():
    return datetime.now().strftime('%y-%m-%d_%H-%M-%S')


class AdaptiveGate:
    """Blocks worker threads until fewer than `limit` requests are in flight.

    The limit is mutable at runtime (see AdaptiveController). Lowering the
    limit never interrupts running requests - it only holds back new ones.
    """

    def __init__(self, limit, min_limit=1, max_limit=None):
        self._min = int(min_limit)
        self._max = int(max_limit) if max_limit is not None else None
        limit = int(limit)
        if self._max is not None:
            limit = min(self._max, limit)
        limit = max(self._min, limit)
        self._limit = limit
        self._in_flight = 0
        self._paused = False
        self._cv = threading.Condition()

    # -- state ----------------------------------------------------------------

    @property
    def limit(self):
        with self._cv:
            return self._limit

    @property
    def in_flight(self):
        with self._cv:
            return self._in_flight

    @property
    def max_limit(self):
        return self._max

    @property
    def paused(self):
        with self._cv:
            return self._paused

    def status(self):
        state = "PAUSED" if self._paused else "ok"
        return f"limit={self.limit}, in_flight={self.in_flight}, {state}"

    # -- control ---------------------------------------------------------------

    def set_limit(self, new_limit, reason="", allow_below_min=False):
        """Clamp new_limit to [min_limit, max_limit] and apply it.

        allow_below_min=True is used for the small "probe" limit after a
        break: the limit is then deliberately below the normal minimum, so
        only a few requests are allowed through before the decision is
        re-checked.

        Prints a log line when the limit actually changes.
        Returns the limit that is in effect afterwards.
        """
        with self._cv:
            new_limit = int(new_limit)
            if not allow_below_min:
                new_limit = max(self._min, new_limit)
            if self._max is not None:
                new_limit = min(self._max, new_limit)
            old = self._limit
            if new_limit != old:
                self._limit = new_limit
                print(f"{_ts()} [ADAPT] thread limit {old} -> {new_limit} ({reason})")
                self._cv.notify_all()
            return self._limit

    def set_paused(self, flag, reason=""):
        """Break switch.

        While paused NO new request is started (acquire() blocks), so the
        server can work through its queue. Requests that are already running
        are NOT interrupted - they run to completion and report their own
        results.

        This does not contact the server in any way: it only flips a local
        flag and wakes up the worker threads that are waiting on the
        condition variable.
        """
        with self._cv:
            flag = bool(flag)
            if flag == self._paused:
                return
            self._paused = flag
            if flag:
                print(f"{_ts()} [ADAPT] BREAK: TTFT too high -> stop sending new requests, "
                      f"wait for the server to drain ({reason})")
            else:
                print(f"{_ts()} [ADAPT] RESUME: server looks healthy again -> "
                      f"allowing {self._limit} threads ({reason})")
            self._cv.notify_all()

    # -- worker side -------------------------------------------------------------

    def acquire(self):
        """Wait until a request may be sent.

        While the controller has a break active (paused) no new request is
        started; already running requests are never interrupted.

        Returns the number of in-flight requests (including this one) at the
        moment the slot was granted - used for the 'in_flight' result column.
        """
        with self._cv:
            while self._paused or self._in_flight >= self._limit:
                self._cv.wait()
            self._in_flight += 1
            return self._in_flight

    def release(self):
        """Free the slot taken in acquire()."""
        with self._cv:
            self._in_flight -= 1
            self._cv.notify_all()

    def abort(self):
        """Release everything and stop admitting new requests.

        Used to fail fast after repeated break timeouts so the run can still
        write the results collected so far."""
        with self._cv:
            self._limit = 0
            self._paused = True
            self._cv.notify_all()


class AdaptiveController:
    """Adjusts an AdaptiveGate's limit based on a rolling window of TTFTs.

    Evaluation happens on every recorded sample, but at most once per
    `cooldown_s` to avoid flapping. An increase additionally requires a full
    window of "good" samples, so old high TTFTs have to age out first.

    If TTFT gets really bad (median > ttft_break) the controller does not just
    lower the limit - it stops the run (break), waits for the running requests
    to finish and for the server to drain, and then resumes with a few probe
    requests. Set ttft_break = None to switch the break off.
    """

    def __init__(self, gate, window=20, ttft_high=5.0, ttft_low=2.0,
                 ttft_break=15.0, break_enabled=True,
                 decrease_factor=0.5, increase_step=2, cooldown_s=15.0,
                 resume_limit=2, drain_poll_s=0.5, heartbeat_s=10.0,
                 break_timeout_s=120.0, max_breaks=2,
                 min_samples=None):
        if gate is None:
            raise ValueError("gate is required")
        self.gate = gate
        self.window = int(window)
        self.ttft_high = float(ttft_high)
        self.ttft_low = float(ttft_low)
        # None => no break, only the gradual limit adjustment
        self.ttft_break = None if ttft_break is None else float(ttft_break)
        self.break_enabled = bool(break_enabled)
        self.decrease_factor = float(decrease_factor)
        self.increase_step = int(increase_step)
        self.cooldown_s = float(cooldown_s)
        # limit used when the run is resumed after a break
        self.resume_limit = max(1, int(resume_limit))
        self.drain_poll_s = float(drain_poll_s) if drain_poll_s else 0.5
        self.heartbeat_s = float(heartbeat_s) if heartbeat_s else 10.0
        self.break_timeout_s = float(break_timeout_s) if break_timeout_s else 120.0
        self.max_breaks = int(max_breaks) if max_breaks else 2
        # Need at least this many samples before any decision is made.
        self.min_samples = int(min_samples) if min_samples else max(3, self.window // 2)
        self._samples = deque(maxlen=self.window)
        self._lock = threading.Lock()
        self._last_change = 0.0
        self._draining = False
        self._break_count = 0
        self._aborted = False
        self._drain_thread = None

    def record(self, ttft_s):
        """Record the TTFT (seconds) of one finished request.

        None / non-numeric / negative values are ignored (e.g. a failed
        request or the non-streaming ollama path has no TTFT).
        """
        if ttft_s is None:
            return
        try:
            ttft_s = float(ttft_s)
        except (TypeError, ValueError):
            return
        if ttft_s < 0:
            return
        with self._lock:
            if self._aborted:
                return
            self._samples.append(ttft_s)
            self._evaluate_locked()

    def _evaluate_locked(self):
        # Called with self._lock held.
        now = time.monotonic()
        if self._draining:
            # A break is active: the drain thread watches the server, so no
            # new decision is made here. _last_change is deliberately NOT
            # refreshed, so the increase path can act as soon as the probe
            # requests proved the server is healthy again.
            return
        if now - self._last_change < self.cooldown_s:
            return
        samples = list(self._samples)
        if len(samples) < self.min_samples:
            return
        limit = self.gate.limit
        median = statistics.median(samples)
        if median > self.ttft_high:
            if self.break_enabled and self.ttft_break is not None and median > self.ttft_break:
                # TTFT exploded: the server does not even finish prefilling
                # new requests in reasonable time -> full stop, let the
                # running requests finish, wait for the server to drain, then
                # resume with a few probe requests.
                self._break_count += 1
                self._do_break(f"TTFT median {median:.2f}s > {self.ttft_break:.1f}s")
            else:
                # Mild overload: reduce the number of parallel requests.
                self._last_change = now
                new_limit = int(limit * self.decrease_factor)
                if new_limit < limit:
                    self.gate.set_limit(new_limit,
                                        reason=f"TTFT median {median:.1f}s > {self.ttft_high:.1f}s")
        elif len(samples) == self.window and max(samples) < self.ttft_low:
            if self.gate.max_limit is None or limit < self.gate.max_limit:
                self._last_change = now
                self.gate.set_limit(limit + self.increase_step,
                                    reason=f"TTFT max {max(samples):.1f}s < {self.ttft_low:.1f}s")

    def _do_break(self, reason=""):
        """Full break: stop sending new requests, let the running ones finish,
        wait for the server to drain, then resume slowly.

        Guarded by _draining, so several workers that notice the same problem
        start only ONE break. Nothing is sent to the server here.
        """
        if self._draining:
            return
        if self._break_count > self.max_breaks:
            self._aborted = True
            self.gate.abort()
            print(f"{_ts()} [ADAPT] giving up after {self._break_count} breaks - "
                  f"no further requests will be sent")
            return
        self._draining = True
        self._samples.clear()
        self.gate.set_paused(True, reason)
        self._drain_thread = threading.Thread(
            target=self._drain_watch,
            args=(time.monotonic(),),
            daemon=True)
        self._drain_thread.start()

    def _drain_watch(self, t_start):
        """Wait until (almost) no requests are in flight anymore, then
        release the break. Reads local counters only - no extra traffic.
        """
        last_beat = t_start
        while True:
            in_flight = self.gate.in_flight
            if in_flight < self.resume_limit:
                self._samples.clear()
                self.gate.set_limit(self.resume_limit,
                                    reason="resume after break",
                                    allow_below_min=True)
                self.gate.set_paused(False, "server drained")
                self._draining = False
                return
            now = time.monotonic()
            if now - t_start > self.break_timeout_s:
                self._aborted = True
                self.gate.abort()
                print(f"{_ts()} [ADAPT] break timeout after {self.break_timeout_s:.0f}s "
                      f"(in_flight={in_flight}) -> giving up; results so far are saved")
                return
            if now - last_beat >= self.heartbeat_s:
                print(f"{_ts()} [ADAPT] break active: waiting for the server to drain "
                      f"(in_flight={in_flight}, limit={self.gate.limit})")
                last_beat = now
            time.sleep(self.drain_poll_s)

    @property
    def aborted(self):
        """True once the controller gave up; no new request should start."""
        return self._aborted


    def status(self):
        with self._lock:
            samples = list(self._samples)
            n_breaks = self._break_count
            aborted = self._aborted
        if aborted:
            return f"{self.gate.status()}, ABORTED after {n_breaks} break(s) - no further requests"
        if not samples:
            return f"{self.gate.status()}, breaks={n_breaks}, no TTFT samples yet"
        return (f"{self.gate.status()}, breaks={n_breaks}, "
                f"last {len(samples)} TTFTs: median={statistics.median(samples):.2f}s, "
                f"max={max(samples):.2f}s")