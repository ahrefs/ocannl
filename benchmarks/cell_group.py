#!/usr/bin/env python3
"""One spawn, kill, observe, and reap discipline for benchmark child processes.

The benchmark drivers deliberately differ in what they do after a failed cleanup: the matrix
orchestrator records a failed cell, while a focused measurement driver stops before contaminating
the next pair.  This module owns the mechanism and returns an observation for that caller policy:
``GONE``, ``SURVIVORS``, or ``UNKNOWN``.

POSIX children lead a new session/process group.  Windows children are born suspended, assigned
to a kill-on-close Job Object, and only then resumed; unlike ``taskkill /T``, the Job still owns a
descendant after its original leader exits.
"""

import atexit
import contextlib
import enum
import errno
import os
import select
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


class Observation(enum.Enum):
    GONE = "gone"
    SURVIVORS = "survivors"
    UNKNOWN = "unknown"


GONE = Observation.GONE
SURVIVORS = Observation.SURVIVORS
UNKNOWN = Observation.UNKNOWN


class CleanupFailed(SystemExit):
    """A managed group was not proven gone; its message owns the operator's next action."""


def cancel_on_launcher_exit():
    """Ask Linux to route direct-launcher death through the driver's SIGTERM handler."""
    # A driver that explicitly inherited ignored HUP can outlive its launcher.
    if sys.platform != "linux" or signal.getsignal(signal.SIGHUP) is signal.SIG_IGN:
        return

    parent = os.getppid()
    if parent and hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        try:
            launcher_fd = os.pidfd_open(parent)
        except ProcessLookupError:
            signal.raise_signal(signal.SIGTERM)
            return
        except OSError as exc:
            if exc.errno != errno.ENOSYS:
                raise
        else:
            try:
                if os.getppid() != parent:
                    signal.raise_signal(signal.SIGTERM)
                    return
                _start_launcher_guard(launcher_fd)
            finally:
                os.close(launcher_fd)
            return

    # Linux without pidfds still delivers a notification for the launching thread.
    import ctypes

    prctl = ctypes.CDLL(None, use_errno=True).prctl
    # PR_SET_PDEATHSIG. A launcher that exits during registration cannot send this signal.
    if prctl(1, signal.SIGTERM, 0, 0, 0) == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != parent:
        signal.raise_signal(signal.SIGTERM)


def _guard_command(*args):
    return [sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()), *args]


def _start_launcher_guard(launcher_fd):
    driver_fd = os.pidfd_open(os.getpid())
    try:
        guard = subprocess.Popen(
            _guard_command("--launcher-guard", str(launcher_fd), str(driver_fd)),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            pass_fds=(launcher_fd, driver_fd), start_new_session=True,
        )
    finally:
        os.close(driver_fd)

    def reap():
        # The monitor owns only pidfds and can inherit a blocked SIGTERM.
        guard.kill()
        guard.wait()

    atexit.register(reap)


def _guard_launcher(launcher_fd, driver_fd):
    # A process pidfd becomes readable after the last thread exits.
    deaths = select.poll()
    deaths.register(launcher_fd, select.POLLIN)
    deaths.register(driver_fd, select.POLLIN)
    if driver_fd not in dict(deaths.poll()):
        with contextlib.suppress(ProcessLookupError):
            signal.pidfd_send_signal(driver_fd, signal.SIGTERM)


class _DriverDeathGuard:
    """Hold inherited fleet locks until the driver has cleaned its group or we have."""

    def __init__(self):
        owner, reader = socket.socketpair()
        try:
            with reader:
                # The slot's flock and sleep-inhibitor pipe are intentionally inheritable.
                # Only the guard needs to keep them when the driver is gone.
                self.proc = subprocess.Popen(
                    _guard_command("--driver-guard"),
                    stdin=reader, stdout=subprocess.DEVNULL, close_fds=False,
                    start_new_session=True,
                )
        except BaseException:
            owner.close()
            raise
        self.owner = owner

    def register_child(self):
        # Called in the child after setsid and before exec. Even if Popen never returns to
        # the driver, exec cannot start the cell until its group is known to the guard.
        self.owner.sendall(b"%d\n" % os.getpid())
        os.set_inheritable(self.owner.fileno(), False)

    def finish(self):
        # The group is already gone. Preserve its outcome if the guard died independently.
        with self.owner, contextlib.suppress(OSError):
            self.owner.sendall(b"D")
        self.proc.wait()


def _guard_driver():
    registration = sys.stdin.buffer.readline()
    # EOF in the middle of an ID cannot authorize a signal to that numeric prefix.
    if not registration.endswith(b"\n") or not registration[:-1].isdigit():
        return
    if sys.stdin.buffer.read(1) == b"D":
        return
    pgid = int(registration)
    # EOF without a completion byte is a driver death. A live member can still own GPU
    # resources after SIGKILL was sent, so keep the locks until the group is observed gone.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)
    while _observe_posix_group(pgid, allow_zombie_gone=True) is not GONE:
        time.sleep(0.05)


class CancellationDeferral:
    """Defer SIGINT/SIGTERM/SIGHUP while a managed child has not yet got a safe cleanup path.

    ``deferring`` covers spawn and cleanup; ``cancellable`` opens the one intentional hole around
    the blocking wait.  Signal masking is deliberately not used because a forked child would
    inherit the mask and ignore the supervisor's graceful termination phase. ``cleanup_message``
    is the driver-specific fact appended when a deferred SIGTERM is delivered after cleanup.

    SIGHUP is a termination like SIGTERM (gh-ocannl-1200 review).  A managed child leads a session
    of its own, so a terminal or ssh hangup reaches only the driver's foreground group; left at
    its default action the driver dies without unwinding, and the child it was waiting on runs on,
    detached and past its deadline.  A driver started with SIGHUP ignored (``nohup``) asked to
    outlive the hangup, and keeps that disposition.
    """

    def __init__(
        self,
        label,
        *,
        cleanup_message="the child process group was cleaned first",
    ):
        self.label = label
        self.cleanup_message = cleanup_message
        self.depth = 0
        self.held_signal = None

    def install(self):
        def terminate(signum, _frame):
            if self.depth:
                self.held_signal = signum
                return
            raise SystemExit(f"{self.label}: terminated by signal {signum}")

        def interrupt(signum, _frame):
            if self.depth:
                self.held_signal = signum
                return
            raise KeyboardInterrupt

        handlers = [(signal.SIGTERM, terminate), (signal.SIGINT, interrupt)]
        hangup = getattr(signal, "SIGHUP", None)  # POSIX only
        if hangup is not None and signal.getsignal(hangup) is not signal.SIG_IGN:
            handlers.append((hangup, terminate))
        for signum, handler in handlers:
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                pass

    def _raise_held(self):
        if self.held_signal is None:
            return
        signum, self.held_signal = self.held_signal, None
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(f"{self.label}: terminated by signal {signum}; {self.cleanup_message}")

    def _deliver_or_annotate(self):
        if self.held_signal is None:
            return
        active = sys.exc_info()[1]
        if isinstance(active, CleanupFailed):
            signum, self.held_signal = self.held_signal, None
            if hasattr(active, "add_note"):
                active.add_note(f"signal {signum} was also received while cleanup was failing")
            return
        self._raise_held()

    @contextlib.contextmanager
    def deferring(self):
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
            if self.depth == 0:
                # A cleanup failure is the stronger fact: replacing it with the cancellation's
                # generic "cleaned first" exit would invite a retry over the very survivor it
                # reported. Other active exceptions still yield to the held operator request.
                self._deliver_or_annotate()

    @contextlib.contextmanager
    def cancellable(self):
        held_depth, self.depth = self.depth, 0
        try:
            self._raise_held()
            yield
        finally:
            self.depth = held_depth


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    TH32CS_SNAPTHREAD = 0x00000004
    THREAD_SUSPEND_RESUME = 0x0002
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    JobObjectBasicAccountingInformation = 1
    JobObjectExtendedLimitInformation = 9
    CREATE_SUSPENDED = getattr(subprocess, "CREATE_SUSPENDED", 0x00000004)

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_longlong),
            ("TotalKernelTime", ctypes.c_longlong),
            ("ThisPeriodTotalUserTime", ctypes.c_longlong),
            ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    class THREADENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ThreadID", wintypes.DWORD),
            ("th32OwnerProcessID", wintypes.DWORD),
            ("tpBasePri", wintypes.LONG),
            ("tpDeltaPri", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    _kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _kernel32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
    _kernel32.Thread32First.restype = wintypes.BOOL
    _kernel32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
    _kernel32.Thread32Next.restype = wintypes.BOOL
    _kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenThread.restype = wintypes.HANDLE
    _kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    _kernel32.ResumeThread.restype = wintypes.DWORD
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD


def _resume_windows_initial_thread(pid):
    """Resume the sole thread of a process created with ``CREATE_SUSPENDED``."""
    snapshot = _kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    if snapshot == INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = THREADENTRY32()
        entry.dwSize = ctypes.sizeof(entry)
        more = _kernel32.Thread32First(snapshot, ctypes.byref(entry))
        while more:
            if entry.th32OwnerProcessID == pid:
                thread = _kernel32.OpenThread(THREAD_SUSPEND_RESUME, False, entry.th32ThreadID)
                if not thread:
                    raise ctypes.WinError(ctypes.get_last_error())
                try:
                    if _kernel32.ResumeThread(thread) == 0xFFFFFFFF:
                        raise ctypes.WinError(ctypes.get_last_error())
                    return
                finally:
                    _kernel32.CloseHandle(thread)
            more = _kernel32.Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        _kernel32.CloseHandle(snapshot)
    raise OSError(f"could not find the suspended initial thread of process {pid}")


class _WindowsJob:
    """Kill-on-close Job Object; instantiated only on Windows."""

    def __init__(self):
        self.handle = _kernel32.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _kernel32.SetInformationJobObject(
            self.handle,
            JobObjectExtendedLimitInformation,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign_and_resume(self, proc):
        if not _kernel32.AssignProcessToJobObject(self.handle, int(proc._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        _resume_windows_initial_thread(proc.pid)

    def observe(self):
        if not self.handle:
            return UNKNOWN
        accounting = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        if not _kernel32.QueryInformationJobObject(
            self.handle,
            JobObjectBasicAccountingInformation,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            None,
        ):
            return UNKNOWN
        return SURVIVORS if accounting.ActiveProcesses else GONE

    def terminate(self):
        if self.handle and not _kernel32.TerminateJobObject(self.handle, 1):
            error = ctypes.get_last_error()
            if error:
                raise ctypes.WinError(error)

    def close(self):
        if self.handle:
            _kernel32.CloseHandle(self.handle)
            self.handle = None


@dataclass
class ManagedProcess:
    """A ``Popen`` plus the stable group/job identity created with it."""

    proc: subprocess.Popen
    pgid: int = None
    job: object = None
    guard: object = None

    def __getattr__(self, name):
        return getattr(self.proc, name)

    def observe(self, allow_zombie_gone=False):
        if self.job is not None:
            return self.job.observe()
        if self.pgid is not None:
            return _observe_posix_group(self.pgid, allow_zombie_gone=allow_zombie_gone)
        # A Windows group whose Job handle has been closed.  `close` is called once the group was
        # observed GONE -- and KILL_ON_JOB_CLOSE means the closing itself took anything that had
        # somehow outlived that observation -- so there is nothing left to look at, and GONE is the
        # answer.  Dispatching on the job alone sent this case to the POSIX census instead, which is
        # how `_run_cell`'s leftover probe, taken after a timeout already killed and closed the
        # group, came to call `os.killpg` on Windows: an AttributeError, out of the one call whose
        # whole job is to report on cleanup.
        return GONE

    def signal(self, force):
        if self.job is not None:
            if force:
                self.job.terminate()
            else:
                with contextlib.suppress(ProcessLookupError, OSError):
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            return
        if self.pgid is None:
            return  # Closed Windows group; see `observe`.  KILL_ON_JOB_CLOSE has done the signalling.
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            os.killpg(self.pgid, sig)
        except ProcessLookupError:
            # The group is normally established before ``Popen`` returns.  The direct fallback
            # covers an implementation/platform where it was not, without ever signalling the
            # benchmark driver's own process group.
            if self.proc.poll() is None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    self.proc.send_signal(sig)

    def close(self):
        if self.job is not None:
            self.job.close()
            self.job = None
        if self.guard is not None:
            guard, self.guard = self.guard, None
            guard.finish()

    def __del__(self):
        # KILL_ON_JOB_CLOSE is the last safety net if a caller loses the object during unwinding.
        with contextlib.suppress(Exception):
            if self.job is not None:
                self.job.close()
            if self.guard is not None:
                self.guard.owner.close()


def _cleanup_failed_windows_spawn(job, proc):
    """Clean both possible owners when Windows Job setup fails partway through."""
    with contextlib.suppress(Exception):
        job.terminate()
    job.close()
    if proc is not None:
        # Assignment itself can fail, leaving the suspended child outside the empty Job. Killing
        # the Job then reaches nothing, so the direct process is an independent obligation.
        with contextlib.suppress(Exception):
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=1)


def spawn(args, *, watch_driver=False, **kwargs):
    """Spawn ``args`` isolated from the driver and return its managed group."""
    if os.name == "posix":
        if kwargs.pop("start_new_session", True) is not True:
            raise ValueError("benchmark children must start in their own session")
        guard = None
        if watch_driver and sys.platform == "linux":
            if kwargs.get("preexec_fn") is not None:
                raise ValueError("the driver death guard owns the benchmark child's preexec_fn")
            guard = _DriverDeathGuard()
            # The synchronous cells driver owns this path. The child callback only
            # publishes its group ID on the inherited socket.
            kwargs["preexec_fn"] = guard.register_child
            kwargs["pass_fds"] = (*kwargs.get("pass_fds", ()), guard.owner.fileno())
        try:
            proc = subprocess.Popen(args, start_new_session=True, **kwargs)
        except BaseException:
            if guard is not None:
                guard.owner.close()
                guard.proc.wait()
            raise
        return ManagedProcess(proc=proc, pgid=proc.pid, guard=guard)
    if os.name == "nt":
        requested = kwargs.pop("creationflags", 0)
        flags = requested | subprocess.CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED
        job = _WindowsJob()
        proc = None
        try:
            proc = subprocess.Popen(args, creationflags=flags, **kwargs)
            job.assign_and_resume(proc)
            return ManagedProcess(proc=proc, job=job)
        except BaseException:
            _cleanup_failed_windows_spawn(job, proc)
            raise
    raise NotImplementedError(f"no child-group implementation for os.name={os.name!r}")


def process_is_alive(pid):
    """Whether ``pid`` names a process that has not exited yet.

    ``os.kill(pid, 0)`` is the POSIX spelling of this question and is the wrong thing to run on
    Windows for two independent reasons.  CPython implements ``os.kill`` there as ``OpenProcess``
    followed by ``TerminateProcess(handle, sig)``, so asking with signal 0 KILLS the process it was
    supposed to be asking about; and a pid that has already exited fails the ``OpenProcess`` with
    ``ERROR_INVALID_PARAMETER``, which surfaces as a bare ``OSError`` (``WinError 87``) that no
    ``ProcessLookupError`` handler catches.  A probe that either kills its subject or raises is not
    a probe.

    The Windows answer is a handle wait: a process object becomes signalled when the process exits
    and stays that way while a handle is open, so a zero-timeout wait that TIMES OUT means "still
    running".  ``ERROR_ACCESS_DENIED`` from the open is a live process this account may not look
    at, which is alive; anything else is gone.
    """
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    SYNCHRONIZE = 0x00100000
    WAIT_TIMEOUT = 0x00000102
    ERROR_ACCESS_DENIED = 5
    handle = _kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        return ctypes.get_last_error() == ERROR_ACCESS_DENIED
    try:
        return _kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
    finally:
        _kernel32.CloseHandle(handle)


def _observe_posix_group(pgid, allow_zombie_gone=False):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return GONE
    except PermissionError:
        return UNKNOWN
    except OSError:
        return UNKNOWN

    proc_fs = Path("/proc")
    if not (proc_fs / "self" / "stat").exists():
        # Signal zero proves that the group has process-table entries, but cannot distinguish a
        # runnable member from a zombie.  The caller should escalate conservatively.
        return UNKNOWN

    matched = False
    complete = True
    for entry in proc_fs.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.getpgid(int(entry.name)) != pgid:
                continue
        except ProcessLookupError:
            continue
        except OSError:
            complete = False
            continue
        try:
            # ``comm`` can contain spaces and parentheses.  From the final ')', state is field 1
            # and process-group id field 3 of the remaining text.
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != pgid:
                continue
            matched = True
            if fields[0] != "Z":
                return SURVIVORS
        except (OSError, IndexError, ValueError):
            complete = False
            continue
    # A matched group of corpses is gone for benchmark purposes only after SIGKILL has made a
    # concurrent fork impossible. During the graceful phase, a non-atomic /proc traversal cannot
    # prove that a live member did not join after its directory snapshot. An unread entry
    # makes the census incomplete if it could belong to this group.
    return GONE if allow_zombie_gone and matched and complete else UNKNOWN


@dataclass
class Termination:
    stdout: object
    stderr: object
    observation: Observation
    reaped: bool


def terminate(group, grace, poll_interval=0.05, final_reap=1.0, observe=None):
    """TERM, KILL if needed, reap, and report what remains.

    Output from ``TimeoutExpired`` is cumulative, so the latest partial value replaces rather
    than appends to the prior one.  This preserves diagnostics even when an escaped or genuinely
    unkillable descendant keeps a captured pipe open through every reap attempt.
    """
    observe = observe or group.observe
    stdout = ""
    stderr = ""
    reaped = False

    def output_bytes(value):
        if isinstance(value, str):
            return value.encode(group.encoding or "utf-8", group.errors or "strict")
        return value

    def preserve_output(current, candidate):
        # `communicate` and `TimeoutExpired` return cumulative snapshots, but a later reap can
        # still report an empty or shorter snapshot on a platform-specific pipe-close path. A
        # child killed mid-write has no second chance to produce those bytes, so never replace a
        # longer observation with a shorter one. TimeoutExpired.output remains bytes under
        # text=True, so compare encoded byte counts rather than bytes against Unicode code points.
        if candidate is None or len(output_bytes(candidate)) < len(output_bytes(current)):
            return current
        return candidate

    def reap(timeout):
        nonlocal stdout, stderr, reaped
        if reaped:
            return
        try:
            got_out, got_err = group.communicate(timeout=max(0, timeout))
            stdout = preserve_output(stdout, got_out)
            stderr = preserve_output(stderr, got_err)
            reaped = True
        except subprocess.TimeoutExpired as expired:
            stdout = preserve_output(stdout, expired.output)
            stderr = preserve_output(stderr, expired.stderr)
        except ValueError:
            # An earlier communicate already closed the pipes; poll still reaps the leader.
            group.poll()
            reaped = group.returncode is not None

    for force in (False, True):
        with contextlib.suppress(ProcessLookupError, OSError):
            group.signal(force)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            reap(min(0.5, max(0, deadline - time.monotonic())))
            observation = observe(force)
            if reaped and observation is GONE:
                group.close()
                return Termination(stdout, stderr, observation, reaped)
            time.sleep(poll_interval)

    reap(final_reap)
    observation = observe(True)
    if not reaped:
        # A survivor outside the managed group may still own the inherited pipe.  Stop reading it
        # forever, then reap the direct child independently of that pipe.
        for pipe in (group.stdout, group.stderr, group.stdin):
            if pipe is not None:
                with contextlib.suppress(OSError):
                    pipe.close()
        with contextlib.suppress(OSError):
            group.poll()
        reaped = group.returncode is not None
    if observation is GONE:
        group.close()
    return Termination(stdout, stderr, observation, reaped)


if __name__ == "__main__":
    if sys.argv[1:] == ["--driver-guard"]:
        _guard_driver()
    elif len(sys.argv) == 4 and sys.argv[1] == "--launcher-guard":
        _guard_launcher(*map(int, sys.argv[2:]))
