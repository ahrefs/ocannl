# This file contains old completed small-granularity tasks.
(B) bin/moons_benchmark with the cc backend crashes with half-prec overflow {cm:2024-11-24}
(B) remove syncing from the data parallel algo: stream-to-stream syncing is now automatic {cm:2024-11-23}
(A) cuda backend crashes in bin/moons_benchmark {cm:2024-11-22}
(B) figure out why cuda backend parallelism slows down in later epochs {cm:2024-11-25}
clean up event hashtables when a stream or device gets synchronized {cm:2024-12-03}
(A) Ensure that reading from host on CPU performs required synchronization {cm:2024-12-31}
Update `anatomy_of_a_backend.md` {cm:2025-01-01}
Update introductory slides {cm:2024-12-17}
Config to skip capturing logs from stdout {cm:2024-12-18}
Automatic blocking on access of a host array when a scheduled `to_host` transfer has not finished {cm:2025-01-01}
Migrate graphing to PrintBox-distributed extension {cm:2025-01-24}