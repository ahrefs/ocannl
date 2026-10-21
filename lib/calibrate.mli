(** Memory-bandwidth calibration (gh-ocannl-578): STREAM-style streaming kernels whose byte counts
    are exact by construction — one access site per tensor node and direction, no guards — so the
    calibration rows they append under config [autotune_calibration_file] satisfy the fitter's
    per-leg exactness rule and can constrain the envelope's memory leg
    ([model_peak_memory_bandwidth]). Matmul-family tuning data cannot: its rows are compute-bound
    (their bytes-over-time understates achievable bandwidth no matter how exactly counted) and
    typically carry guards-taken/union upper bounds, so the fit of the memory leg needs exactly this
    kind of calibration diversity — see [tools/calibrate_bandwidth.ml] for the command-line entry
    point and [Ir.Cost_model.Calibration] for the fit semantics. *)

val stream : ?elems:int -> ?repeats:int -> Context.t -> (string * Autotune.report) list
(** [stream ?elems ?repeats ctx] times the four STREAM kernels — copy [c =: a], scale
    [b =: 0.5 *. a], add [c =: a + b], triad [a =: b + 0.5 *. c] — over 1-D single-precision tensors
    of [elems] cells each (default [2^26], 256 MiB per tensor: well past any last-level cache, so
    the measured rates are main-memory rates). Each kernel goes through {!Autotune.tune} with seed
    candidates only ([~rounds:0]) and the schedule cache disabled, so every timed candidate appends
    a calibration row through the ordinary emission path; the search is forced on ([~search:true]) —
    timing is the point of a calibration pass, whatever profile the config picked. Choose [elems] a
    power of two: an extent every workgroup size divides evenly keeps the parallelized candidates
    free of range guards, hence their rows bytes-exact.

    Returns the kernels' tuning reports in order, [(name, report)]. Raises {!Utils.User_error} when
    config [autotune_calibration_file] is not set — the pass would time kernels with nowhere to
    record them. *)

(** {2 Cache-capacity calibration (gh-ocannl-1253)}

    The ranking score ({!Ir.Cost_model.ranking_seconds}) charges an untiled contraction's operand
    re-reads once its working set — per-block global read bytes times resident blocks — exceeds a
    capacity. That capacity is a fitted constant of the proxy, not the device's L2 size, so it is
    calibrated the way the bandwidth leg is: by a sweep whose kernels sit on both sides of it. *)

type ws_point = {
  wp_k : int;  (** The contraction's reduction extent. *)
  wp_loops : int list;
      (** The lowered loop extents of the nest writing the output, outermost first: [[k; m; n]] is
          backprop's weight-gradient order. *)
  wp_working_set : int;
      (** The contraction kernel's working set under the default plan, in bytes
          ({!Ir.Cost_model.reuse_traffic}). *)
  wp_requested : int;  (** Its global read bytes under per-block caching. *)
  wp_distinct : int;  (** Its distinct-cell bytes, read and written ({!Ir.Cost_model.analyze}). *)
  wp_ms : float;  (** The routine's best synced wall time over the repeats, in milliseconds. *)
  wp_rank_ms : float option;
      (** The kernel's ranking score under the capacity in force ({!Autotune.ranking_ms}). *)
  wp_bound_ms : float option;  (** Its distinct-cell roofline lower bound. *)
}
(** One point of the {!working_set} sweep. *)

val working_set : ?m:int -> ?n:int -> ?ks:int list -> ?repeats:int -> Context.t -> ws_point list
(** [working_set ?m ?n ?ks ?repeats ctx] times the untiled weight-gradient contraction
    [dw[o,i] = sum_k dy[k,o] * x[k,i]] — [m] rows of output (default 256), [n] columns (default
    1024: a gpt2_mini_train layer's [l*_ffn_w2] gradient) — at each reduction extent of [ks]
    (default [256] to [65536] in powers of two, batch 2 to 512 at sequence 128), on the plain
    default pipeline ({!Context.compile}: the plan whose re-reads the score charges, never a seeded
    or tuned one). The inputs are explicitly initialized host arrays (a standalone fixture: nothing
    upstream of the kernel is timed or left lazily initialized); the time is the whole routine's,
    synced, best of [repeats] (default 10) after one warm-up run. Each point carries the contraction
    kernel's working set and byte charges as the cost model reads them off the scheduled code. *)

type capacity_fit =
  | Transition of {
      cached_ws : int;  (** The largest working set still served at the cache rate. *)
      spilled_ws : int;  (** The smallest working set served at the DRAM rate. *)
      capacity : int;  (** Their geometric mean: the fitted [model_cache_capacity]. *)
      monotone : bool;
          (** Every cached point's working set is below every spilled point's. [false] means the
              sweep's rates do not split cleanly by working set, and the capacity is a guess. *)
      cache_rate : float;
          (** The median requested rate of the cached points, bytes/s: the fitted
              [model_cache_bandwidth]. *)
    }
  | All_cached of { largest_ws : int; cache_rate : float }
      (** Every point is served at the cache rate: the capacity is at least [largest_ws], and the
          sweep says no more than that (no capacity is invented; extend [ks] to find one). *)
  | All_spilled of { smallest_ws : int }
      (** Every point is served at the DRAM rate: the capacity is below [smallest_ws]. *)
  | No_transition of { contrast : float }
      (** Without a DRAM rate to compare against, the fastest and slowest requested rates differ by
          [contrast] < 2: the sweep cannot tell which side of the capacity it sits on. *)
  | Too_few_points  (** What {!fit_capacity} reads off a sweep. *)

val fit_capacity : ?dram_rate:float -> ws_point list -> capacity_fit
(** The capacity and cache bandwidth the sweep implies. A point's requested rate is its per-block
    read bytes over its time: within capacity, re-reads are served from cache and the rate sits far
    above DRAM bandwidth; past it, the rate falls to it. Points with a smaller working set than the
    fastest one are dropped first: a kernel too short to amortize its launch shows a LOW rate for a
    reason that is not the cache (tuf's 0.4 ms floor at the smallest extent read as spilled and made
    the first fit non-monotone). Over the rest, a point is cached when its rate is at least the
    geometric mean of the fastest rate and [dram_rate] (bytes/s, the calibrated DRAM bandwidth) —
    or, without one, of the fastest and the slowest rates, which needs a contrast of at least 2 to
    mean anything. The capacity is placed at the geometric mean of the working sets either side of
    the split — the model's decision boundary, between the shapes that demonstrably sit on each side
    of it — and the cache bandwidth is the cached points' median rate. *)

(** {2 Issue-rate calibration (gh-ocannl-1442)}

    The ranking score prices a kernel's scalar operations at an effective issue rate
    ([model_issue_rate], the backends' advisory [issue_rate]): what a register-blocktiled kernel
    achieves, its operand staging included — far below the envelope's [peak_flops], which stays the
    bound's ceiling. It is fitted the way the capacity is, on a sweep: the seeded sketch of the
    weight-gradient contraction over reduction extents, timed, against its scalar operation count.
*)

type issue_point = {
  ip_k : int;  (** The contraction's reduction extent. *)
  ip_scalar_ops : int;
      (** The contraction kernel's scalar operations ({!Ir.Cost_model.analyze}'s [flops] less its
          [mma_flops]): the count the score divides by the issue rate. *)
  ip_mma_ops : int;  (** Its tile-MMA operations: 0 for the scalar sketch the sweep seeds. *)
  ip_kernels : int;  (** The routine's shipped kernels. *)
  ip_ms : float;  (** The routine's best synced wall time over the repeats, in milliseconds. *)
  ip_rank_ms : float option;
      (** The contraction kernel's ranking score under the configuration in force
          ({!Autotune.ranking_ms}). *)
  ip_bound_ms : float option;  (** Its distinct-cell roofline lower bound. *)
}
(** One point of the {!issue_sweep}. *)

val issue_sweep : ?m:int -> ?n:int -> ?ks:int list -> ?repeats:int -> Context.t -> issue_point list
(** [issue_sweep ?m ?n ?ks ?repeats ctx] times the contraction [dw[o,i] = sum_k dy[k,o] * x[k,i]] —
    [m] rows (default 512), [n] columns (default 512: deliberately not the 256 x 1024 layer the
    calibration's validation cells time, so that predicting those is not reading the fit back) — at
    each reduction extent of [ks] (default [256] to [32768] in powers of two), compiled through the
    seeded default ({!Autotune.seeded_transform}) at threshold 32 with the split and the reduction
    sink off: the one register-blocktiled kernel whose rate the score prices, whatever gate the
    configuration picks. Raises {!Utils.User_error} where the seeded default does not apply (a
    backend other than CUDA or HIP, scheduling or fission off) or a point did not seed — a fallback
    is the default's kernel, never timed under the sketch's name. Inputs, timing and the kernel's
    selection (the shipped segment with the most operations) are {!working_set}'s. *)

type issue_fit =
  | Issue_fit of {
      rate : float;
          (** Scalar operations per second: the slope's reciprocal, the fitted [model_issue_rate].
          *)
      latency_ms : float;
          (** The fit's intercept: time not proportional to the operations (launch, ramp-up). The
              score does not price it; it is reported so a prediction's residual can be read against
              it. *)
      max_residual : float;  (** The largest relative error of the fitted line over the points. *)
      points : int;  (** The points fitted. *)
    }
  | Issue_unfittable of string  (** Why no line fits: too few points, or no growth. *)

val fit_issue_rate : issue_point list -> issue_fit
(** The least-squares line [time = latency + ops / rate] over the timed points with operations. *)
