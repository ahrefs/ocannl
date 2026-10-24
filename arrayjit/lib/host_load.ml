open Base

(* See the interface (gh-ocannl-1558). *)

external busy_seconds_stub : unit -> float = "ocannl_host_cpu_busy_seconds"

type reading = { wall_s : float; busy_s : float; own_s : float } [@@deriving sexp_of]
type verdict = Quiet of float | Cotenant of float | Unavailable [@@deriving sexp_of, equal]

let read () =
  let busy_s = busy_seconds_stub () in
  if not (Float.is_finite busy_s) then None
  else
    let t = Unix.times () in
    let own_s = t.tms_utime +. t.tms_stime +. t.tms_cutime +. t.tms_cstime in
    let wall_s = Mtime.Span.to_float_ns (Mtime_clock.elapsed ()) /. 1e9 in
    Some { wall_s; busy_s; own_s }

let cotenant_cores = 1.
let counter_allowance_s = 0.02
let min_span_s = 0.25

let classify ?window_s ~before ~after () =
  let span = after.wall_s -. before.wall_s
  and busy = after.busy_s -. before.busy_s
  and own = after.own_s -. before.own_s in
  if
    (not (Float.is_finite span))
    || (not (Float.is_finite busy))
    || (not (Float.is_finite own))
    || Float.(span <= 0.)
    || Option.exists window_s ~f:(fun w -> (not (Float.is_finite w)) || Float.(w <= 0. || w > span))
    || Float.(busy < 0.)
    || Float.(own < 0.)
  then Unavailable
  else
    let window = Option.value window_s ~default:span in
    let foreign = (busy -. own) /. window in
    if Float.(foreign >= cotenant_cores +. (counter_allowance_s /. window)) then Cotenant foreign
    else Quiet foreign
