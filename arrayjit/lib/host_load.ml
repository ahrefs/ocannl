open Base

(* See the interface (gh-ocannl-1558, gh-ocannl-1640). *)

external cpu_times_stub : unit -> float * float = "ocannl_host_cpu_times"
external supported_stub : unit -> bool = "ocannl_host_cpu_supported"
external tick_seconds_stub : unit -> float = "ocannl_host_cpu_tick_seconds"

type reading = { wall_s : float; busy_s : float; own_s : float } [@@deriving sexp_of]
type verdict = Quiet of float | Cotenant of float | Unavailable [@@deriving sexp_of, equal]

type detail = {
  verdict : verdict;
  span_s : float option;
  window_s : float option;
  busy_s : float option;
  own_s : float option;
  foreign_cores : float option;
  allowance_cores : float option;
  bar_cores : float option;
}
[@@deriving sexp_of]

let supported = supported_stub ()

let read () =
  let busy_s, own_s = cpu_times_stub () in
  if not (Float.is_finite busy_s) then None
  else
    let own_s =
      if Float.is_finite own_s then Some own_s
      else
        try
          let t = Unix.times () in
          Some (t.tms_utime +. t.tms_stime +. t.tms_cutime +. t.tms_cstime)
        with Unix.Unix_error _ -> None
    in
    Option.map own_s ~f:(fun own_s ->
        let wall_s = Mtime.Span.to_float_ns (Mtime_clock.elapsed ()) /. 1e9 in
        { wall_s; busy_s; own_s })

let cotenant_cores = 1.

let counter_allowance_s =
  let tick = tick_seconds_stub () in
  if Float.is_finite tick && Float.(tick > 0.) then 2. *. tick else 0.02

let min_span_s = 0.25

let unavailable_detail =
  {
    verdict = Unavailable;
    span_s = None;
    window_s = None;
    busy_s = None;
    own_s = None;
    foreign_cores = None;
    allowance_cores = None;
    bar_cores = None;
  }

let classify_detail ?window_s ~before ~after () =
  let span = after.wall_s -. before.wall_s
  and busy = after.busy_s -. before.busy_s
  and own = after.own_s -. before.own_s in
  let span_s = if Float.is_finite span then Some span else None in
  let busy_s = if Float.is_finite busy then Some busy else None in
  let own_s = if Float.is_finite own then Some own else None in
  let valid_span = Float.is_finite span && Float.(span > 0.) in
  let valid_window =
    match window_s with
    | None -> valid_span
    | Some w -> valid_span && Float.is_finite w && Float.(w > 0. && w <= span)
  in
  let effective_window =
    if valid_window then Some (Option.value window_s ~default:span) else None
  in
  let allowance_cores = Option.map effective_window ~f:(fun w -> counter_allowance_s /. w) in
  let bar_cores = Option.map allowance_cores ~f:(fun a -> cotenant_cores +. a) in
  if
    (not valid_window)
    || (not (Float.is_finite busy))
    || (not (Float.is_finite own))
    || Float.(busy < 0.)
    || Float.(own < 0.)
  then
    {
      verdict = Unavailable;
      span_s;
      window_s = effective_window;
      busy_s;
      own_s;
      foreign_cores = None;
      allowance_cores;
      bar_cores;
    }
  else
    let window = Option.value_exn effective_window in
    let bar = Option.value_exn bar_cores in
    let foreign = (busy -. own) /. window in
    let verdict = if Float.(foreign >= bar) then Cotenant foreign else Quiet foreign in
    {
      verdict;
      span_s = Some span;
      window_s = Some window;
      busy_s = Some busy;
      own_s = Some own;
      foreign_cores = Some foreign;
      allowance_cores;
      bar_cores;
    }

let classify ?window_s ~before ~after () = (classify_detail ?window_s ~before ~after ()).verdict

let verdict_name = function
  | Quiet _ -> "quiet"
  | Cotenant _ -> "cotenant"
  | Unavailable -> "unavailable"

let format_opt_float = function
  | Some x when Float.is_finite x -> Printf.sprintf "%.4f" x
  | Some _ | None -> "none"

let format_detail d =
  Printf.sprintf
    "verdict=%s foreign=%s bar=%s allowance=%s span_s=%s window_s=%s busy_s=%s own_s=%s"
    (verdict_name d.verdict)
    (format_opt_float d.foreign_cores)
    (format_opt_float d.bar_cores)
    (format_opt_float d.allowance_cores)
    (format_opt_float d.span_s) (format_opt_float d.window_s) (format_opt_float d.busy_s)
    (format_opt_float d.own_s)
