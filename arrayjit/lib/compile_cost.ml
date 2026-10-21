open Base

type t = {
  lower_s : float;
  transform_s : float;
  backend_compile_s : float;
  device_s : float option;
  link_s : float;
  kernels : int;
  kernels_compiled : int;
  reuse_capable : bool;
}
[@@deriving sexp_of]

let empty =
  {
    lower_s = 0.;
    transform_s = 0.;
    backend_compile_s = 0.;
    device_s = None;
    link_s = 0.;
    kernels = 0;
    kernels_compiled = 0;
    reuse_capable = false;
  }

let codegen_s c = c.backend_compile_s -. Option.value c.device_s ~default:0.
let total_s c = c.lower_s +. c.transform_s +. c.backend_compile_s +. c.link_s

let add a b =
  {
    lower_s = a.lower_s +. b.lower_s;
    transform_s = a.transform_s +. b.transform_s;
    backend_compile_s = a.backend_compile_s +. b.backend_compile_s;
    device_s =
      (match (a.device_s, b.device_s) with
      | None, None -> None
      | x, y -> Some (Option.value x ~default:0. +. Option.value y ~default:0.));
    link_s = a.link_s +. b.link_s;
    kernels = a.kernels + b.kernels;
    kernels_compiled = a.kernels_compiled + b.kernels_compiled;
    reuse_capable = a.reuse_capable || b.reuse_capable;
  }

(* Domain-local, like [Backends.segments_census]: backends are process-wide singletons, and two
   domains compiling at once must not record into each other's brackets. *)
let census =
  let key : t ref option ref Stdlib.Domain.DLS.key =
    Stdlib.Domain.DLS.new_key (fun () -> ref None)
  in
  fun () -> Stdlib.Domain.DLS.get key

let with_census f =
  let cell = census () in
  let saved = !cell in
  let fresh = ref empty in
  cell := Some fresh;
  let close () =
    cell := saved;
    Option.iter saved ~f:(fun outer -> outer := add !outer !fresh)
  in
  match f () with
  | result ->
      close ();
      (result, !fresh)
  | exception exn ->
      let backtrace = Stdlib.Printexc.get_raw_backtrace () in
      close ();
      Stdlib.Printexc.raise_with_backtrace exn backtrace

let update g = match !(census ()) with None -> () | Some r -> r := g !r

let timed g f =
  match !(census ()) with
  | None -> f ()
  | Some _ ->
      let c = Mtime_clock.counter () in
      let record () =
        let s = Mtime.Span.to_float_ns (Mtime_clock.count c) /. 1e9 in
        update (fun r -> g r s)
      in
      Exn.protect ~f ~finally:record

let time_lower f = timed (fun r s -> { r with lower_s = r.lower_s +. s }) f
let time_transform f = timed (fun r s -> { r with transform_s = r.transform_s +. s }) f

let time_backend_compile f =
  timed (fun r s -> { r with backend_compile_s = r.backend_compile_s +. s }) f

let time_device f =
  timed (fun r s -> { r with device_s = Some (Option.value r.device_s ~default:0. +. s) }) f

let time_link f = timed (fun r s -> { r with link_s = r.link_s +. s }) f

let record_kernels ~rendered ~compiled =
  update (fun r ->
      { r with kernels = r.kernels + rendered; kernels_compiled = r.kernels_compiled + compiled })

let record_reuse_capable () = update (fun r -> { r with reuse_capable = true })
