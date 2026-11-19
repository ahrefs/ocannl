open Base

type record = { kernel : string; static_workgroup_bytes : int } [@@deriving sexp_of]

(* Domain-local, like [Compile_cost.census]: backends are process-wide singletons, and two domains
   linking at once must not record into each other's brackets. Records accumulate newest first. *)
let census =
  let key : record list ref option ref Stdlib.Domain.DLS.key =
    Stdlib.Domain.DLS.new_key (fun () -> ref None)
  in
  fun () -> Stdlib.Domain.DLS.get key

let armed () = Option.is_some !(census ())

let record ~kernel ~static_workgroup_bytes =
  Option.iter !(census ()) ~f:(fun r -> r := { kernel; static_workgroup_bytes } :: !r)

let with_census f =
  let cell = census () in
  let saved = !cell in
  let fresh = ref [] in
  cell := Some fresh;
  let close () =
    cell := saved;
    Option.iter saved ~f:(fun outer -> outer := !fresh @ !outer)
  in
  match f () with
  | result ->
      close ();
      (result, List.rev !fresh)
  | exception exn ->
      let backtrace = Stdlib.Printexc.get_raw_backtrace () in
      close ();
      Stdlib.Printexc.raise_with_backtrace exn backtrace
