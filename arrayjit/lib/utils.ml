open Base

module Set_O = struct
  let ( + ) = Set.union
  let ( - ) = Set.diff
  let ( & ) = Set.inter

  let ( -* ) s1 s2 =
    Set.of_sequence (Set.comparator_s s1) @@ Sequence.map ~f:Either.value @@ Set.symmetric_diff s1 s2
end

let no_ints = Set.empty (module Int)
let one_int = Set.singleton (module Int)

let map_merge m1 m2 ~f =
  Map.merge m1 m2 ~f:(fun ~key:_ m ->
      match m with `Right v | `Left v -> Some v | `Both (v1, v2) -> Some (f v1 v2))

let mref_add mref ~key ~data ~or_ =
  match Map.add !mref ~key ~data with `Ok m -> mref := m | `Duplicate -> or_ (Map.find_exn !mref key)

let mref_add_missing mref key ~f =
  if Map.mem !mref key then () else mref := Map.add_exn !mref ~key ~data:(f ())

module Debug_runtime =
  (val Minidebug_runtime.debug_file (* ~split_files_after:(1 lsl 16) *)
         ~time_tagged:true ~for_append:false (* ~hyperlink:"./" *)
         ~hyperlink:"vscode://file//wsl.localhost/Ubuntu/home/lukstafi/ocannl/" ~values_first_mode:true
         ~backend:(`Html Minidebug_runtime.default_html_config) ~snapshot_every_sec:4.
         (* ~backend:(`Html PrintBox_html.Config.(tree_summary true default))  *)
         (* ~prune_upto:5 *)
         (* ~highlight_terms:Re.(alt [ str "(sh_id 46)" ]) *)
         (* ~exclude_on_path:(Re.str "env") *)
         "debug")
(* (val let debug_ch = Stdlib.open_out "debug.log" in
     Minidebug_runtime.debug_flushing ~debug_ch ~time_tagged:false ~print_entry_ids:true ()) *)

(** Retrieves [arg_name] argument from the command line or from an environment variable, returns
    [default] if none found. *)
let%debug_sexp get_global_arg ~default ~arg_name:n =
  [%log "Retrieving commandline or environment variable", "ocannl_" ^ n];
  let variants = [ n; String.uppercase n ] in
  let env_variants =
    List.concat_map variants ~f:(fun n -> [ "ocannl_" ^ n; "OCANNL_" ^ n; "ocannl-" ^ n; "OCANNL-" ^ n ])
  in
  let cmd_variants = List.concat_map env_variants ~f:(fun n -> [ n; "-" ^ n; "--" ^ n ]) in
  let cmd_variants = List.concat_map cmd_variants ~f:(fun n -> [ n ^ "_"; n ^ "-"; n ^ "=" ]) in
  match
    Array.find_map (Core.Sys.get_argv ()) ~f:(fun arg ->
        List.find_map cmd_variants ~f:(fun prefix ->
            Option.some_if (String.is_prefix ~prefix arg) (prefix, arg)))
  with
  | Some (prefix, arg) ->
      let result = String.suffix arg (String.length prefix) in
      [%log "found", result, "commandline", arg];
      result
  | None -> (
      match
        List.find_map env_variants ~f:(fun env_n ->
            Option.map (Core.Sys.getenv env_n) ~f:(fun v -> (env_n, v)))
      with
      | Some (env_n, v) ->
          [%log "found", v, "environment", env_n];
          v
      | None ->
          [%log "not found, using default", default];
          default)

let rec union_find ~equal map ~key ~rank =
  match Map.find map key with
  | None -> (key, rank)
  | Some data -> if equal key data then (key, rank) else union_find ~equal map ~key:data ~rank:(rank + 1)

let union_add ~equal map k1 k2 =
  if equal k1 k2 then map
  else
    let root1, rank1 = union_find ~equal map ~key:k1 ~rank:0
    and root2, rank2 = union_find ~equal map ~key:k2 ~rank:0 in
    if rank1 < rank2 then Map.update map root1 ~f:(fun _ -> root2)
    else Map.update map root2 ~f:(fun _ -> root1)

(** Filters the list keeping the first occurrence of each element. *)
let unique_keep_first ~equal l =
  let rec loop acc = function
    | [] -> List.rev acc
    | hd :: tl -> if List.mem acc hd ~equal then loop acc tl else loop (hd :: acc) tl
  in
  loop [] l

let sorted_diff ~compare l1 l2 =
  let rec loop acc l1 l2 =
    match (l1, l2) with
    | [], _ -> []
    | l1, [] -> List.rev_append acc l1
    | h1 :: t1, h2 :: t2 -> (
        match compare h1 h2 with
        | c when c < 0 -> loop (h1 :: acc) t1 l2
        | 0 -> loop acc t1 l2
        | _ -> loop acc l1 t2)
  in
  (loop [] l1 l2 [@nontail])

(** [parallel_merge merge num_devices] progressively invokes the pairwise [merge] callback, converging
    on the 0th position, with [from] ranging from [0] to [num_devices - 1], and [to_ < from]. *)
let%debug_sexp parallel_merge merge (num_devices : int) =
  let rec loop (from : int) (to_ : int) : unit =
    if to_ > from then
      let is_even = (to_ - from + 1) % 2 = 0 in
      if is_even then (
        let half = (to_ - from + 1) / 2 in
        for i = from to from + half - 1 do
          merge ~from:i ~to_:(i + half)
        done;
        loop 0 (from + half - 1))
      else loop (from + 1) to_
    else if from > 0 then loop 0 from
  in
  loop 0 (num_devices - 1)

type waiter = { await : unit -> unit; release : unit -> unit; finalize : unit -> unit }

let waiter () =
  let pipe_inp, pipe_out = Unix.pipe ~cloexec:true () in
  let await () =
    let _ = Unix.select [ pipe_inp ] [] [] (-1.0) in
    let n = Unix.read pipe_inp (Bytes.create 1) 0 1 in
    assert (n = 1)
  in
  let release () =
    let n = Unix.write pipe_out (Bytes.create 1) 0 1 in
    assert (n = 1)
  in
  let finalize () =
    Unix.close pipe_inp;
    Unix.close pipe_out
  in
  { await; release; finalize }

type settings = {
  mutable debug_log_jitted : bool;
  mutable output_debug_files_in_run_directory : bool;
  mutable with_debug : bool;
  mutable fixed_state_for_init : int option;
  mutable print_decimals_precision : int;  (** When rendering arrays etc., outputs this many decimal digits. *)
}
[@@deriving sexp]

let settings =
  {
    debug_log_jitted = false;
    output_debug_files_in_run_directory = false;
    with_debug = false;
    fixed_state_for_init = None;
    print_decimals_precision = 2;
  }

let sexp_append ~elem = function
  | Sexp.List l -> Sexp.List (elem :: l)
  | Sexp.Atom _ as e2 -> Sexp.List [ elem; e2 ]

let sexp_mem ~elem = function
  | Sexp.Atom _ as e2 -> Sexp.equal elem e2
  | Sexp.List l -> Sexp.(List.mem ~equal l elem)

let rec sexp_deep_mem ~elem = function
  | Sexp.Atom _ as e2 -> Sexp.equal elem e2
  | Sexp.List l -> Sexp.(List.mem ~equal l elem) || List.exists ~f:(sexp_deep_mem ~elem) l
