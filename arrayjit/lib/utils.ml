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

type settings = {
  mutable debug_log_jitted : bool;
  mutable debug_memory_locations : bool;
  mutable output_debug_files_in_run_directory : bool;
  mutable with_debug : bool;
  mutable fixed_state_for_init : int option;
  mutable print_decimals_precision : int;  (** When rendering arrays etc., outputs this many decimal digits. *)
}
[@@deriving sexp]

let settings =
  {
    debug_log_jitted = false;
    debug_memory_locations = false;
    output_debug_files_in_run_directory = false;
    with_debug = false;
    fixed_state_for_init = None;
    print_decimals_precision = 2;
  }

let accessed_global_args = Hash_set.create (module String)

let config_file_args =
  Stdio.In_channel.read_lines "ocannl_config"
  |> List.map ~f:(String.split ~on:'=')
  |> List.concat_map ~f:(function [] -> [] | key :: vals -> List.map vals ~f:(fun v -> (key, v)))
  |> Hashtbl.of_alist_exn (module String)

(** Retrieves [arg_name] argument from the command line or from an environment variable, returns
    [default] if none found. *)
let get_global_arg ~default ~arg_name:n =
  if settings.with_debug then Stdio.printf "Retrieving commandline or environment variable %s\n%!" n;
  Hash_set.add accessed_global_args n;
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
      if settings.with_debug then Stdio.printf "Found %s, commandline %s\n%!" result arg;
      result
  | None -> (
      match
        List.find_map env_variants ~f:(fun env_n ->
            Option.map (Core.Sys.getenv env_n) ~f:(fun v -> (env_n, v)))
      with
      | Some (env_n, v) ->
          if settings.with_debug then Stdio.printf "Found %s, environment %s\n%!" v env_n;
          v
      | None -> (
          match
            List.find_map env_variants ~f:(fun env_n ->
                Option.map (Hashtbl.find config_file_args env_n) ~f:(fun v -> (env_n, v)))
          with
          | Some (env_n, v) ->
              if settings.with_debug then Stdio.printf "Found %s, config file %s\n%!" v env_n;
              v
          | None ->
              if settings.with_debug then Stdio.printf "Not found, using default %s\n%!" default;
              default))

let get_debug name =
  let snapshot_every_sec = get_global_arg ~default:"" ~arg_name:"snapshot_every_sec" in
  let snapshot_every_sec =
    if String.is_empty snapshot_every_sec then None else Float.of_string_opt snapshot_every_sec
  in
  let time_tagged =
    match String.lowercase @@ get_global_arg ~default:"elapsed" ~arg_name:"time_tagged" with
    | "not_tagged" -> Minidebug_runtime.Not_tagged
    | "clock" -> Clock
    | "elapsed" -> Elapsed
    | s -> invalid_arg @@ "ocannl_time_tagged setting should be none, clock or elapsed; found: " ^ s
  in
  let elapsed_times =
    match String.lowercase @@ get_global_arg ~default:"not_reported" ~arg_name:"elapsed_times" with
    | "not_reported" -> Minidebug_runtime.Not_reported
    | "seconds" -> Seconds
    | "milliseconds" -> Milliseconds
    | "microseconds" -> Microseconds
    | "nanoseconds" -> Nanoseconds
    | s ->
        invalid_arg
        @@ "ocannl_elapsed_times setting should be not_reported, seconds or milliseconds, microseconds or \
            nanoseconds; found: " ^ s
  in
  let location_format =
    match String.lowercase @@ get_global_arg ~default:"beg_pos" ~arg_name:"location_format" with
    | "no_location" -> Minidebug_runtime.No_location
    | "file_only" -> File_only
    | "beg_line" -> Beg_line
    | "beg_pos" -> Beg_pos
    | "range_line" -> Range_line
    | "range_pos" -> Range_pos
    | s -> invalid_arg @@ "ocannl_location_format setting should be none, clock or elapsed; found: " ^ s
  in
  let flushing, backend =
    match String.lowercase @@ String.strip @@ get_global_arg ~default:"html" ~arg_name:"debug_backend" with
    | "text" -> (false, `Text)
    | "html" -> (false, `Html Minidebug_runtime.default_html_config)
    | "markdown" -> (false, `Markdown Minidebug_runtime.default_md_config)
    | "flushing" -> (true, `Text)
    | s ->
        invalid_arg @@ "ocannl_debug_backend setting should be text, html, markdown or flushing; found: " ^ s
  in
  let hyperlink = get_global_arg ~default:"./" ~arg_name:"hyperlink_prefix" in
  let print_entry_ids = Bool.of_string @@ get_global_arg ~default:"false" ~arg_name:"logs_print_entry_ids" in
  let filename = if String.is_empty name then "debug" else "debug-" ^ name in
  let log_level =
    match
      String.lowercase @@ String.strip @@ get_global_arg ~default:"nonempty_entries" ~arg_name:"log_level"
    with
    | "nothing" -> Minidebug_runtime.Nothing
    | "prefixed_error" -> Prefixed [| "ERROR" |]
    | "prefixed_warn_error" -> Prefixed [| "WARN"; "ERROR" |]
    | "prefixed_info_warn_error" -> Prefixed [| "INFO"; "WARN"; "ERROR" |]
    | "explicit_logs" -> Prefixed [||]
    | "nonempty_entries" -> Nonempty_entries
    | "everything" -> Everything
    | s ->
        invalid_arg
        @@ "ocannl_log_level setting should be one of: nothing, prefixed_error, prefixed_warn_error, \
            prefixed_info_warn_error, explicit_logs, nonempty_entries, everything; found: " ^ s
  in
  if flushing then
    Minidebug_runtime.debug_flushing ~filename ~time_tagged ~elapsed_times ~print_entry_ids
      ~global_prefix:name ~for_append:false (* ~log_level *) ()
  else
    Minidebug_runtime.forget_printbox
    @@ Minidebug_runtime.debug_file ~time_tagged ~elapsed_times ~location_format ~print_entry_ids
         ~global_prefix:name ~for_append:false ~max_inline_sexp_length:120 ~hyperlink ~values_first_mode:true
         ~backend ~log_level ?snapshot_every_sec filename

module Debug_runtime = (val get_debug "")

[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

(* [%%global_debug_interrupts { max_nesting_depth = 100; max_num_children = 1000 }] *)

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
    on the 0th position, with [from] ranging from [1] to [num_devices - 1], and [to_ < from]. *)
let%track_sexp parallel_merge merge (num_devices : int) =
  let rec loop (upper : int) : unit =
    let is_even = (upper + 1) % 2 = 0 in
    let lower = if is_even then 0 else 1 in
    let half : int = (upper - (lower - 1)) / 2 in
    if half > 0 then (
      let midpoint : int = half + lower - 1 in
      for i = lower to midpoint do
        (* Maximal [from] is [2 * half + lower - 1 = upper]. *)
        merge ~from:(half + i) ~to_:i
      done;
      loop midpoint)
  in
  loop (num_devices - 1)

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

let sexp_append ~elem = function
  | Sexp.List l -> Sexp.List (elem :: l)
  | Sexp.Atom _ as e2 -> Sexp.List [ elem; e2 ]

let sexp_mem ~elem = function
  | Sexp.Atom _ as e2 -> Sexp.equal elem e2
  | Sexp.List l -> Sexp.(List.mem ~equal l elem)

let rec sexp_deep_mem ~elem = function
  | Sexp.Atom _ as e2 -> Sexp.equal elem e2
  | Sexp.List l -> Sexp.(List.mem ~equal l elem) || List.exists ~f:(sexp_deep_mem ~elem) l

let split_with_seps sep s =
  let tokens = Re.split_full sep s in
  List.map tokens ~f:(function `Text tok -> tok | `Delim sep -> Re.Group.get sep 0)

module Lazy = struct
  include Lazy

  let sexp_of_t = Minidebug_runtime.sexp_of_lazy_t
  let sexp_of_lazy_t = Minidebug_runtime.sexp_of_lazy_t
end
