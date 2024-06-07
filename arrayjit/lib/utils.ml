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
  mutable debug_log_from_routines : bool;
  mutable debug_memory_locations : bool;
  mutable output_debug_files_in_run_directory : bool;
  mutable with_debug_level : int;
  mutable fixed_state_for_init : int option;
  mutable print_decimals_precision : int;  (** When rendering arrays etc., outputs this many decimal digits. *)
}
[@@deriving sexp]

let settings =
  {
    debug_log_from_routines = false;
    debug_memory_locations = false;
    output_debug_files_in_run_directory = false;
    with_debug_level = 0;
    fixed_state_for_init = None;
    print_decimals_precision = 2;
  }

let accessed_global_args = Hash_set.create (module String)

let read_cmdline_or_env_var n =
  let with_debug = settings.with_debug_level > 0 && not (Hash_set.mem accessed_global_args n) in
  let env_variants = [ "ocannl_" ^ n; "ocannl-" ^ n ] in
  let env_variants = List.concat_map env_variants ~f:(fun n -> [ n; String.uppercase n ]) in
  let cmd_variants = List.concat_map env_variants ~f:(fun n -> [ n; "-" ^ n; "--" ^ n ]) in
  let cmd_variants = List.concat_map cmd_variants ~f:(fun n -> [ n; n ^ "_"; n ^ "-"; n ^ "=" ]) in
  match
    Array.find_map (Core.Sys.get_argv ()) ~f:(fun arg ->
        List.find_map cmd_variants ~f:(fun prefix ->
            Option.some_if (String.is_prefix ~prefix arg) (prefix, arg)))
  with
  | Some (p, arg) ->
      let result = String.(lowercase @@ drop_prefix arg (length p)) in
      if with_debug then Stdio.printf "Found %s, commandline %s\n%!" result arg;
      Some result
  | None -> (
      match
        List.find_map env_variants ~f:(fun env_n ->
            Option.(
              join
              @@ map (Core.Sys.getenv env_n) ~f:(fun v -> if String.is_empty v then None else Some (env_n, v))))
      with
      | None | Some (_, "") -> None
      | Some (p, arg) ->
          let result = String.lowercase arg in
          if with_debug then Stdio.printf "Found %s, environment %s\n%!" result p;
          Some result)

let config_file_args =
  match read_cmdline_or_env_var "no_config_file" with
  | None | Some "false" ->
      let read = Stdio.In_channel.read_lines in
      let fname, config_lines =
        let rev_dirs = List.rev @@ Filename_base.parts @@ Stdlib.Sys.getcwd () in
        let rec find_up = function
          | [] -> failwith "OCANNL could not find the ocannl_config file along current path"
          | _ :: tl as rev_dirs -> (
              let fname = Filename_base.of_parts (List.rev @@ ("ocannl_config" :: rev_dirs)) in
              try (fname, read fname) with Sys_error _ -> find_up tl)
        in
        find_up rev_dirs
      in
      Stdio.printf "\nWelcome to OCANNL! Reading configuration defaults from %s.\n%!" fname;
      config_lines
      |> List.map ~f:(String.split ~on:'=')
      |> List.filter_map ~f:(function
           | [] -> None
           | key :: [ v ] ->
               let key =
                 String.(lowercase @@ strip ~drop:(fun c -> equal_char '-' c || equal_char ' ' c) key)
               in
               let key = if String.is_prefix key ~prefix:"ocannl" then String.drop_prefix key 6 else key in
               Some (String.strip ~drop:(equal_char '_') key, v)
           | _ ->
               failwith @@ "OCANNL: invalid syntax in the config file " ^ fname
               ^ ", should have a single '=' on each non-empty line")
      |> Hashtbl.of_alist_exn (module String)
  | Some _ ->
      Stdio.printf "\nWelcome to OCANNL! Configuration defaults file is disabled.\n%!";
      Hashtbl.create (module String)

(** Retrieves [arg_name] argument from the command line or from an environment variable, returns [default] if
    none found. *)
let get_global_arg ~default ~arg_name:n =
  let with_debug = settings.with_debug_level > 0 && not (Hash_set.mem accessed_global_args n) in
  if with_debug then
    Stdio.printf "Retrieving commandline, environment, or config file variable ocannl_%s\n%!" n;
  let result =
    Option.value_or_thunk (read_cmdline_or_env_var n) ~default:(fun () ->
        match Hashtbl.find config_file_args n with
        | Some v ->
            if with_debug then Stdio.printf "Found %s, in the config file\n%!" v;
            v
        | None ->
            if with_debug then Stdio.printf "Not found, using default %s\n%!" default;
            default)
  in
  Hash_set.add accessed_global_args n;
  result

let () =
  settings.with_debug_level <- Int.of_string @@ get_global_arg ~arg_name:"with_debug_level" ~default:"0";
  settings.debug_log_from_routines <-
    Bool.of_string @@ get_global_arg ~arg_name:"debug_log_from_routines" ~default:"false";
  settings.debug_memory_locations <-
    Bool.of_string @@ get_global_arg ~arg_name:"debug_memory_locations" ~default:"false";
  settings.output_debug_files_in_run_directory <-
    Bool.of_string @@ get_global_arg ~arg_name:"output_debug_files_in_run_directory" ~default:"false";
  settings.fixed_state_for_init <-
    (let seed = get_global_arg ~arg_name:"fixed_state_for_init" ~default:"" in
     if String.is_empty seed then None else Some (Int.of_string seed));
  settings.print_decimals_precision <-
    Int.of_string @@ get_global_arg ~arg_name:"print_decimals_precision" ~default:"2"

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
  let verbose_entry_ids =
    Bool.of_string @@ get_global_arg ~default:"false" ~arg_name:"logs_verbose_entry_ids"
  in
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
  let toc_entry_minimal_depth =
    let arg = get_global_arg ~default:"" ~arg_name:"toc_entry_minimal_depth" in
    if String.is_empty arg then [] else [ Minidebug_runtime.Minimal_depth (Int.of_string arg) ]
  in
  let toc_entry_minimal_size =
    let arg = get_global_arg ~default:"" ~arg_name:"toc_entry_minimal_size" in
    if String.is_empty arg then [] else [ Minidebug_runtime.Minimal_size (Int.of_string arg) ]
  in
  let toc_entry_minimal_span =
    let arg = get_global_arg ~default:"" ~arg_name:"toc_entry_minimal_span" in
    if String.is_empty arg then []
    else
      let arg, period = (String.prefix arg (String.length arg - 2), String.suffix arg 2) in
      let period =
        match period with
        | "ns" -> Mtime.Span.ns
        | "us" -> Mtime.Span.us
        | "ms" -> Mtime.Span.ms
        | _ ->
            invalid_arg @@ "ocannl_toc_entry_minimal_span setting should end with one of: ns, us, ms; found: "
            ^ period
      in
      [ Minidebug_runtime.Minimal_span Mtime.Span.(Int.of_string arg * period) ]
  in
  let toc_entry =
    Minidebug_runtime.And (toc_entry_minimal_depth @ toc_entry_minimal_size @ toc_entry_minimal_span)
  in
  if flushing then
    Minidebug_runtime.debug_flushing ~filename ~time_tagged ~elapsed_times ~print_entry_ids ~verbose_entry_ids
      ~global_prefix:name ~for_append:false (* ~log_level *) ()
  else
    Minidebug_runtime.forget_printbox
    @@ Minidebug_runtime.debug_file ~time_tagged ~elapsed_times ~location_format ~print_entry_ids
         ~verbose_entry_ids ~global_prefix:name ~toc_flame_graph:true ~flame_graph_separation:50 ~toc_entry
         ~for_append:false ~max_inline_sexp_length:120 ~hyperlink ~toc_specific_hyperlink:""
         ~highlight_terms:Re.(alt [])
         ~exclude_on_path:Re.(str "env")
         ~values_first_mode:true ~backend ~log_level ?snapshot_every_sec filename

module Debug_runtime = (val get_debug "")

[%%global_debug_log_level Nothing]
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

(** [parallel_merge merge num_devices] progressively invokes the pairwise [merge] callback, converging on the
    0th position, with [from] ranging from [1] to [num_devices - 1], and [to_ < from]. *)
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

type atomic_bool = bool Atomic.t

let sexp_of_atomic_bool flag = sexp_of_bool @@ Atomic.get flag
let ( !@ ) = Atomic.get

type waiter = {
  await : keep_waiting:(unit -> bool) -> unit -> bool;
      (** Returns [true] if the waiter was not already waiting (in another thread) and waiting was needed
          ([keep_waiting] always returned true). *)
  release_if_waiting : unit -> bool;
      (** Returns [true] if the waiter both was waiting and was not already released. *)
  is_waiting : unit -> bool;
  finalize : unit -> unit;
}
(** Note: this waiter is meant for sequential waiting. *)

let waiter ~name:_ () =
  let is_open = Atomic.make true in
  (* TODO: since OCaml 5.2, use [make_contended] for at least [is_released] and maybe [is_waiting]. *)
  let is_released = Atomic.make false in
  let is_waiting = Atomic.make false in
  let pipe_inp, pipe_out = Unix.pipe ~cloexec:true () in
  let await ~keep_waiting =
    let rec wait () =
      let need_waiting = keep_waiting () in
      if
        need_waiting
        &&
        let inp_pipes, _, _ = Unix.select [ pipe_inp ] [] [] 5.0 in
        List.is_empty inp_pipes
      then wait ()
      else need_waiting
    in
    fun () ->
      if Atomic.compare_and_set is_waiting false true then (
        Atomic.set is_released false;
        let result =
          if wait () then (
            let n = Unix.read pipe_inp (Bytes.create 1) 0 1 in
            assert (n = 1);
            true)
          else false
        in
        assert (Atomic.compare_and_set is_waiting true false);
        result)
      else false
  in
  let release_if_waiting () =
    let result =
      if !@is_waiting && Atomic.compare_and_set is_released false true then (
        let n = Unix.write pipe_out (Bytes.create 1) 0 1 in
        assert (n = 1);
        true)
      else false
    in
    result
  in
  let finalize () =
    if Atomic.compare_and_set is_open true false then (
      Unix.close pipe_inp;
      Unix.close pipe_out)
  in
  let is_waiting () = !@is_waiting in
  { await; release_if_waiting; is_waiting; finalize }

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

type requirement =
  | Skip
  | Required
  | Optional of { callback_if_missing : unit -> unit [@sexp.opaque] [@compare.ignore] }
[@@deriving compare, sexp]

let get_debug_formatter ~fname =
  if settings.output_debug_files_in_run_directory then
    let f = Stdio.Out_channel.create fname in
    let ppf = Stdlib.Format.formatter_of_out_channel f in
    Some ppf
  else None

exception User_error of string

let header_sep =
  let open Re in
  compile (seq [ str " "; opt any; str "="; str " " ])

let%diagn_rt_sexp log_trace_tree logs =
  let rec loop = function
    | [] -> []
    | line :: more when String.is_empty line -> loop more
    | "COMMENT: end" :: more -> more
    | comment :: more when String.is_prefix comment ~prefix:"COMMENT: " ->
        let more =
          [%log_entry
            String.chop_prefix_exn ~prefix:"COMMENT: " comment;
            loop more]
        in
        loop more
    | source :: trace :: more when String.is_prefix source ~prefix:"# " ->
        (let source =
           String.concat ~sep:"\n" @@ String.split ~on:'$' @@ String.chop_prefix_exn ~prefix:"# " source
         in
         match split_with_seps header_sep trace with
         | [] | [ "" ] -> [%log source]
         | header1 :: assign1 :: header2 :: body ->
             let header = String.concat [ header1; assign1; header2 ] in
             let body = String.concat body in
             let _message = Sexp.(List [ Atom header; Atom source; Atom body ]) in
             [%log (_message : Sexp.t)]
         | _ -> [%log source, trace]);
        loop more
    | _line :: more ->
        [%log _line];
        loop more
  in
  let rec loop_logs logs =
    let output = loop logs in
    if not (List.is_empty output) then
      [%log_entry
        "TRAILING LOGS:";
        loop_logs output]
  in
  loop_logs logs

type 'a mutable_list = Empty | Cons of { hd : 'a; mutable tl : 'a mutable_list }
[@@deriving equal, sexp, variants]

let insert ~next = function
  | Empty -> Cons { hd = next; tl = Empty }
  | Cons cons ->
      cons.tl <- Cons { hd = next; tl = cons.tl };
      cons.tl

let tl_exn = function Empty -> raise @@ Not_found_s (Sexp.Atom "mutable_list.tl_exn") | Cons { tl; _ } -> tl

let pp_file ~name pp_v v =
  let column_width = 110 in
  let f_name =
    if settings.output_debug_files_in_run_directory then name ^ ".ml"
    else Stdlib.Filename.temp_file (name ^ "_") ".ml"
  in
  (* (try Stdlib.Sys.remove f_name with _ -> ()); *)
  let oc = Out_channel.open_text f_name in
  (* FIXME(#32): the following outputs truncated source code -- missing the last line: {[ *
       let ppf = Stdlib.Format.formatter_of_out_channel oc in
       Stdlib.Format.pp_set_geometry Caml.Format.str_formatter
       ~max_indent:(column_width/2) ~margin:column_width;
       let () = format_low_level ~as_toplevel:true ppf compiled in
       let () = Stdio.Out_channel.close oc in
       let () = Stdio.printf "FIXME(32): file content:\n%s\nEND file content\n%!"
       (Stdio.In_channel.read_all fname) in
   * ]} Defensive variant: *)
  Stdlib.Format.pp_set_geometry Stdlib.Format.str_formatter ~max_indent:(column_width / 2)
    ~margin:column_width;
  let result = pp_v Stdlib.Format.str_formatter v in
  let contents = Stdlib.Format.flush_str_formatter () in
  Stdio.Out_channel.output_string oc contents;
  Stdio.Out_channel.flush oc;
  Stdio.Out_channel.close oc;
  (f_name, result)
