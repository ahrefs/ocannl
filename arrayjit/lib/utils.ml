open Base

module Set_O = struct
  let ( + ) = Set.union
  let ( - ) = Set.diff
  let ( & ) = Set.inter

  let ( -* ) s1 s2 =
    Set.of_sequence (Set.comparator_s s1)
    @@ Sequence.map ~f:Either.value @@ Set.symmetric_diff s1 s2
end

let no_ints = Set.empty (module Int)
let one_int = Set.singleton (module Int)

let map_merge m1 m2 ~f =
  Map.merge m1 m2 ~f:(fun ~key:_ m ->
      match m with `Right v | `Left v -> Some v | `Both (v1, v2) -> Some (f v1 v2))

let mref_add mref ~key ~data ~or_ =
  match Map.add !mref ~key ~data with
  | `Ok m -> mref := m
  | `Duplicate -> or_ (Map.find_exn !mref key)

let mref_add_missing mref key ~f =
  if Map.mem !mref key then () else mref := Map.add_exn !mref ~key ~data:(f ())

type settings = {
  mutable log_level : int;
  mutable debug_log_from_routines : bool;
      (** If the [debug_log_from_routines] flag is true _and_ the flag [log_level > 1], backends
          should generate code (e.g. fprintf statements) to log the execution, and arrange for the
          logs to be emitted via ppx_minidebug. *)
  mutable output_debug_files_in_build_directory : bool;
      (** Writes compilation related files in the [build_files] subdirectory of the run directory
          (additional files, or files that would otherwise be in temp directory). When both
          [output_debug_files_in_build_directory = true] and [log_level > 1], compilation should
          also preserve debug and line information for runtime debugging. *)
  mutable fixed_state_for_init : int option;
  mutable print_decimals_precision : int;
      (** When rendering arrays etc., outputs this many decimal digits. *)
  mutable check_half_prec_constants_cutoff : float option;
      (** If given, generic code optimization should fail if a half precision FP16 constant exceeds
          the cutoff. *)
  mutable automatic_host_transfers : bool;
      (** If true, [from_host] and [to_host] happen automatically in specific situations.
          - When a host array is about to be read, we transfer to host from the context that most
            recently updated the node.
          - When a routine is about to be run, we transfer the routine's inputs from host to the
            routine's context if the host array was not yet transfered since its creation or most
            recent modification. *)
  mutable default_prng_variant : string;
      (** The default variant of threefry4x32 PRNG to use. Options: "crypto" (20 rounds) or "light"
          (2 rounds). Defaults to "light" for better performance. *)
}
[@@deriving sexp]

let settings =
  {
    log_level = 0;
    debug_log_from_routines = false;
    output_debug_files_in_build_directory = false;
    fixed_state_for_init = None;
    print_decimals_precision = 2;
    check_half_prec_constants_cutoff = Some (2. **. 14.);
    automatic_host_transfers = true;
    default_prng_variant = "light";
  }

let accessed_global_args = Hash_set.create (module String)
let str_nonempty ~f s = if String.is_empty s then None else Some (f s)
let pair a b = (a, b)

let read_cmdline_or_env_var n =
  let with_debug =
    (settings.log_level > 0 || equal_string n "log_level")
    && not (Hash_set.mem accessed_global_args n)
  in
  let env_variants = [ "ocannl_" ^ n; "ocannl-" ^ n ] in
  let env_variants = List.concat_map env_variants ~f:(fun n -> [ n; String.uppercase n ]) in
  let cmd_variants = List.concat_map env_variants ~f:(fun n -> [ "-" ^ n; "--" ^ n; n ]) in
  let cmd_variants = List.concat_map cmd_variants ~f:(fun n -> [ n ^ "_"; n ^ "-"; n ^ "="; n ]) in
  match
    Array.find_map Stdlib.Sys.argv ~f:(fun arg ->
        List.find_map cmd_variants ~f:(fun prefix ->
            Option.some_if (String.is_prefix ~prefix arg) (prefix, arg)))
  with
  | Some (p, arg) ->
      let result = String.(drop_prefix arg (length p)) in
      if with_debug then Stdio.printf "Found %s, commandline %s\n%!" result arg;
      Some result
  | None -> (
      match
        List.find_map env_variants ~f:(fun env_n ->
            Option.(join @@ map (Stdlib.Sys.getenv_opt env_n) ~f:(str_nonempty ~f:(pair env_n))))
      with
      | None | Some (_, "") -> None
      | Some (p, result) ->
          if with_debug then Stdio.printf "Found %s, environment %s\n%!" result p;
          Some result)

(* Originally from the library core.filename_base. *)
let filename_parts filename =
  let rec loop acc filename =
    match (Stdlib.Filename.dirname filename, Stdlib.Filename.basename filename) with
    | ("." as base), "." -> base :: acc
    | ("/" as base), "/" -> base :: acc
    | disk, base when String.is_suffix disk ~suffix:":\\" -> disk :: base :: acc
    | rest, dir -> loop (dir :: acc) rest
  in
  loop [] filename

(* Originally from the library core.filename_base. *)
let filename_of_parts = function
  | [] -> invalid_arg "Utils.filename_of_parts: empty parts list"
  | root :: rest -> List.fold rest ~init:root ~f:Stdlib.Filename.concat

let config_file_args =
  let suppress_welcome_message () =
    Option.value_map ~default:false ~f:Bool.of_string
    @@ read_cmdline_or_env_var "suppress_welcome_message"
  in
  match read_cmdline_or_env_var "no_config_file" with
  | None | Some "false" ->
      let read = Stdio.In_channel.read_lines in
      let fname, config_lines =
        let rev_dirs = List.rev @@ filename_parts @@ Stdlib.Sys.getcwd () in
        let rec find_up = function
          | [] ->
              if not (suppress_welcome_message ()) then
                Stdio.printf
                  "\nWelcome to OCANNL! No ocannl_config file found along current path.\n%!";
              ("", [])
          | _ :: tl as rev_dirs -> (
              let fname = filename_of_parts (List.rev @@ ("ocannl_config" :: rev_dirs)) in
              try (fname, read fname) with Sys_error _ -> find_up tl)
        in
        find_up rev_dirs
      in
      let result =
        config_lines
        |> List.filter ~f:(fun l ->
               not (String.is_prefix ~prefix:"~~" l || String.is_prefix ~prefix:"#" l))
        |> List.map ~f:(String.split ~on:'=')
        |> List.filter_map ~f:(function
             | [] -> None
             | [ s ] when String.is_empty s -> None
             | key :: [ v ] ->
                 let key =
                   String.(
                     lowercase @@ strip ~drop:(fun c -> equal_char '-' c || equal_char ' ' c) key)
                 in
                 let key =
                   if String.is_prefix key ~prefix:"ocannl" then
                     String.drop_prefix key 6 |> String.strip ~drop:(equal_char '_')
                   else key
                 in
                 str_nonempty ~f:(pair key) v
             | l ->
                 failwith @@ "OCANNL: invalid syntax in the config file " ^ fname
                 ^ ", should have a single '=' on each non-empty line, found: " ^ String.concat l)
        |> Hashtbl.of_alist (module String)
        |> function
        | `Ok h -> h
        | `Duplicate_key key ->
            failwith @@ "OCANNL: duplicate key in config file " ^ fname ^ ": " ^ key
      in
      if
        String.length fname > 0
        && (not (suppress_welcome_message ()))
        && not
             (Option.value_map ~default:false ~f:Bool.of_string
             @@ Hashtbl.find result "suppress_welcome_message")
      then Stdio.printf "\nWelcome to OCANNL! Reading configuration defaults from %s.\n%!" fname;
      result
  | Some _ ->
      if not (suppress_welcome_message ()) then
        Stdio.printf "\nWelcome to OCANNL! Configuration defaults file is disabled.\n%!";
      Hashtbl.create (module String)

(** Retrieves [arg_name] argument from the command line or from an environment variable, returns
    [default] if none found. *)
let get_global_arg ~default ~arg_name:n =
  let with_debug =
    (settings.log_level > 0 || equal_string n "log_level")
    && not (Hash_set.mem accessed_global_args n)
  in
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

let get_global_flag ~default ~arg_name:n =
  let s = get_global_arg ~default:(if default then "true" else "false") ~arg_name:n in
  match String.lowercase s with
  | "true" | "1" -> true
  | "false" | "0" -> false
  | _ -> invalid_arg @@ "ocannl_" ^ n ^ " setting should be a boolean; found: " ^ s

let original_log_level =
  let log_level =
    let s = String.strip @@ get_global_arg ~default:"1" ~arg_name:"log_level" in
    match Int.of_string_opt s with
    | Some ll -> ll
    | None -> invalid_arg @@ "ocannl_log_level setting should be an integer; found: " ^ s
  in
  settings.log_level <- log_level;
  log_level

(* Originally from the library core.filename_base. *)
let filename_concat p1 p2 =
  if String.is_empty p1 then
    invalid_arg
    @@ "Utils.filename_concat called with an empty string as its first argument, second argument: "
    ^ p2;
  let rec collapse_trailing s =
    match String.rsplit2 s ~on:'/' with
    | Some ("", ("." | "")) -> ""
    | Some (s, ("." | "")) -> collapse_trailing s
    | None | Some _ -> s
  in
  let rec collapse_leading s =
    match String.lsplit2 s ~on:'/' with
    | Some (("." | ""), s) -> collapse_leading s
    | Some _ | None -> s
  in
  collapse_trailing p1 ^ "/" ^ collapse_leading p2

let clean_filename fname =
  let fname = String.strip fname in
  let fname =
    String.map
      ~f:(fun c -> if List.exists ~f:(equal_char c) [ '/'; '\\'; ':' ] then '-' else c)
      fname
  in
  fname

let build_file fname =
  let build_files_dir = "build_files" in
  (try assert (Stdlib.Sys.is_directory build_files_dir)
   with Stdlib.Sys_error _ -> Stdlib.Sys.mkdir build_files_dir 0o777);
  filename_concat build_files_dir @@ clean_filename fname

let diagn_log_file fname =
  let log_files_dir = "log_files" in
  (try assert (Stdlib.Sys.is_directory log_files_dir)
   with Stdlib.Sys_error _ -> (
     (* NOTE: is this can be called concurrently. *)
     try Stdlib.Sys.mkdir log_files_dir 0o777 with Stdlib.Sys_error _ -> ()));
  filename_concat log_files_dir @@ clean_filename fname

let () =
  (* Cleanup needs to happen before get_local_debug_runtime (or any other code is run). *)
  let remove_dir_if_exists dirname =
    if Stdlib.Sys.file_exists dirname && Stdlib.Sys.is_directory dirname then
      try
        Array.iter (Stdlib.Sys.readdir dirname) ~f:(fun fname ->
            Stdlib.Sys.remove (Stdlib.Filename.concat dirname fname));
        Stdlib.Sys.rmdir dirname
      with exn ->
        Stdio.eprintf "Failed to delete directory %s: %s\n%!" dirname (Exn.to_string exn)
    else if Stdlib.Sys.file_exists dirname then
      try Stdlib.Sys.remove dirname
      with exn ->
        Stdio.eprintf "Failed to delete file %s (expected a directory): %s\n%!" dirname
          (Exn.to_string exn)
  in
  let clean_up_log_files_on_startup =
    get_global_flag ~default:true ~arg_name:"clean_up_log_files_on_startup"
  in
  if clean_up_log_files_on_startup then remove_dir_if_exists "log_files";
  let clean_up_build_files_on_startup =
    get_global_flag ~default:true ~arg_name:"clean_up_build_files_on_startup"
  in
  if clean_up_build_files_on_startup then remove_dir_if_exists "build_files"

let get_local_debug_runtime =
  let snapshot_every_sec =
    Option.join
    @@ str_nonempty ~f:Float.of_string_opt
    @@ get_global_arg ~default:"" ~arg_name:"snapshot_every_sec"
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
        @@ "ocannl_elapsed_times setting should be not_reported, seconds or milliseconds, \
            microseconds or nanoseconds; found: " ^ s
  in
  let location_format =
    match String.lowercase @@ get_global_arg ~default:"beg_pos" ~arg_name:"location_format" with
    | "no_location" -> Minidebug_runtime.No_location
    | "file_only" -> File_only
    | "beg_line" -> Beg_line
    | "beg_pos" -> Beg_pos
    | "range_line" -> Range_line
    | "range_pos" -> Range_pos
    | s ->
        invalid_arg
        @@ "ocannl_location_format setting should be one of: no_location, file_only, beg_line, \
            beg_pos, range_line, range_pos; found: " ^ s
  in
  let flushing, toc_flame_graph, backend =
    match
      String.lowercase @@ String.strip @@ get_global_arg ~default:"html" ~arg_name:"debug_backend"
    with
    | "text" -> (false, false, `Text)
    | "html" -> (false, true, `Html Minidebug_runtime.default_html_config)
    | "markdown" -> (false, false, `Markdown Minidebug_runtime.default_md_config)
    | "flushing" -> (true, false, `Text)
    | s ->
        invalid_arg
        @@ "ocannl_debug_backend setting should be text, html, markdown or flushing; found: " ^ s
  in
  let hyperlink = get_global_arg ~default:"./" ~arg_name:"hyperlink_prefix" in
  let print_entry_ids = get_global_flag ~default:false ~arg_name:"logs_print_entry_ids" in
  let verbose_entry_ids = get_global_flag ~default:false ~arg_name:"logs_verbose_entry_ids" in
  let log_main_domain_to_stdout =
    get_global_flag ~default:false ~arg_name:"log_main_domain_to_stdout"
  in
  let file_stem =
    if log_main_domain_to_stdout then None
    else Some (get_global_arg ~default:"debug" ~arg_name:"log_file_stem")
  in
  let filename = Option.map file_stem ~f:(fun stem -> diagn_log_file @@ stem) in
  let prev_run_file =
    let prefix = str_nonempty ~f:Fn.id @@ get_global_arg ~default:"" ~arg_name:"prev_run_prefix" in
    Option.map2 prefix file_stem ~f:(fun prefix stem -> diagn_log_file @@ prefix ^ stem ^ ".raw")
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
            invalid_arg
            @@ "ocannl_toc_entry_minimal_span setting should end with one of: ns, us, ms; found: "
            ^ period
      in
      [ Minidebug_runtime.Minimal_span Mtime.Span.(Int.of_string arg * period) ]
  in
  let toc_entry =
    Minidebug_runtime.And (toc_entry_minimal_depth @ toc_entry_minimal_size @ toc_entry_minimal_span)
  in
  let debug_highlights =
    let arg = get_global_arg ~default:"" ~arg_name:"debug_highlights" in
    if String.is_empty arg then [] else String.split arg ~on:'|'
  in
  let highlight_re =
    let arg = get_global_arg ~default:"" ~arg_name:"debug_highlight_pcre" in
    Option.to_list @@ str_nonempty ~f:Re.Pcre.re arg
  in
  let highlight_terms = Re.(alt (highlight_re @ List.map debug_highlights ~f:str)) in
  let diff_ignore_pattern =
    str_nonempty ~f:Re.Pcre.re @@ get_global_arg ~default:"" ~arg_name:"diff_ignore_pattern_pcre"
  in
  let max_distance_factor =
    str_nonempty ~f:Int.of_string @@ get_global_arg ~default:"" ~arg_name:"diff_max_distance_factor"
  in
  let entry_id_pairs =
    let pairs_str = get_global_arg ~default:"" ~arg_name:"debug_entry_id_pairs" in
    if String.is_empty pairs_str then []
    else
      String.split pairs_str ~on:';'
      |> List.filter_map ~f:(fun pair_str ->
             match String.split pair_str ~on:',' with
             | [ id1; id2 ] ->
                 Option.try_with (fun () ->
                     (Int.of_string (String.strip id1), Int.of_string (String.strip id2)))
             | _ -> None)
  in
  let truncate_children =
    let arg = get_global_arg ~default:"" ~arg_name:"debug_log_truncate_children" in
    if String.is_empty arg then None else Some (Int.of_string arg)
  in
  let name = get_global_arg ~default:"debug" ~arg_name:"log_file_stem" in
  match (flushing, filename) with
  | true, None ->
      Minidebug_runtime.prefixed_runtime_flushing ~time_tagged ~elapsed_times ~print_entry_ids
        ~verbose_entry_ids ~global_prefix:name ~for_append:false ~log_level:original_log_level ()
  | true, Some filename ->
      Minidebug_runtime.local_runtime_flushing ~time_tagged ~elapsed_times ~print_entry_ids
        ~verbose_entry_ids ~global_prefix:name ~for_append:false ~log_level:original_log_level
        filename
  | false, None ->
      Minidebug_runtime.prefixed_runtime ~time_tagged ~elapsed_times ~location_format
        ~print_entry_ids ~verbose_entry_ids ~global_prefix:name ~toc_entry
        ~toc_specific_hyperlink:"" ~highlight_terms ?truncate_children
        ~exclude_on_path:Re.(str "env")
        ~log_level:original_log_level ?snapshot_every_sec ()
  | false, Some filename ->
      Minidebug_runtime.local_runtime ~time_tagged ~elapsed_times ~location_format ~print_entry_ids
        ~verbose_entry_ids ~global_prefix:name ~toc_flame_graph ~flame_graph_separation:50
        ~toc_entry ~for_append:false ~max_inline_sexp_length:120 ~hyperlink
        ~toc_specific_hyperlink:"" ~highlight_terms ?truncate_children
        ~exclude_on_path:Re.(str "env")
        ~backend ~log_level:original_log_level ?snapshot_every_sec ?prev_run_file
        ?diff_ignore_pattern ?max_distance_factor ~entry_id_pairs filename

let _get_local_debug_runtime = get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_UTILS=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_UTILS"]

(* [%%global_debug_interrupts { max_nesting_depth = 100; max_num_children = 1000 }] *)

let%diagn_sexp set_log_level level =
  settings.log_level <- level;
  [%log
    "Set log_level to",
    (Debug_runtime.log_level := level;
     level
      : int)]

let restore_settings () =
  set_log_level original_log_level;
  settings.debug_log_from_routines <-
    get_global_flag ~default:false ~arg_name:"debug_log_from_routines";
  settings.output_debug_files_in_build_directory <-
    get_global_flag ~default:false ~arg_name:"output_debug_files_in_build_directory";
  settings.fixed_state_for_init <-
    (let seed = get_global_arg ~arg_name:"fixed_state_for_init" ~default:"" in
     if String.is_empty seed then None else Some (Int.of_string seed));
  settings.print_decimals_precision <-
    Int.of_string @@ get_global_arg ~arg_name:"print_decimals_precision" ~default:"2";
  settings.check_half_prec_constants_cutoff <-
    Float.of_string_opt
    @@ get_global_arg ~arg_name:"check_half_prec_constants_cutoff" ~default:"16384.0";
  settings.automatic_host_transfers <-
    get_global_flag ~default:true ~arg_name:"automatic_host_transfers";
  settings.default_prng_variant <- get_global_arg ~default:"light" ~arg_name:"default_prng_variant"

let () = restore_settings ()
let with_runtime_debug () = settings.output_debug_files_in_build_directory && settings.log_level > 1
let debug_log_from_routines () = settings.debug_log_from_routines && settings.log_level > 1
let never_capture_stdout () = get_global_flag ~default:false ~arg_name:"never_capture_stdout"

let enable_runtime_debug () =
  settings.output_debug_files_in_build_directory <- true;
  set_log_level @@ max 2 settings.log_level

let rec union_find ~equal map ~key ~rank =
  match Map.find map key with
  | None -> (key, rank)
  | Some data ->
      if equal key data then (key, rank) else union_find ~equal map ~key:data ~rank:(rank + 1)

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

(** Returns the multiset difference of [l1] and [l2], where [l1] and [l2] must be sorted in
    increasing order. *)
let sorted_diff ~compare l1 l2 =
  let rec loop acc l1 l2 =
    match (l1, l2) with
    | [], _ -> List.rev acc
    | l1, [] -> List.rev_append acc l1
    | h1 :: t1, h2 :: t2 -> (
        match compare h1 h2 with
        | c when c < 0 -> loop (h1 :: acc) t1 l2
        | 0 ->
            (* Depending on this line this can be either a set diff or a multiset: currently it's
               multiset diff. *)
            loop acc t1 t2
        | _ -> loop acc l1 t2)
  in
  (loop [] l1 l2 [@nontail])

(** Removes the first occurrence of an element from the list that is equal to the given element. *)
let remove_elem ~equal elem l =
  let rec loop acc = function
    | [] -> List.rev acc
    | hd :: tl -> if equal elem hd then List.rev_append acc tl else loop (hd :: acc) tl
  in
  loop [] l

(** [parallel_merge merge num_devices] progressively invokes the pairwise [merge] callback,
    converging on the 0th position, with [from] ranging from [1] to [num_devices - 1], and
    [to_ < from]. *)
let parallel_merge merge (num_devices : int) =
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

let ( !@ ) = Atomic.get

type atomic_bool = bool Atomic.t

let sexp_of_atomic_bool flag = sexp_of_bool @@ Atomic.get flag

type atomic_int = int Atomic.t

let sexp_of_atomic_int flag = sexp_of_int @@ Atomic.get flag

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

let default_indent = ref 2

let doc_of_sexp sexp =
  let open Sexp in
  let open Int in
  let module Bytes = Stdlib.Bytes in
  let must_escape str =
    let len = String.length str in
    len = 0
    ||
    let rec loop str ix =
      match str.[ix] with
      | '"' | '(' | ')' | ';' | '\\' -> true
      | '|' ->
          ix > 0
          &&
          let next = ix - 1 in
          Char.equal str.[next] '#' || loop str next
      | '#' ->
          ix > 0
          &&
          let next = ix - 1 in
          Char.equal str.[next] '|' || loop str next
      | '\000' .. '\032' | '\127' .. '\255' -> true
      | _ -> ix > 0 && loop str (ix - 1)
    in
    loop str (len - 1)
  in

  let escaped s =
    let n = ref 0 in
    for i = 0 to String.length s - 1 do
      n :=
        !n
        +
        match String.unsafe_get s i with
        | '\"' | '\\' | '\n' | '\t' | '\r' | '\b' -> 2
        | ' ' .. '~' -> 1
        | _ -> 4
    done;
    if !n = String.length s then s
    else
      let s' = Bytes.create !n in
      n := 0;
      for i = 0 to String.length s - 1 do
        (match String.unsafe_get s i with
        | ('\"' | '\\') as c ->
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n c
        | '\n' ->
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n 'n'
        | '\t' ->
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n 't'
        | '\r' ->
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n 'r'
        | '\b' ->
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n 'b'
        | ' ' .. '~' as c -> Bytes.unsafe_set s' !n c
        | c ->
            let a = Stdlib.Char.code c in
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n (Stdlib.Char.chr (48 + (a / 100)));
            incr n;
            Bytes.unsafe_set s' !n (Stdlib.Char.chr (48 + (a / 10 % 10)));
            incr n;
            Bytes.unsafe_set s' !n (Stdlib.Char.chr (48 + (a % 10))));
        incr n
      done;
      Bytes.unsafe_to_string s'
  in

  let esc_str str =
    let estr = escaped str in
    let elen = String.length estr in
    let res = Bytes.create (elen + 2) in
    Bytes.blit_string estr 0 res 1 elen;
    Bytes.unsafe_set res 0 '"';
    Bytes.unsafe_set res (elen + 1) '"';
    Bytes.unsafe_to_string res
  in

  let index_of_newline str start = Stdlib.String.index_from_opt str start '\n' in

  let get_substring str index end_pos_opt =
    let end_pos = match end_pos_opt with None -> String.length str | Some end_pos -> end_pos in
    String.sub str ~pos:index ~len:(end_pos - index)
  in

  let is_one_line str =
    match index_of_newline str 0 with
    | None -> true
    | Some index -> Int.(index + 1 = String.length str)
  in

  let open PPrint in
  let doc_maybe_esc_str str =
    if not (must_escape str) then string str
    else if is_one_line str then string (esc_str str)
    else
      let rec loop index acc =
        let next_newline = index_of_newline str index in
        let next_line = get_substring str index next_newline in
        let acc = acc ^^ string (escaped next_line) in
        match next_newline with
        | None -> acc
        | Some newline_index ->
            loop (newline_index + 1) (acc ^^ string "\\" ^^ hardline ^^ string "\\n")
      in
      (* the leading space is to line up the lines *)
      string " \"" ^^ loop 0 empty ^^ string "\""
  in

  let rec doc_of_sexp_indent indent = function
    | Atom str -> doc_maybe_esc_str str
    | List (h :: t) ->
        group (string "(" ^^ nest indent (doc_of_sexp_indent indent h ^^ doc_of_sexp_rest indent t))
    | List [] -> string "()"
  and doc_of_sexp_rest indent = function
    | h :: t -> space ^^ doc_of_sexp_indent indent h ^^ doc_of_sexp_rest indent t
    | [] -> string ")"
  in

  doc_of_sexp_indent !default_indent sexp

let output_to_build_file ~fname =
  if settings.output_debug_files_in_build_directory then
    let f = Stdio.Out_channel.create @@ build_file fname in
    let print doc =
      PPrint.ToChannel.pretty 0.7 100 f doc;
      Stdio.Out_channel.flush f
    in
    Some print
  else None

let get_debug_output_channel ~fname =
  if settings.output_debug_files_in_build_directory then
    Some (Stdio.Out_channel.create @@ build_file fname)
  else None

exception User_error of string

let header_sep =
  let open Re in
  compile (seq [ str " "; opt any; str "="; str " " ])

let%diagn_sexp log_trace_tree _logs =
  [%log_block
    "trace tree";
    let sep s = String.concat ~sep:"\n" @@ String.split ~on:'$' s in
    let rec loop = function
      | [] -> []
      | line :: more when String.is_empty line -> loop more
      | "COMMENT: end" :: more -> more
      | comment :: more when String.is_prefix comment ~prefix:"COMMENT: " ->
          let more =
            [%log_entry
              sep @@ String.chop_prefix_exn ~prefix:"COMMENT: " comment;
              loop more]
          in
          loop more
      | source :: trace :: more when String.is_prefix source ~prefix:"# " ->
          (let source = sep @@ String.chop_prefix_exn ~prefix:"# " source in
           match split_with_seps header_sep @@ sep trace with
           | [] | [ "" ] -> [%log source]
           | header1 :: assign1 :: header2 :: body ->
               let header = String.concat [ header1; assign1; header2 ] in
               let body = String.concat body in
               let _message = Sexp.(List [ Atom header; Atom source; Atom body ]) in
               [%log (_message : Sexp.t)]
           | _ -> [%log source, trace]);
          loop more
      | _line :: more ->
          [%log sep _line];
          loop more
    in
    let rec loop_logs logs =
      let output = loop logs in
      if not (List.is_empty output) then
        [%log_block
          "TRAILING LOGS:";
          loop_logs output]
    in
    loop_logs _logs]

type 'a mutable_list = Empty | Cons of { hd : 'a; mutable tl : 'a mutable_list }
[@@deriving equal, sexp, variants]

let insert ~next = function
  | Empty -> Cons { hd = next; tl = Empty }
  | Cons cons ->
      cons.tl <- Cons { hd = next; tl = cons.tl };
      cons.tl

let tl_exn = function
  | Empty -> raise @@ Not_found_s (Sexp.Atom "mutable_list.tl_exn")
  | Cons { tl; _ } -> tl

type build_file_channel = { f_path : string; oc : Stdlib.out_channel; finalize : unit -> unit }

let open_build_file ~base_name ~extension : build_file_channel =
  let f_path =
    if settings.output_debug_files_in_build_directory then build_file @@ base_name ^ extension
    else Stdlib.Filename.temp_file (base_name ^ "_") extension
  in
  (* (try Stdlib.Sys.remove f_path with _ -> ()); *)
  let oc = Out_channel.open_text f_path in
  let finalize () =
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc
  in
  { f_path; oc; finalize }

let captured_log_prefix = ref "!@#"

type captured_log_processor = { log_processor_prefix : string; process_logs : string list -> unit }

let captured_log_processors : captured_log_processor list ref = ref []

let add_log_processor ~prefix process_logs =
  captured_log_processors :=
    { log_processor_prefix = prefix; process_logs } :: !captured_log_processors

external input_scan_line : Stdlib.in_channel -> int = "caml_ml_input_scan_line"

let input_line chan =
  let n = input_scan_line chan in
  if n = 0 then raise End_of_file;
  let line = Stdlib.really_input_string chan (abs n) in
  ( n > 0,
    String.chop_suffix_if_exists ~suffix:"\n" @@ String.chop_suffix_if_exists line ~suffix:"\r\n" )

let capture_stdout_logs arg =
  if never_capture_stdout () || not (debug_log_from_routines ()) then arg ()
  else (
    Stdlib.flush Stdlib.stdout;
    (* Ensure previous stdout is flushed *)
    let original_stdout_fd = Unix.dup Unix.stdout in

    let pipe_read_fd, pipe_write_fd = Unix.pipe ~cloexec:true () in
    Unix.dup2 pipe_write_fd Unix.stdout;

    (* pipe_write_fd is now the new Stdlib.stdout, do not close it in parent until done. *)
    (* The reader domain will close pipe_read_fd. *)
    let collected_logs_ref = ref [] in
    let reader_domain_failed = Atomic.make false in

    let reader_domain_logic () =
      let in_channel = Unix.in_channel_of_descr pipe_read_fd in
      (* Create an output channel to the original stdout for immediate passthrough *)
      let orig_out = Unix.out_channel_of_descr (Unix.dup original_stdout_fd) in
      try
        while true do
          let _is_endlined, line = input_line in_channel in
          match String.chop_prefix ~prefix:!captured_log_prefix line with
          | Some logline -> collected_logs_ref := logline :: !collected_logs_ref
          | None ->
              (* Forward non-log lines to original stdout immediately *)
              Stdlib.output_string orig_out (line ^ "\n");
              Stdlib.flush orig_out
        done;
        Stdlib.close_out_noerr orig_out;
        Stdlib.close_in_noerr in_channel (* This closes pipe_read_fd *)
      with
      | End_of_file -> () (* Normal termination of the reader *)
      | exn ->
          Stdlib.close_out_noerr orig_out;
          Atomic.set reader_domain_failed true;
          Stdio.eprintf "Exception in stdout reader domain: %s\\nBacktrace:\\n%s\\n%!"
            (Exn.to_string exn)
            (Stdlib.Printexc.get_backtrace ());
          Stdlib.close_in_noerr in_channel (* This closes pipe_read_fd *);
          Stdlib.Printexc.raise_with_backtrace exn (Stdlib.Printexc.get_raw_backtrace ())
    in

    let reader_domain = Domain.spawn reader_domain_logic in

    let result =
      try arg ()
      with exn ->
        (* Ensure cleanup even if arg() fails *)
        Stdlib.flush Stdlib.stdout;
        (* Flush to pipe_write_fd *)
        Unix.close pipe_write_fd;
        (* Signal EOF to reader domain *)
        (* Restore stdout before waiting for the reader domain so that the write end of the pipe is
           effectively closed (both the explicit [pipe_write_fd] descriptor above _and_ the
           descriptor 1 obtained via [dup2] earlier). Otherwise the reader domain would never see an
           EOF and [Domain.join] would block indefinitely. *)
        Unix.dup2 original_stdout_fd Unix.stdout;
        (* Restore stdout *)
        Unix.close original_stdout_fd;

        (* Now that all write descriptors for the pipe are closed, we can wait for the reader domain
           to finish. *)
        (try Domain.join reader_domain
         with e ->
           Stdio.eprintf "Exception while joining reader domain (arg failed): %s\\n%!"
             (Exn.to_string e));

        (if not (Atomic.get reader_domain_failed) then
           let captured_output = List.rev !collected_logs_ref in
           List.iter (List.rev !captured_log_processors)
             ~f:(fun { log_processor_prefix; process_logs } ->
               process_logs
               @@ List.filter_map captured_output
                    ~f:(String.chop_prefix ~prefix:log_processor_prefix)));
        captured_log_processors := [];
        (* Clear processors *)
        Stdlib.Printexc.raise_with_backtrace exn (Stdlib.Printexc.get_raw_backtrace ())
    in

    (* Normal path: arg() completed successfully *)
    Stdlib.flush Stdlib.stdout;
    (* Flush to pipe_write_fd *)
    Unix.close pipe_write_fd;

    (* Signal EOF to reader domain *)

    (* Restore stdout before waiting for the reader domain so that the write end of the pipe is
       effectively closed and the reader can finish properly. *)
    Unix.dup2 original_stdout_fd Unix.stdout;
    (* Restore stdout *)
    Unix.close original_stdout_fd;

    (try Domain.join reader_domain
     with e ->
       Stdio.eprintf "Exception while joining reader domain (arg succeeded): %s\\n%!"
         (Exn.to_string e);
       if Atomic.get reader_domain_failed then
         Stdlib.Printexc.raise_with_backtrace e (Stdlib.Printexc.get_raw_backtrace ()));

    if not (Atomic.get reader_domain_failed) then
      let captured_output = List.rev !collected_logs_ref in
      Exn.protect
        ~f:(fun () ->
          (* Process captured logs by processors first. *)
          List.iter (List.rev !captured_log_processors)
            ~f:(fun { log_processor_prefix; process_logs } ->
              process_logs
              @@ List.filter_map captured_output
                   ~f:(String.chop_prefix ~prefix:log_processor_prefix)))
        ~finally:(fun () -> captured_log_processors := [])
    else captured_log_processors := [] (* Clear processors if reader failed *);
    result)

let log_debug_routine_logs ~log_contents ~stream_name =
  if get_global_flag ~default:false ~arg_name:"debug_log_to_stream_files" then
    let stream_file_name = diagn_log_file @@ stream_name ^ ".log" in
    Stdio.Out_channel.with_file stream_file_name ~append:true ~f:(fun oc ->
        List.iter log_contents ~f:(fun line -> Stdio.Out_channel.output_line oc line))
  else log_trace_tree log_contents

let log_debug_routine_file ~log_file_name ~stream_name =
  let log_contents = Stdio.In_channel.read_lines log_file_name in
  log_debug_routine_logs ~log_contents ~stream_name;
  Stdlib.Sys.remove log_file_name

type 'a weak_dynarray = 'a Stdlib.Weak.t ref

let weak_create () : 'a weak_dynarray = ref @@ Stdlib.Weak.create 0

let sexp_of_weak_dynarray sexp_of_elem arr =
  sexp_of_array (sexp_of_option sexp_of_elem) Stdlib.Weak.(Array.init (length !arr) ~f:(get !arr))

let register_new (arr : 'a weak_dynarray) ?(grow_by = 1) create =
  let module W = Stdlib.Weak in
  let old = !arr in
  let pos = ref 0 in
  while !pos < W.length old && W.check old !pos do
    Int.incr pos
  done;
  if !pos >= W.length old then (
    arr := Stdlib.Weak.create (W.length old + grow_by);
    Stdlib.Weak.blit old 0 !arr 0 (Stdlib.Weak.length old));
  let v = create !pos in
  W.set !arr !pos (Some v);
  v

let weak_iter (arr : 'a weak_dynarray) ~f =
  let module W = Stdlib.Weak in
  for i = 0 to W.length !arr - 1 do
    Option.iter (W.get !arr i) ~f
  done

type 'a safe_lazy = {
  mutable value : [ `Callback of unit -> 'a | `Value of 'a ];
  unique_id : string;
}
[@@deriving sexp_of]

let safe_lazy unique_id f = { value = `Callback f; unique_id }

let%track9_sexp safe_force gated =
  match gated.value with
  | `Value v -> v
  | `Callback f ->
      let v = f () in
      gated.value <- `Value v;
      v

let is_safe_val = function { value = `Value _; _ } -> true | _ -> false

let safe_map ~upd ~f gated =
  let unique_id = gated.unique_id ^ "_" ^ upd in
  match gated.value with
  | `Value v -> { value = `Value (f v); unique_id }
  | `Callback callback -> { value = `Callback (fun () -> f (callback ())); unique_id }

let equal_safe_lazy equal_elem g1 g2 =
  match (g1.value, g2.value) with
  | `Value v1, `Value v2 ->
      (* Both values are forced - assert uniqueness *)
      let id_equal = String.equal g1.unique_id g2.unique_id in
      if id_equal then assert (equal_elem v1 v2);
      id_equal
  | _ -> String.equal g1.unique_id g2.unique_id

let compare_safe_lazy compare_elem g1 g2 =
  match (g1.value, g2.value) with
  | `Value v1, `Value v2 ->
      (* Both values are forced - assert uniqueness *)
      let id_cmp = String.compare g1.unique_id g2.unique_id in
      if id_cmp = 0 then assert (compare_elem v1 v2 = 0);
      id_cmp
  | _ -> String.compare g1.unique_id g2.unique_id

let hash_fold_safe_lazy _hash_elem state gated = hash_fold_string state gated.unique_id

let sexp_of_safe_lazy sexp_of_elem gated =
  let status =
    match gated.value with `Callback _ -> Sexp.Atom "pending" | `Value v -> sexp_of_elem v
  in
  Sexp.List
    [
      Sexp.Atom "safe_lazy";
      Sexp.List [ Sexp.Atom "id"; Sexp.Atom gated.unique_id ];
      Sexp.List [ Sexp.Atom "value"; status ];
    ]

let gcd a b =
  let rec loop a b = if b = 0 then a else loop b (a % b) in
  loop (abs a) (abs b)
