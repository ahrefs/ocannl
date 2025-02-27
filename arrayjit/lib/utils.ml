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
  }

let accessed_global_args = Hash_set.create (module String)

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
      let result = String.(lowercase @@ drop_prefix arg (length p)) in
      if with_debug then Stdio.printf "Found %s, commandline %s\n%!" result arg;
      Some result
  | None -> (
      match
        List.find_map env_variants ~f:(fun env_n ->
            Option.(
              join
              @@ map (Stdlib.Sys.getenv_opt env_n) ~f:(fun v ->
                     if String.is_empty v then None else Some (env_n, v))))
      with
      | None | Some (_, "") -> None
      | Some (p, arg) ->
          let result = String.lowercase arg in
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
  match read_cmdline_or_env_var "no_config_file" with
  | None | Some "false" -> (
      let read = Stdio.In_channel.read_lines in
      let fname, config_lines =
        let rev_dirs = List.rev @@ filename_parts @@ Stdlib.Sys.getcwd () in
        let rec find_up = function
          | [] -> failwith "OCANNL could not find the ocannl_config file along current path"
          | _ :: tl as rev_dirs -> (
              let fname = filename_of_parts (List.rev @@ ("ocannl_config" :: rev_dirs)) in
              try (fname, read fname) with Sys_error _ -> find_up tl)
        in
        find_up rev_dirs
      in
      Stdio.printf "\nWelcome to OCANNL! Reading configuration defaults from %s.\n%!" fname;
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
               if String.is_empty v then None else Some (key, v)
           | l ->
               failwith @@ "OCANNL: invalid syntax in the config file " ^ fname
               ^ ", should have a single '=' on each non-empty line, found: " ^ String.concat l)
      |> Hashtbl.of_alist (module String)
      |> function
      | `Ok h -> h
      | `Duplicate_key key ->
          failwith @@ "OCANNL: duplicate key in config file " ^ fname ^ ": " ^ key)
  | Some _ ->
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

let build_file fname =
  let build_files_dir = "build_files" in
  (try assert (Stdlib.Sys.is_directory build_files_dir)
   with Stdlib.Sys_error _ -> Stdlib.Sys.mkdir build_files_dir 0o777);
  filename_concat build_files_dir fname

let diagn_log_file fname =
  let log_files_dir = "log_files" in
  (try assert (Stdlib.Sys.is_directory log_files_dir)
   with Stdlib.Sys_error _ -> (
     (* FIXME: is this called concurrently or what? *)
     try Stdlib.Sys.mkdir log_files_dir 0o777 with Stdlib.Sys_error _ -> ()));
  filename_concat log_files_dir fname

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
  let flushing, backend =
    match
      String.lowercase @@ String.strip @@ get_global_arg ~default:"html" ~arg_name:"debug_backend"
    with
    | "text" -> (false, `Text)
    | "html" -> (false, `Html Minidebug_runtime.default_html_config)
    | "markdown" -> (false, `Markdown Minidebug_runtime.default_md_config)
    | "flushing" -> (true, `Text)
    | s ->
        invalid_arg
        @@ "ocannl_debug_backend setting should be text, html, markdown or flushing; found: " ^ s
  in
  let hyperlink = get_global_arg ~default:"./" ~arg_name:"hyperlink_prefix" in
  let print_entry_ids =
    Bool.of_string @@ get_global_arg ~default:"false" ~arg_name:"logs_print_entry_ids"
  in
  let verbose_entry_ids =
    Bool.of_string @@ get_global_arg ~default:"false" ~arg_name:"logs_verbose_entry_ids"
  in
  let log_main_domain_to_stdout =
    Bool.of_string @@ get_global_arg ~default:"false" ~arg_name:"log_main_domain_to_stdout"
  in
  let file_stem =
    if log_main_domain_to_stdout && String.is_empty name then None
    else Some ((if String.is_empty name then "debug" else "debug-") ^ name)
  in
  let filename = Option.map file_stem ~f:(fun stem -> diagn_log_file @@ stem) in
  let prev_run_file =
    let prefix = get_global_arg ~default:"" ~arg_name:"prev_run_prefix" in
    Option.map file_stem ~f:(fun stem -> diagn_log_file @@ prefix ^ stem ^ ".raw")
  in
  let log_level =
    let s = String.strip @@ get_global_arg ~default:"1" ~arg_name:"log_level" in
    match Int.of_string_opt s with
    | Some ll -> ll
    | None -> invalid_arg @@ "ocannl_log_level setting should be an integer; found: " ^ s
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
    if String.is_empty arg then [] else [ Re.Pcre.re arg ]
  in
  let highlight_terms = Re.(alt (highlight_re @ List.map debug_highlights ~f:str)) in
  let diff_ignore_pattern =
    let arg = get_global_arg ~default:"" ~arg_name:"diff_ignore_pattern_pcre" in
    if String.is_empty arg then None else Some (Re.Pcre.re arg)
  in
  if flushing then
    Minidebug_runtime.debug_flushing ?filename ~time_tagged ~elapsed_times ~print_entry_ids
      ~verbose_entry_ids ~global_prefix:name ~for_append:false ~log_level ()
  else
    match filename with
    | None ->
        Minidebug_runtime.forget_printbox
        @@ Minidebug_runtime.debug ~time_tagged ~elapsed_times ~location_format ~print_entry_ids
             ~verbose_entry_ids ~global_prefix:name ~toc_entry ~toc_specific_hyperlink:""
             ~highlight_terms
             ~exclude_on_path:Re.(str "env")
             ~log_level ?snapshot_every_sec ()
    | Some filename ->
        Minidebug_runtime.forget_printbox
        @@ Minidebug_runtime.debug_file ~time_tagged ~elapsed_times ~location_format
             ~print_entry_ids ~verbose_entry_ids ~global_prefix:name ~toc_flame_graph:true
             ~flame_graph_separation:50 ~toc_entry ~for_append:false ~max_inline_sexp_length:120
             ~hyperlink ~toc_specific_hyperlink:"" ~highlight_terms
             ~exclude_on_path:Re.(str "env")
             ~backend ~log_level ?snapshot_every_sec ?prev_run_file ?diff_ignore_pattern filename

let _get_local_debug_runtime =
  let open Stdlib.Domain in
  let get_runtime () =
    (* IMPORTANT: Domain.self() returns unique_id that is never reused, spinning up a new stream
       will create a new log file. *)
    get_debug @@ if is_main_domain () then "" else "Domain-" ^ Int.to_string (self () :> int)
  in
  let debug_runtime_key = DLS.new_key get_runtime in
  fun () ->
    let module Debug_runtime = (val DLS.get debug_runtime_key) in
    if not (is_main_domain ()) then Debug_runtime.log_level := settings.log_level;
    (module Debug_runtime : Minidebug_runtime.Debug_runtime)

module Debug_runtime = (val _get_local_debug_runtime ())

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

(* [%%global_debug_interrupts { max_nesting_depth = 100; max_num_children = 1000 }] *)

let%diagn_sexp set_log_level level =
  settings.log_level <- level;
  Debug_runtime.log_level := level;
  [%log "Set log_level to", (level : int)]

let restore_settings () =
  set_log_level (Int.of_string @@ get_global_arg ~arg_name:"log_level" ~default:"0");
  settings.debug_log_from_routines <-
    Bool.of_string @@ get_global_arg ~arg_name:"debug_log_from_routines" ~default:"false";
  settings.output_debug_files_in_build_directory <-
    Bool.of_string
    @@ get_global_arg ~arg_name:"output_debug_files_in_build_directory" ~default:"false";
  settings.fixed_state_for_init <-
    (let seed = get_global_arg ~arg_name:"fixed_state_for_init" ~default:"" in
     if String.is_empty seed then None else Some (Int.of_string seed));
  settings.print_decimals_precision <-
    Int.of_string @@ get_global_arg ~arg_name:"print_decimals_precision" ~default:"2";
  settings.check_half_prec_constants_cutoff <-
    Float.of_string_opt
    @@ get_global_arg ~arg_name:"check_half_prec_constants_cutoff" ~default:"16384.0";
  settings.automatic_host_transfers <-
    Bool.of_string @@ get_global_arg ~arg_name:"automatic_host_transfers" ~default:"true"

let () = restore_settings ()
let with_runtime_debug () = settings.output_debug_files_in_build_directory && settings.log_level > 1
let debug_log_from_routines () = settings.debug_log_from_routines && settings.log_level > 1

let never_capture_stdout () =
  Bool.of_string @@ get_global_arg ~arg_name:"never_capture_stdout" ~default:"false"

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

let get_debug_formatter ~fname =
  if settings.output_debug_files_in_build_directory then
    let f = Stdio.Out_channel.create @@ build_file fname in
    let ppf = Stdlib.Format.formatter_of_out_channel f in
    Some ppf
  else None

exception User_error of string

let header_sep =
  let open Re in
  compile (seq [ str " "; opt any; str "="; str " " ])

let%diagn_l_sexp log_trace_tree _logs =
  [%log_block
    "trace tree";
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
             String.concat ~sep:"\n" @@ String.split ~on:'$'
             @@ String.chop_prefix_exn ~prefix:"# " source
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

type pp_file = { f_name : string; ppf : Stdlib.Format.formatter; finalize : unit -> unit }

let pp_file ~base_name ~extension =
  let column_width = 110 in
  let f_name =
    if settings.output_debug_files_in_build_directory then build_file @@ base_name ^ extension
    else Stdlib.Filename.temp_file (base_name ^ "_") extension
  in
  (* (try Stdlib.Sys.remove f_name with _ -> ()); *)
  let oc = Out_channel.open_text f_name in
  (* FIXME(#32): is the truncated source problem (missing the last line) solved? *)
  let ppf = Stdlib.Format.formatter_of_out_channel oc in
  Stdlib.Format.pp_set_geometry ppf ~max_indent:(column_width / 2) ~margin:column_width;
  let finalize () =
    Stdlib.Format.pp_print_newline ppf ();
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc
  in
  { f_name; ppf; finalize }

let captured_log_prefix = ref "!@#"

(** To avoid the complication of a concurrent thread, we expose a callback for collaborative log
    processing. *)
let advance_captured_logs = ref None

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
    let ls = ref [] in
    let lastl = ref "" in
    let backup = ref (Unix.dup Unix.stdout) in
    let exit_entrance = ref (Unix.pipe ()) in
    let pre_advance () =
      Unix.dup2 (snd !exit_entrance) Unix.stdout;
      Unix.set_nonblock (snd !exit_entrance)
    in
    let advance is_last () =
      Stdlib.flush Stdlib.stdout;
      Unix.close (snd !exit_entrance);
      Unix.dup2 !backup Unix.stdout;
      let channel = Unix.in_channel_of_descr (fst !exit_entrance) in
      (try
         while true do
           let is_endlined, line = input_line channel in
           let line = !lastl ^ line in
           if is_endlined then (
             (match String.chop_prefix ~prefix:!captured_log_prefix line with
             | None -> Stdlib.print_endline line
             (* ls := line :: !ls *)
             | Some logline -> ls := logline :: !ls);
             lastl := "")
           else lastl := line
         done
       with End_of_file -> ());
      if not is_last then (
        backup := Unix.dup Unix.stdout;
        exit_entrance := Unix.pipe ();
        pre_advance ())
    in
    advance_captured_logs := Some (advance false);
    pre_advance ();
    let result =
      try arg ()
      with Sys_blocked_io ->
        advance_captured_logs := None;
        invalid_arg
          "capture_stdout_logs: unfortunately, flushing stdout inside captured code is prohibited"
    in
    advance true ();
    let output = List.rev !ls in
    Exn.protect
      ~f:(fun () ->
        (* Preserve the order in which kernels were launched. *)
        List.iter (List.rev !captured_log_processors)
          ~f:(fun { log_processor_prefix; process_logs } ->
            process_logs
            @@ List.filter_map output ~f:(String.chop_prefix ~prefix:log_processor_prefix)))
      ~finally:(fun () ->
        advance_captured_logs := None;
        captured_log_processors := []);
    result)

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
