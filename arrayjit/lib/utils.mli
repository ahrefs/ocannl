(** Shared utilities of arrayjit: the global settings and configuration reading (command line,
    environment, config file, profiles) with its key registry and cache-identity classification,
    debug-artifact and log paths with their startup cleanup, routine-log capture, C-stderr and
    device-init safeguards, and small list, sexp and float-rendering helpers. *)

open Base

val count_backend_compile : unit -> unit
(** Counts one completed backend compile (code generation or deferred device compile) on this
    domain. *)

val backend_compiles : unit -> int
(** The number of backend compiles completed on this domain so far; see {!count_backend_compile}. *)

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
  mutable default_prng_variant : string;
      (** The default variant of threefry4x32 PRNG to use. Options: "crypto" (20 rounds) or "light"
          (2 rounds). Defaults to "light" for better performance. *)
  mutable large_models : bool;
      (** If true, use uint64 for indexing arithmetic. If false, use uint32 for indexing arithmetic.
          This affects all backends' kernel index parameters and local index variables, and gates
          the per-pool offset width (uint32 caps a pool at 4 GB; see the pool allocator). Decoupled
          in intent from element indexing within a single tensor, though both currently follow it.
      *)
}
[@@deriving sexp_of]
(** Process-wide runtime settings, initialized from the configuration. *)

val settings : settings
(** The global settings record, set from the configuration at startup; see {!restore_settings}. *)

val accessed_global_args : String.t Hash_set.t
(** The configuration keys this process has read through the configuration getters so far. *)

val known_config_keys : (String.t, String.comparator_witness) Set.t
(** Every recognized configuration key; a source naming any other key is warned about. *)

(** What a configuration key does to the identity a compilation cache keys on. *)
type config_key_class =
  | Aggregate
      (** Selects values for other keys rather than acting itself; those keys carry the
          classification. *)
  | Code_borne
      (** Changes the lowered code or its placements, hence the canonical digest
          ([Ir.Schedule_cache.digest], the ["digest"] key component) that every cache key starts
          with. Nothing to add: the code {e is} the identity. *)
  | Keyed of string
      (** Not fully captured by the lowered code, so it has to be carried explicitly — by this named
          component of [Ir.Schedule_cache.key_components]. This is the class gh-ocannl-568 was an
          omission from. *)
  | Search_shaping
      (** Shapes which schedule a search proposes, times or crowns, but not what a crowned one means
          or how fast it then runs: a saved schedule carries its own ops and replay re-derives
          nothing from these. Two processes differing only here may find different winners; each is
          a valid winner for the other. *)
  | Placement_shaping
      (** Steers [Train.tune_placements] above any single search: which placement programs it
          searches (each a different program, keyed on its own digest) and which of them ships. No
          single [Autotune.tune] call reads it, so it is neither in the cache key nor in the search
          trajectory's identity ({!config_class_fingerprint} of [Search_shaping]): a search's timed
          record is the same record whatever the level above it was told (gh-ocannl-1192). *)
  | Untimed_model
      (** Steers the untuned regime's untimed pick: [Autotune.model_default], the pick by the
          analytic cost model -- whether a recipe's untuned compile routes through it, and how
          widely it searches. It times nothing and neither replays nor stores a schedule-cache
          entry, and no [Autotune.tune] call reads these keys, so they are neither in the cache key
          nor in the search trajectory's identity ({!config_class_fingerprint} of [Search_shaping]):
          a search's timed record is the same record whatever the untuned regime beside it was told
          (gh-ocannl-1241). [Autotune.seeded_default]'s keys started here and are [Search_shaping]
          since the search seeds its plan first (gh-ocannl-1388). *)
  | Execution_neutral
      (** Host-side behavior only: logging, debug artifacts, directories, validation and error
          reporting, allocation layout, launch mechanics. Nothing a kernel does depends on it. *)

val config_key_classification : (config_key_class * string * string list) list
(** The class of every known key, as [(class, reason, keys)] groups. *)

val classify_config_key : string -> (config_key_class * string) option
(** The class and reason recorded for a key, or [None] when it is unclassified. *)

val bool_of_config_string : arg_name:String.t -> String.t -> bool
(** Parses [true]/[1] or [false]/[0], any case, stripped; raises [Invalid_argument] naming
    [arg_name]. *)

(** How a key's value parses: the one parse shared by the typed getters and the class fingerprint.
*)
type config_value_kind =
  | Flag  (** {!bool_of_config_string}: [true]/[1] and [false]/[0], any case, stripped *)
  | Integer  (** [Int.of_string] of the stripped text *)
  | Float_number  (** [Float.of_string] of the stripped text, NaN excluded *)
  | Keyword
      (** a mode name, stripped and lowercased, such as [auto]; which names are modes is the read
          site's [~parse] ({!registered_keyword}), so the kind itself refuses nothing *)
  | Text  (** stripped text, such as a path *)
  | Positive_float_or_unset
      (** an override that only a positive float sets: empty and non-positive numbers mean "not
          overridden", as [Autotune.peak_override] reads it *)
  | Float_at_least of { floor : float }
      (** a float clamped up to [floor], NaN excluded: every spelling the read site cannot tell
          apart from another canonicalizes alike, so the read ([Sketch_families.pad_waste_max]) and
          the fingerprint share one owner of the clamp *)

val canonical_config_value : config_value_kind -> String.t -> String.t Option.t
(** The value's canonical spelling under the kind, or [None] when it does not parse as one. *)

val config_key_defaults : (string * (config_value_kind * string)) list
(** The built-in default and value kind of each key whose resolved value an identity renders. *)

val registered_config_default : string -> (config_value_kind * string) option
(** The key's {!config_key_defaults} row, if it has one. *)

val log_config_sourcing : bool ref
(** Whether to trace each setting's source to stderr; unused in-tree, a switch a program may flip.
*)

val log_config_json : bool ref
(** Whether to print each resolved setting to stderr as JSON; unused in-tree, a switch to flip. *)

val env_var_name : String.t -> String.t
(** The environment spelling of a config key: [OCANNL_] followed by the key in uppercase. *)

val env_names_case_insensitive : bool
(** Whether the runtime's [getenv] matches names case-insensitively (native Windows, not Cygwin). *)

(** What an environment variable name is to OCANNL's configuration. *)
type env_var_class =
  | Env_not_addressed  (** not [ocannl]-prefixed: someone else's variable entirely *)
  | Env_reserved of string  (** in a reserved non-configuration namespace, named by its prefix *)
  | Env_config_key of string  (** a spelling {!read_env_var} reads, of that key *)
  | Env_unread_spelling of string
      (** a known key, spelled in a way nothing reads: dashed, or not fully uppercase where case
          matters. A configuration ERROR rather than a warning -- see {!unread_env_vars}. *)
  | Env_unread_reserved of string
      (** in a reserved namespace, in a casing its reader does not consult *)
  | Env_unknown_key of string  (** addressed to the configuration, naming no key *)

val classify_env_var : ?case_insensitive:bool -> String.t -> env_var_class
(** Classifies a variable name; [case_insensitive] defaults to {!env_names_case_insensitive}. *)

val env_var_in_ambient_record : ?case_insensitive:bool -> String.t -> bool
(** Whether the test suite's ambient record carries the variable: config-addressed, not the backend.
*)

val unread_env_vars : unit -> (String.t * bool * String.t) List.t
(** Non-empty config-addressed variables nothing reads, as sorted [(name, is_fatal, reason)]. *)

val cmdline_var_names : ?qualified_only:bool -> String.t -> String.t List.t
(** A key's dash-led command-line spellings: prefixed, then prefix-free unless [qualified_only]. *)

val cmdline_var_prefixes : ?qualified_only:bool -> String.t -> String.t List.t
(** Argument prefixes: each {!cmdline_var_names} spelling followed by [_], [-], [=] or nothing. *)

type config_token_shape =
  | Command_line_token
  | Environment_assignment_token
  | Documentation_assignment_token
      (** The grammatical shape of a configuration token, independent of the key registry. *)

type config_token = { token_shape : config_token_shape; token_key : string }
(** A configuration token's shape and its normalized key. *)

val config_token_command_line_prefixes : String.t List.t
(** The prefixes that can begin a qualified command-line configuration token. *)

type config_token_rejection =
  | Not_configuration_syntax
  | Empty_qualified_key
  | Invalid_command_line_key
  | Invalid_environment_key
  | Invalid_documentation_key
      (** Why a token has no configuration reading; syntax only, never registry membership. *)

type config_token_parse =
  | Accepted of config_token
  | Ambiguous of {
      explicit : config_token option;
      candidates : config_token list;
      fallback : config_token;
    }
  | Rejected of { reason : config_token_rejection; qualified : config_token option }
      (** A lexed token: one reading, longest-first candidate readings, or a rejection. *)

val parse_config_token : ?documentation:bool -> String.t -> config_token_parse
(** Lexes a token without consulting {!known_config_keys}; [documentation] admits bare assignments.
*)

val qualified_only_config_keys : (String.t, String.comparator_witness) Set.t
(** Keys whose prefix-free command-line spellings are never claimed, such as [profile]. *)

val cmdline_arg_value : ?qualified_only:bool -> String.t -> String.t -> String.t option
(** [cmdline_arg_value n arg] is [arg]'s value for key [n], unless a longer registered key owns
    [arg]. *)

val read_cmdline_var : ?qualified_only:bool -> String.t -> (String.t * String.t) option
(** The first command-line value for a key, with the [Sys.argv] element it came from. *)

val cmdline_arg_is_config_key : String.t -> bool
(** Whether a raw command-line argument addresses some known key under any spelling read. *)

val read_env_var : String.t -> (String.t * String.t) option
(** The environment value for a key, with the variable's name; an empty value counts as unset. *)

type config_level =
  | Cmdline_level
  | Env_level
  | Config_file_level  (** The source levels a setting can come from, in decreasing priority. *)

val describe_config_level : config_level -> string
(** A level's name in prose, such as ["the commandline"]. *)

type config_source =
  | From_cmdline of string  (** the matching [Sys.argv] element *)
  | From_env of string  (** the matching environment variable *)
  | From_config_file
  | From_profile of config_level * string  (** the level that picked the profile, and its name *)
  | From_default  (** Where {!get_global_arg} found a value. *)

val config_source_label : config_source -> string
(** A short provenance tag, such as ["environment"] or
    ["profile 'reproducible' via the commandline"]. *)

val describe_config_source : value:string -> default:string -> config_source -> string
(** The sourcing-trace line for a resolved [value], or for the [default] used when nothing set it.
*)

val resolve_config_value :
  cmdline:('a -> ('b * string) Option.t) ->
  env:('a -> ('b * string) Option.t) ->
  file:('a -> 'b Option.t) ->
  profile:(config_level * string * ('a -> 'b Option.t)) option ->
  default:'b ->
  arg_name:'a ->
  'b * config_source
(** The precedence walk: each level's explicit value, then the profile picked there, else [default].
*)

val parse_config_lines : source:String.t -> String.t List.t -> (String.t * String.t) List.t
(** Parses [ocannl_config]-syntax lines into [(key, value)] pairs; [source] names them in errors. *)

val config_file_args : (String.t, String.t) Hashtbl.t
(** The settings of the [ocannl_config] file found up from the working directory, if any. *)

val config_file_path : string
(** The path of the config file read, or [""] when none was. *)

val profile_payloads : (string * string) list
(** The embedded profile payloads by name, each a partial [ocannl_config] text. *)

val parse_profile_payload : name:string -> String.t -> (String.t, String.t) Hashtbl.t
(** Parses a profile payload into a table; fails on an unknown or profile-ineligible key. *)

val resolve_profile_selection :
  cmdline:(string -> (String.t * string) Option.t) ->
  env:(string -> (String.t * string) Option.t) ->
  file:(string -> String.t Option.t) ->
  String.t * config_source
(** The normalized profile name picked by ordinary source precedence, with its source. *)

val active_profile : (config_level * String.t * (String.t, String.t) Hashtbl.t) Option.t
(** The profile picked for this run, if any: its level, name and parsed payload. *)

type config_record = { key : string; value : string; source : config_source }
(** One resolution of one setting: the key, the raw value settled on, and where it came from. *)

val json_string_literal : String.t -> string
(** A JSON string literal: quote, backslash and control characters escaped, other bytes as they are.
*)

val config_record_to_json : config_record -> string
(** The one-line JSON object (schema version 1) the [log_config_json] stream prints for a record. *)

val resolved_config : unit -> config_record List.t
(** Every setting this process has resolved so far, in first-resolution order. *)

val registered_getter_of_kind : config_value_kind -> string
(** The name of the typed getter reading keys of a kind, such as ["registered_int"]. *)

val get_global_arg_with_source : default:String.t -> arg_name:string -> String.t * config_source
(** {!get_global_arg} with the value's source; refuses a key with a {!config_key_defaults} row. *)

val get_global_arg : default:String.t -> arg_name:string -> String.t
(** The value of config key [arg_name] from the command line, environment, config file or profile,
    else [default]. *)

exception Invalid_config_value of { key : string; value : string; expected : string }
(** Raised by a typed getter when a setting's value does not parse as its key's kind. *)

val registered_flag : arg_name:string -> bool
(** A registered [Flag] key's value, defaulting to its {!config_key_defaults} row. *)

val registered_int : arg_name:string -> Int.t
(** A registered [Integer] key's value; raises {!Invalid_config_value} on unparsable text. *)

val registered_float : arg_name:string -> Float.t
(** A registered [Float_number] key's value; raises {!Invalid_config_value} on unparsable text. *)

val registered_text : arg_name:string -> String.t
(** A registered [Text] key's stripped value. *)

val registered_positive_float_or_unset : arg_name:string -> Float.t option
(** A registered [Positive_float_or_unset] key's value; [None] when empty or not positive. *)

val registered_float_at_least : arg_name:string -> Float.t
(** A registered [Float_at_least] key's value, clamped up to its floor. *)

val registered_keyword : arg_name:string -> expected:string -> parse:(String.t -> 'a option) -> 'a
(** A registered [Keyword] key's mode by [parse] of the lowercased value; [expected] lists the
    modes. *)

val profile_key_source : String.t -> (String.t * config_source) option
(** A key's resolved value and source, [None] when nothing sets it; neither logged nor recorded. *)

val config_class_fingerprint :
  ?resolve:(key:String.t -> String.t -> string) ->
  ?reads:(string -> bool) ->
  config_key_class ->
  String.t
(** [key=value;] for each read key of the class resolved off its default: the class's identity. *)

val profile_payload_sources : String.t -> (String.t * (String.t * config_source) option) List.t
(** Each key of the named profile's payload with its {!profile_key_source} resolution. *)

val inherited_env_key_sources :
  unit -> (String.t * string * (String.t * config_source) option) List.t
(** Each environment variable spelling a key, with the key and where it actually resolved from. *)

val get_global_flag : default:bool -> arg_name:String.t -> bool
(** A boolean setting: {!get_global_arg} parsed by {!bool_of_config_string}. *)

external os_process_id : unit -> int = "ocannl_os_process_id"
(** The operating system's id of this process (unlike [Unix.getpid] on Windows). *)

(** The advisory lock that keeps a sibling's startup cleanup off this process's artifacts. *)
module Artifacts_lock : sig
  val path : string -> string
end

val build_files_dir : unit -> String.t
(** The debug-artifact directory [build_files/<prefix>/], created if missing. *)

val build_file : String.t -> String.t
(** The path of an artifact in {!build_files_dir}; shields this process's artifacts from cleanups.
*)

val dynamic_library_extension : string
(** The file extension of a dynamically loaded library on this platform. *)

external lock_holder_pid : Unix.file_descr -> int = "ocannl_lock_holder_pid"
(** The OS id of a process blocking an exclusive lock on the file: [0] if none, [-1] if unknown. *)

external process_alive : int -> bool = "ocannl_process_alive"
(** Whether a process with this OS id is running; one that cannot be queried counts as running. *)

val process_nonce : unit -> string
(** A random token drawn once per process, telling apart processes that report the same OS id. *)

val library_kernel_name_limit : int
(** The most bytes of a kernel name that an owner-recording artifact's file name carries. *)

val library_kernel_name : String.t -> String.t
(** The cleaned kernel name, cut at a UTF-8 boundary with a digest suffix if over the limit. *)

val kept_library_name : pid:int -> nonce:string -> run_id:int -> String.t -> string
(** The owner- and run-recording file name of a library kept in the build directory. *)

type kept_library = {
  kernel_name : string;
  owner_pid : int;
  owner_nonce : string;
  library_run_id : int;
}
(** The owner and the compile a library's file name records. *)

val temp_library_marker : string
(** The prefix every OCANNL temporary library's file name opens with. *)

val temp_library_prefix : pid:int -> nonce:string -> run_id:int -> String.t -> string
(** The owner- and run-recording [Filename.temp_file] prefix of a temporary library. *)

val temp_library_sweep_min_age : float
(** Seconds since its last write before a startup removes a gone owner's temporary library. *)

val parse_kept_library_name : String.t -> kept_library option
(** Inverts {!kept_library_name}; [None] for any other file name. *)

val parse_temp_library_name : String.t -> kept_library option
(** The owner and compile of a {!temp_library_prefix}-named library; [None] for any other name. *)

val owned_artifact_file : run_id:int -> extension:string -> String.t -> String.t
(** Where this process keeps an artifact of a kernel's compile [run_id], under an owner-recording
    name. *)

val kept_library_file : run_id:int -> String.t -> String.t
(** Where this process keeps the library of a kernel's compile [run_id]. *)

val log_files_dir : unit -> String.t
(** The log directory [log_files/<prefix>/], created if missing, prefixed as {!build_files_dir}. *)

val diagn_log_file : String.t -> String.t
(** The path of a log in {!log_files_dir}; shields this process's logs from sibling cleanups. *)

val get_local_debug_runtime : unit -> (module Minidebug_runtime.Debug_runtime)
(** The ppx_minidebug runtime, configured from the debug-logging settings. *)

val set_log_level : int -> unit
(** Sets [settings.log_level] and the debug runtime's log level. *)

val restore_settings : unit -> unit
(** Resets {!settings} to the values the configuration gives. *)

val with_runtime_debug : unit -> bool
(** Whether compilation keeps debug information: debug files on and [log_level > 1]. *)

val debug_log_from_routines : unit -> bool
(** Whether kernels log their execution: [debug_log_from_routines] on and [log_level > 1]. *)

external detach_c_stderr : unit -> bool = "ocannl_detach_c_stderr"
(** Points C [stderr] at its own dup of the real stderr; the stub behind {!c_stderr_detached},
    unused in-tree, so a caller that took over fd 2 can redo the detachment. *)

val c_stderr_detached : bool
(** Whether C [stderr] was detached at startup, so foreign C chatter skips a later fd 2 redirect. *)

external c_stderr_replaceable : unit -> bool = "ocannl_c_stderr_replaceable"
(** Whether this C runtime's [stderr] can be replaced at all (glibc, Darwin). *)

val host_is_darwin : bool
(** Whether the host is macOS (Darwin), as this library's build target says (gh-ocannl-1741): the
    one predicate behind the cc backend's kernel link flags and the tests gating on Darwin, read
    without shelling out to [uname]. *)

val host_is_darwin_arm64 : bool
(** Whether the host is arm64 macOS, which refuses to map unsigned code (gh-ocannl-1741). *)

external collapse_c_stderr_noise : unit -> bool = "ocannl_collapse_c_stderr_noise"
(** Collapses ROCm's per-event C [stderr] lines into a count; the stub behind
    {!c_stderr_noise_collapsed}, unused in-tree. *)

val c_stderr_noise_collapsed : bool lazy_t
(** Whether ROCm's per-event C [stderr] chatter is collapsed, installing the filter when forced. *)

(** What {!bounded_exit_teardown} did with one resource. *)
type exit_teardown_outcome =
  | Torn_down
  | Disabled  (** [exit_stream_teardown_timeout] is [0] or negative. *)
  | Still_busy  (** Not idle within the bound: left to the driver, never waited on further. *)
  | Failed of exn  (** [is_idle] or [teardown] raised; the exception is swallowed. *)

val default_exit_stream_teardown_timeout : float
(** The [exit_stream_teardown_timeout] used when the setting is absent or not a finite number. *)

type exit_teardown_resource = { what : string; is_idle : unit -> bool; teardown : unit -> unit }
(** One resource for {!bounded_exit_teardown}; [what] names it in diagnostics. *)

val bounded_exit_teardown : exit_teardown_resource list -> exit_teardown_outcome list
(** Tears down each resource once idle, all under one exit timeout; never raises. *)

external device_init_deadline_supported : unit -> bool = "ocannl_device_init_deadline_supported"
(** Whether this build can arm the device-init deadline at all (not on Windows). *)

val device_init_deadline_tag : string
(** The fixed start of the diagnostic the device-init deadline writes. *)

val device_init_test_stall : (unit -> unit) option ref
(** Test-only: run inside {!with_device_init_deadline}'s guarded region, before the guarded call. *)

val driver_init_test_stall : (unit -> unit) option ref
(** Test-only: run right before a GPU backend's one-time driver initialization call. *)

val with_device_init_deadline : backend_name:string -> (unit -> 'a) -> 'a
(** Runs a device init or query under [device_init_deadline] seconds, dying by SIGALRM past it. *)

val get_global_run_id : unit -> Int.t
(** A fresh run id per call, unique across the process's backends. *)

val union_find :
  equal:('a -> 'a -> bool) -> ('a, 'a, 'b) Map.t -> key:'a -> rank:Int.t -> 'a * Int.t
(** The root of [key] in a parent-pointer map, with [rank] plus the number of steps taken. *)

val union_add : equal:('a -> 'a -> bool) -> ('a, 'a, 'b) Map.t -> 'a -> 'a -> ('a, 'a, 'b) Map.t
(** Merges the classes of two elements in a parent-pointer map. *)

val unique_keep_first : equal:('a -> 'a -> bool) -> 'a list -> 'a List.t
(** Filters the list keeping the first occurrence of each element. *)

val sorted_diff : compare:('a -> 'b -> int) -> 'a List.t -> 'b list -> 'a List.t
(** The multiset difference of two lists sorted in increasing order. *)

val remove_elem : equal:('a -> 'b -> bool) -> 'a -> 'b list -> 'b List.t
(** Removes the first element of the list equal to the given one. *)

type atomic_bool = bool Atomic.t
(** An atomic boolean. *)

val sexp_of_atomic_bool : bool Atomic.t -> Sexp.t
(** The current value as a sexp. *)

type atomic_int = int Atomic.t
(** An atomic integer. *)

val sexp_of_atomic_int : int Atomic.t -> Sexp.t
(** The current value as a sexp. *)

val sexp_append : elem:Sexp.t -> Sexp.t -> Sexp.t
(** Prepends [elem] to a sexp list, or pairs it with an atom. *)

val sexp_mem : elem:Sexp.t -> Sexp.t -> bool
(** Whether [elem] equals the atom, or is a member of the list. *)

val normalize_exponent : String.t -> String.t
(** Strips leading zeros from a float rendering's exponent, so it does not depend on the C runtime.
*)

val decimal_float_literal : Float.t -> String.t
(** A decimal floating literal that parses back to exactly the double; non-finite ones as words. *)

(** [Base.Lazy] plus ppx_minidebug's sexp converters. *)
module Lazy : sig
  include module type of struct
    include Base.Lazy
  end

  val sexp_of_t : ('a -> Sexp.t) -> 'a t -> Sexp.t
  val sexp_of_lazy_t : ('a -> Sexp.t) -> 'a t -> Sexp.t
end

(** Whether something is skipped, required, or optional with a callback for when it is missing. *)
type requirement =
  | Skip
  | Required
  | Optional of { callback_if_missing : unit -> unit [@sexp.opaque] }
[@@deriving sexp_of]

val output_to_build_file : fname:String.t -> (PPrint.ToChannel.document -> unit) option
(** When debug files are on, truncates the build file [fname] and returns a printer appending to it.
*)

exception User_error of string
(** An error the user's program or configuration caused, rather than a bug. *)

type build_file_channel = { f_path : string; oc : Stdlib.out_channel; finalize : unit -> unit }
(** An open file for generated code: its path, its channel, and the flush-and-close. *)

val open_build_file : base_name:String.t -> extension:String.t -> build_file_channel
(** Opens [base_name ^ extension] in the build directory when debug files are on, else a temp file.
*)

val captured_log_prefix : string ref
(** The prefix marking a routine log line on stdout for {!capture_stdout_logs} to collect. *)

type captured_log_processor = { log_processor_prefix : string; process_logs : string list -> unit }
(** A handler of the captured log lines carrying its prefix. *)

val add_log_processor : prefix:string -> (string list -> unit) -> unit
(** Registers a handler of the captured lines carrying [prefix] (prefix removed). *)

external flush_c_streams : unit -> unit = "ocannl_flush_c_streams"
(** Flushes the C runtime's stdio streams. *)

val capture_stdout_logs : (unit -> 'a) -> 'a
(** Runs [f], handing stdout's log lines to the registered processors, then clearing them. *)

val routine_log_window : drain:(unit -> unit) -> (unit -> unit) -> unit
(** Runs a logged GPU [launch] (then [drain]) in a capture window when stream-file logs need one. *)

val log_debug_routine_logs : log_contents:string List.t -> stream_name:String.t -> unit
(** Appends routine log lines to the stream's log file, or to the debug trace when that is off. *)

val log_debug_routine_file : log_file_name:string -> stream_name:String.t -> unit
(** {!log_debug_routine_logs} of a file's lines, then removes the file. *)

val gcd : Int.t -> Int.t -> Int.t
(** The greatest common divisor of the absolute values. *)

val macos_platform_uuid : String.t -> string option
(** The digest of a well-formed, nonzero [IOPlatformUUID] found in [ioreg] output. *)

val macos_timing_environment : (string * String.t option) Option.t lazy_t
(** On macOS, the host's hashed platform UUID and, if available, the OS build version. *)

val included_headers : String.t -> String.t List.t
(** The headers a C or C++ text includes by [#include <...>], in order. *)

val capability_dir : requires:string List.t -> String.t option -> String.t
(** A found directory as a capability record: resolved path, plus missing headers or their digest.
*)

(** Re-exports the shared datatypes: [Set_O], [safe_lazy], [Tree_map], map-ref helpers, ... *)
include module type of struct
  include Datatypes
end

module Cpu_topology = Cpu_topology
(** CPU topology facts for the worker-pool uniformity policy. *)

module Atomic_file = Atomic_file
(** Publishing a file so that no reader ever observes a half-written one. *)

module Host_load = Host_load
(** Host CPU accounting, telling a co-tenant apart from a routine's own variance. *)
