(* Configuration names used outside OCaml stay tied to [Utils.known_config_keys].

   The source forms are deliberately narrow and syntactic. Shell and Python scripts contribute every
   qualified command-line token and environment assignment classified by [Utils.parse_config_token],
   and every identifier-delimited [OCANNL_<KEY>] token, including tokens in comments and quoted
   command strings -- those are still instructions or commands a reader can reuse. Markdown
   contributes an inline code span whose whole rendered content is one assignment in a prefixed form
   or the parser's narrowed lowercase snake-case bare form; counted site judgments retain ambiguous
   one-word config assignments. Fenced code and longer snippets describe arbitrary APIs and
   expression languages, so treating every equals sign in them as configuration would make the scan
   unusable.

   gh-ocannl-790. *)

open Base
open Stdio

(* The live corpus comes from [Test_utils.Source_inventory]: a clean Dune sandbox keeps source paths
   off the command line and lets this scanner select by file kind rather than by root. *)
let argv = Stdlib.Sys.argv

module Markdown = Test_utils.Agent_notes_scan
module Refusal_manifest = Test_utils.Refusal_control_manifest
module Inventory = Test_utils.Source_inventory

let printf = Refusal_manifest.printf

type kind =
  | Cli_flag
  | Prefix_free_cli_flag
  | Environment_assignment
  | Standalone_environment_mention
  | Markdown_assignment
  | Config_file_assignment

type occurrence = {
  path : string;
  line : int;
  key : string;
  spelling : string;
  kind : kind;
  ambiguous_bare : bool;
}

let lowercase_key_char c = Char.is_lowercase c || Char.is_digit c || Char.equal c '_'
let uppercase_key_char c = Char.is_uppercase c || Char.is_digit c || Char.equal c '_'

(* [OCANNL_TOOL_*] is the repository's explicit tool-private namespace, and
   [OCANNL_LOG_LEVEL_<MODULE>] is ppx_minidebug's compile-time tracing-gate namespace. Neither is
   OCANNL runtime configuration, and both are open vocabularies by design. *)
let non_config_env_key key =
  String.is_prefix key ~prefix:"TOOL_" || String.is_prefix key ~prefix:"LOG_LEVEL_"

(* Discovery and classification both come from the registry-independent grammar in [Utils]. This
   still finds a REMOVED key: the parser recognizes shape, never registry membership. *)
let cli_name_prefixes = Utils.config_token_command_line_prefixes
let cli_key_char c = Char.is_alpha c || Char.is_digit c || Char.equal c '_' || Char.equal c '-'

let cli_token_char c =
  cli_key_char c || List.mem [ '='; '.'; '/'; ':'; '+'; '$'; '{'; '}' ] c ~equal:Char.equal

(* A qualified spelling can be ambiguous when a runtime value separator is also a legal key
   character: [backend_cuda=true] can mean key [backend] with value [cuda=true], or a key named
   [backend_cuda]. The no-equals separators have the same problem. Prefer the explicit syntactic
   name so a removed longer key cannot be absorbed by a surviving prefix. Exceptional runtime-value
   readings and deliberately invalid mixed-key examples are counted site judgments: the judgment
   records which registry key the site discusses, not that the runtime accepts its spelling. For
   example, the runtime-valid [--ocannl_print_decimals_precision-7] mixes separators only across the
   key/value boundary, so it needs that judgment before the parser can know where the key ends.
   Store suffixes separately so these declarations do not scan themselves as more command-line
   occurrences. *)
let ambiguous_cli_value_mentions =
  [
    ("arrayjit/lib/utils.ml", "--ocannl_", "log_level_1", "log_level", 1);
    ("docs/agent-notes/build-and-test.md", "--ocannl_", "backend_cuda=true", "backend", 1);
    ("test/operations/config_var_spellings.ml", "--ocannl_", "log_level_0", "log_level", 1);
    ("test/operations/dune", "--ocannl_", "log_level_0", "log_level", 3);
    ( "test/operations/dune",
      "--ocannl-",
      "print-decimals-precision-7",
      "print_decimals_precision",
      1 );
    ( "test/operations/config_usage_scan.ml",
      "--ocannl_",
      "print_decimals_precision-7",
      "print_decimals_precision",
      1 );
    ( "docs/agent-notes/build-and-test.md",
      "--ocannl_",
      "print_decimals_precision-7",
      "print_decimals_precision",
      1 );
    ( "docs/agent-notes/conventions.md",
      "--ocannl-",
      "print_decimals-precision=1",
      "print_decimals_precision",
      1 );
    ( "test/operations/dune",
      "--ocannl-",
      "print_decimals-precision=7",
      "print_decimals_precision",
      1 );
  ]

let declared_ambiguous_cli_value_key ~path token =
  List.find_map ambiguous_cli_value_mentions ~f:(fun (tracked_path, prefix, suffix, key, _) ->
      Option.some_if (String.equal path tracked_path && String.equal token (prefix ^ suffix)) key)

let registry_ambiguous_cli_value_key token =
  Set.to_list Utils.known_config_keys
  |> List.concat_map ~f:(fun key ->
      Utils.cmdline_var_prefixes ~qualified_only:true key
      |> List.filter_map ~f:(fun prefix ->
          Option.some_if
            (String.is_prefix token ~prefix && String.length token > String.length prefix)
            (key, String.length prefix)))
  |> List.max_elt ~compare:(fun (_, left) (_, right) -> Int.compare left right)
  |> Option.map ~f:fst

let registry_independent_unknown_cli_key token =
  let name = Option.value_map (String.lsplit2 token ~on:'=') ~default:token ~f:fst in
  List.find_map cli_name_prefixes ~f:(fun prefix ->
      Option.bind (String.chop_prefix name ~prefix) ~f:(fun raw_key ->
          Option.some_if
            (not (String.is_empty raw_key))
            (String.lowercase raw_key |> String.tr ~target:'-' ~replacement:'_')))

let cli_key_of_token ~path token =
  match Utils.parse_config_token token with
  | Some { token_shape = Utils.Command_line_token; token_key } ->
      Some (Option.value (declared_ambiguous_cli_value_key ~path token) ~default:token_key)
  | Some
      { token_shape = Utils.Environment_assignment_token | Utils.Documentation_assignment_token; _ }
    ->
      None
  | None ->
      Option.first_some
        (declared_ambiguous_cli_value_key ~path token)
        (Option.first_some
           (registry_ambiguous_cli_value_key token)
           (registry_independent_unknown_cli_key token))

(* Prefix-free flags belong to the host application's namespace, so they cannot be discovered
   globally without claiming flags such as [--profile=prod]. Counted site judgments identify the
   places that deliberately document OCANNL's supported prefix-free form. The spelling grammar still
   comes from the runtime helper, and the key remains literal here so removing it from the registry
   makes the old occurrence fail. *)
let prefix_free_config_mentions =
  [
    ("docs/agent-notes/conventions.md", "backend", 1);
    ("docs/proposals/gh-ocannl-409.md", "backend", 3);
    ("tools/calibrate_bandwidth.ml", "backend", 1);
    ("tools/fit_envelope.ml", "backend", 6);
    ("bin/bench_args.ml", "backend", 1);
    ("arrayjit/lib/utils.ml", "virtualize_max_visits", 1);
  ]

let prefix_free_names key =
  if Set.mem Utils.qualified_only_config_keys key then []
  else
    let qualified = Utils.cmdline_var_names ~qualified_only:true key in
    Utils.cmdline_var_names key
    |> List.filter ~f:(fun name -> not (List.mem qualified name ~equal:String.equal))

let prefix_free_occurrences_for ~path ~keys content =
  String.split_lines content
  |> List.mapi ~f:(fun index line ->
      List.concat_map keys ~f:(fun key ->
          List.concat_map (prefix_free_names key) ~f:(fun name ->
              let rec from pos found =
                match String.substr_index line ~pos ~pattern:name with
                | None -> List.rev found
                | Some start
                  when start > 0
                       && (Char.is_alphanum line.[start - 1]
                          || Char.equal line.[start - 1] '_'
                          || Char.equal line.[start - 1] '-') ->
                    from (start + 1) found
                | Some start ->
                    let stop = ref (start + String.length name) in
                    while !stop < String.length line && cli_token_char line.[!stop] do
                      Int.incr stop
                    done;
                    let spelling = String.sub line ~pos:start ~len:(!stop - start) in
                    from
                      (max (start + 1) !stop)
                      ({
                         path;
                         line = index + 1;
                         key;
                         spelling;
                         kind = Prefix_free_cli_flag;
                         ambiguous_bare = false;
                       }
                      :: found)
              in
              from 0 [])))
  |> List.concat

let tracked_prefix_free_keys path =
  List.filter_map prefix_free_config_mentions ~f:(fun (tracked_path, key, _) ->
      Option.some_if (String.equal path tracked_path) key)
  |> List.dedup_and_sort ~compare:String.compare

let prefixed_occurrences ?(start_ok = fun _ _ -> true) ~path ~prefix ~key_char ~kind content =
  String.split_lines content
  |> List.mapi ~f:(fun index line ->
      let rec from pos found =
        match String.substr_index line ~pos ~pattern:prefix with
        | None -> List.rev found
        | Some start when not (start_ok line start) -> from (start + 1) found
        | Some start ->
            let key_start = start + String.length prefix in
            let key_stop = ref key_start in
            while !key_stop < String.length line && key_char line.[!key_stop] do
              Int.incr key_stop
            done;
            let next = max (start + 1) !key_stop in
            if
              !key_stop > key_start
              && !key_stop < String.length line
              && Char.equal line.[!key_stop] '='
            then
              let spelling = String.sub line ~pos:start ~len:(!key_stop - start + 1) in
              match Utils.parse_config_token spelling with
              | None -> from next found
              | Some parsed ->
                  from next
                    ({
                       path;
                       line = index + 1;
                       key = parsed.token_key;
                       spelling;
                       kind;
                       ambiguous_bare = false;
                     }
                    :: found)
            else from next found
      in
      from 0 [])
  |> List.concat

(* These sites deliberately retain or exercise spellings known to be invalid. Counts stop a second
   obsolete instruction or test input in the same file from borrowing the intended exception. This
   list lives beside the spelling classifiers so a repetition changes its count rather than
   disappearing as ordinary non-config prose. *)
let historical_invalid_config_mentions =
  [
    ("docs/agent-notes/conventions.md", "cc_parallel_grid_private_bytes_cap", 1);
    ("docs/agent-notes/conventions.md", "private_bytes_cap", 1);
    ("docs/proposals/gh-ocannl-409.md", "bacend", 1);
    ("docs/proposals/gh-ocannl-409.md", "output_debug_files_in_run_directory", 1);
    ("docs/proposals/gh-ocannl-409.md", "randomness_lib", 4);
    ("test/operations/dune", "backedn", 3);
    ("test/operations/dune", "not_a_real_key", 1);
    ("test/operations/startup_streams/ocannl_config", "definitely_not_a_config_key", 1);
  ]

let tracked_historical_config path key =
  List.exists historical_invalid_config_mentions ~f:(fun (tracked_path, tracked_key, _) ->
      String.equal path tracked_path && String.equal key tracked_key)

let script_occurrences ~path content =
  let cli_occurrences =
    String.split_lines content
    |> List.mapi ~f:(fun index line ->
        List.concat_map cli_name_prefixes ~f:(fun name_prefix ->
            let rec from pos found =
              match String.substr_index line ~pos ~pattern:name_prefix with
              | None -> List.rev found
              | Some start
                when start > 0
                     && (Char.is_alphanum line.[start - 1]
                        || Char.equal line.[start - 1] '_'
                        || Char.equal line.[start - 1] '-') ->
                  from (start + 1) found
              | Some start -> (
                  let stop = ref (start + String.length name_prefix) in
                  while !stop < String.length line && cli_token_char line.[!stop] do
                    Int.incr stop
                  done;
                  let next = max (start + 1) !stop in
                  if !stop = start + String.length name_prefix then from next found
                  else
                    let spelling = String.sub line ~pos:start ~len:(!stop - start) in
                    match cli_key_of_token ~path spelling with
                    | None -> from next found
                    | Some key ->
                        from next
                          ({
                             path;
                             line = index + 1;
                             key;
                             spelling;
                             kind = Cli_flag;
                             ambiguous_bare = false;
                           }
                          :: found))
            in
            from 0 []))
    |> List.concat
  in
  let environment_assignments =
    prefixed_occurrences
      ~start_ok:(fun line start ->
        start = 0 || not (Char.is_alphanum line.[start - 1] || Char.equal line.[start - 1] '_'))
      ~path ~prefix:"OCANNL_" ~key_char:uppercase_key_char ~kind:Environment_assignment content
    |> List.filter ~f:(fun occurrence -> not (non_config_env_key (String.uppercase occurrence.key)))
  in
  let standalone_environment_mentions =
    String.split_lines content
    |> List.mapi ~f:(fun index line ->
        let rec from pos found =
          match String.substr_index line ~pos ~pattern:"OCANNL_" with
          | None -> List.rev found
          | Some start
            when start > 0 && (Char.is_alphanum line.[start - 1] || Char.equal line.[start - 1] '_')
            ->
              from (start + 1) found
          | Some start ->
              let key_start = start + String.length "OCANNL_" in
              let key_stop = ref key_start in
              while !key_stop < String.length line && uppercase_key_char line.[!key_stop] do
                Int.incr key_stop
              done;
              let next = max (start + 1) !key_stop in
              if
                !key_stop = key_start
                || !key_stop < String.length line
                   && (Char.is_alphanum line.[!key_stop]
                      || Char.equal line.[!key_stop] '_'
                      || Char.equal line.[!key_stop] '=')
              then from next found
              else
                let raw_key = String.sub line ~pos:key_start ~len:(!key_stop - key_start) in
                if non_config_env_key raw_key then from next found
                else
                  from next
                    ({
                       path;
                       line = index + 1;
                       key = String.lowercase raw_key;
                       spelling = String.sub line ~pos:start ~len:(!key_stop - start);
                       kind = Standalone_environment_mention;
                       ambiguous_bare = false;
                     }
                    :: found)
        in
        from 0 [])
    |> List.concat
  in
  cli_occurrences @ environment_assignments @ standalone_environment_mentions

let same_range (a, b) (x, y) = Int.equal a x && Int.equal b y

let complete_inline_code_candidates content =
  let scan = Markdown.inert_by_line content in
  let comments line = Markdown.spans_at scan.comment_ranges line in
  let fences line = Markdown.spans_at scan.fence_ranges line in
  Markdown.lines content
  |> List.concat_map ~f:(fun (lineno, line) ->
      Markdown.spans_at scan.ranges lineno
      |> List.filter ~f:(fun range ->
          (not (List.mem (comments lineno) range ~equal:same_range))
          && not (List.mem (fences lineno) range ~equal:same_range))
      |> List.filter_map ~f:(fun (start, stop) ->
          Option.some_if
            (stop > start && Char.equal line.[start] '`' && Char.equal line.[stop - 1] '`')
            (String.sub line ~pos:start ~len:(stop - start) |> Markdown.code_span_content)))

(* Whitespace and one-word names make bare assignments indistinguishable from non-config prose. Pin
   every current config use of either form by file/key/count: a later key removal still scans the
   old site, while a newly added ambiguous use must declare itself here. Prefixed CLI and
   environment forms are unambiguous and do not need this judgment list. *)
let ambiguous_bare_config_mentions =
  [
    ("AGENTS.md", "profile", 1);
    ("docs/agent-notes/backend-dialects-and-idents.md", "backend", 1);
    ("docs/agent-notes/build-and-test.md", "backend", 3);
    ("docs/agent-notes/build-and-test.md", "profile", 1);
    ("docs/proposals/gh-ocannl-409.md", "backend", 1);
    ("ocannl_config.reference", "profile", 1);
    ( "docs/proposals/fix-inline-complex-computations-default-doc.md",
      "inline_complex_computations",
      1 );
    ("docs/proposals/gh-ocannl-344.md", "large_models", 2);
    ("docs/proposals/gh-ocannl-351.md", "inline_complex_computations", 2);
    ("docs/proposals/gh-ocannl-530-pool-uniformity.md", "cc_pool_core_class", 1);
    ("docs/proposals/task-73617488.md", "virtualize_max_visits", 3);
  ]

let tracked_ambiguous_bare_config path key =
  List.exists ambiguous_bare_config_mentions ~f:(fun (tracked_path, tracked_key, _) ->
      String.equal path tracked_path && String.equal key tracked_key)

let control_fixture path = String.equal path "config_usage_scan_bogus.fixture"

let one_assignment ~path rendered =
  match String.lsplit2 rendered ~on:'=' with
  | Some (raw_name, raw_value) ->
      let name = String.strip raw_name in
      let value = String.strip raw_value in
      let spaced = not (String.equal raw_name name && String.equal raw_value value) in
      let value_has_whitespace = String.exists value ~f:Char.is_whitespace in
      if
        String.is_empty name
        || String.is_prefix name ~prefix:"OCANNL_"
        || Option.is_some (cli_key_of_token ~path (name ^ "=" ^ value))
      then None
      else
        let parsed_key =
          match Utils.parse_config_token ~documentation:true (name ^ "=" ^ value) with
          | Some parsed -> Some parsed.token_key
          | None ->
              let registered_one_word =
                (not (String.contains name '_'))
                && (not (String.is_empty name))
                && Char.is_lowercase name.[0]
                && String.for_all name ~f:(fun c -> Char.is_lowercase c || Char.is_digit c)
                && Set.mem Utils.known_config_keys name
              in
              Option.some_if
                (registered_one_word
                || tracked_ambiguous_bare_config path name
                || tracked_historical_config path name)
                name
        in
        Option.bind parsed_key ~f:(fun key ->
            if
              value_has_whitespace
              && not
                   (Set.mem Utils.known_config_keys key
                   || tracked_ambiguous_bare_config path key
                   || control_fixture path
                   || tracked_historical_config path key)
            then None
            else if
              (not spaced)
              || Set.mem Utils.known_config_keys key
              || tracked_ambiguous_bare_config path key
              || control_fixture path
              || tracked_historical_config path key
            then
              Some
                ( key,
                  (spaced || not (String.contains name '_'))
                  && (not (control_fixture path))
                  && not (tracked_historical_config path key) )
            else None)
  | None -> None

let markdown_occurrences ~allow_bare ~path content =
  let scan = Markdown.inert_by_line content in
  let comments line = Markdown.spans_at scan.comment_ranges line in
  let fences line = Markdown.spans_at scan.fence_ranges line in
  let as_markdown lineno occurrence =
    { occurrence with line = lineno; kind = Markdown_assignment; ambiguous_bare = false }
  in
  let lines = Markdown.lines content in
  let inline =
    lines
    |> List.concat_map ~f:(fun (lineno, line) ->
        Markdown.spans_at scan.ranges lineno
        |> List.filter ~f:(fun range ->
            (not (List.mem (comments lineno) range ~equal:same_range))
            && not (List.mem (fences lineno) range ~equal:same_range))
        |> List.concat_map ~f:(fun (start, stop) ->
            if stop <= start then []
            else
              let spelling = String.sub line ~pos:start ~len:(stop - start) in
              let complete = Char.equal line.[start] '`' && Char.equal line.[stop - 1] '`' in
              (* A multiline code span arrives as one range per physical line. Bare assignments
                 still require a complete span, but prefixed tokens are unambiguous on any segment.
                 Remove whichever delimiter this segment owns before applying the token reader. *)
              let rendered =
                if complete then Markdown.code_span_content spelling
                else
                  (spelling |> fun s -> Option.value (String.chop_prefix s ~prefix:"`") ~default:s)
                  |> fun s -> Option.value (String.chop_suffix s ~suffix:"`") ~default:s
              in
              let prefixed =
                script_occurrences ~path rendered |> List.map ~f:(as_markdown lineno)
              in
              let bare =
                if complete && allow_bare then
                  Option.to_list
                  @@ Option.map (one_assignment ~path rendered) ~f:(fun (key, ambiguous_bare) ->
                      {
                        path;
                        line = lineno;
                        key;
                        spelling = rendered;
                        kind = Markdown_assignment;
                        ambiguous_bare;
                      })
                else []
              in
              prefixed @ bare))
  in
  let fenced =
    lines
    |> List.concat_map ~f:(fun (lineno, line) ->
        if List.is_empty (fences lineno) then []
        else script_occurrences ~path line |> List.map ~f:(as_markdown lineno))
  in
  inline @ fenced

(* Keep this normalization beside [Utils.parse_config_lines]: config files accept case-insensitive
   keys, leading dashes/spaces, and an optional [ocannl_] prefix. The registry population remains
   independent so a removed normalized key still fails here. *)
let normalize_config_file_key raw_key =
  let key =
    String.lowercase @@ String.strip raw_key ~drop:(fun c -> Char.equal c '-' || Char.equal c ' ')
  in
  if String.is_prefix key ~prefix:"ocannl" then
    String.drop_prefix key 6 |> String.strip ~drop:(fun c -> Char.equal c '_')
  else key

let config_file_occurrences ?(include_commented = false) ~path content =
  String.split_lines content
  |> List.filter_mapi ~f:(fun index line ->
      (* [Utils.parse_config_lines] tests comment markers against the RAW line. Leading whitespace
         therefore makes [#key=value] an active (invalid) key rather than a comment, and this scan
         must not silently forgive it by stripping first. *)
      let commented = String.is_prefix line ~prefix:"#" in
      let candidate =
        if include_commented then
          Option.value (String.chop_prefix line ~prefix:"#") ~default:line |> String.strip
        else line
      in
      if
        String.is_empty (String.strip candidate)
        || ((not include_commented) && commented)
        || String.is_prefix line ~prefix:"~~"
      then None
      else
        Option.bind (String.lsplit2 candidate ~on:'=') ~f:(fun (raw_key, value) ->
            let key = normalize_config_file_key raw_key in
            Option.some_if
              ((not (String.is_empty value))
              && ((not commented) || String.for_all key ~f:lowercase_key_char))
              {
                path;
                line = index + 1;
                key;
                spelling = String.strip raw_key ^ "=";
                kind = Config_file_assignment;
                ambiguous_bare = false;
              }))

(* Standalone environment names are unambiguous enough to scan throughout command-like text, but
   OCANNL-prefixed C feature macros and synthetic parser-test variables deliberately share the
   namespace. Pin those judgments by file/key/count so the exception cannot absorb a second site. *)
let non_config_environment_mentions =
  [
    ("benchmarks/report-gh550-cuda.md", "550_no_release", 1);
    ("docs/agent-notes/backend-precision-and-simd.md", "half_fma", 3);
    ("docs/agent-notes/backend-precision-and-simd.md", "vec_widen_", 1);
    ("docs/agent-notes/backend-precision-and-simd.md", "vec_narrow_", 1);
    ("docs/agent-notes/backend-precision-and-simd.md", "vec_widen_half", 1);
    ("docs/agent-notes/backend-precision-and-simd.md", "has_elementwise_fma", 3);
    ("docs/proposals/gh-ocannl-163.md", "has_avx2", 1);
    ("docs/proposals/gh-ocannl-163.md", "has_neon", 1);
    ("docs/proposals/gh-ocannl-164.md", "has_avx2", 6);
    ("docs/proposals/gh-ocannl-164.md", "has_neon", 6);
    ("docs/proposals/gh-ocannl-575-narrow-register-tiling.md", "half_fma", 4);
    ("docs/research/ggml-lessons.md", "has_elementwise_fma", 1);
    ("docs/research/ggml-lessons.md", "has_avx2", 2);
    ("docs/research/ggml-lessons.md", "has_neon", 1);
    ("ocannl_config.reference", "print", 2);
    ("test/operations/dune", "dashed_only_key", 4);
    ("test/operations/dune", "demo_key", 6);
    ("test/operations/cc_march_census.ml", "vec_widen_half", 2);
    ("test/operations/cc_march_census.ml", "vec_widen_bfloat16", 1);
    ("test/operations/cc_march_census.ml", "vec_", 1);
    ("test/operations/cc_march_census.ml", "vec_widen_", 1);
    ("test/operations/cc_march_census.ml", "vec_narrow_", 1);
    ("test/operations/codegen_text_scan_cases.ml", "vec_widen_bfloat16", 2);
    ("test/operations/codegen_text_scan_cases.ml", "half_fma", 2);
    ("test/operations/config_var_spellings.ml", "demo_key", 1);
    ("test/operations/config_var_spellings.ml", "print", 1);
    ("test/operations/config_var_spellings.ml", "backedn", 1);
    ("test/operations/env_var_deps.ml", "backedn", 1);
    ("test/operations/env_var_deps.ml", "demo_key", 1);
    ("test/operations/env_var_deps.ml", "dashed_only_key", 1);
    ("test/operations/env_var_deps.ml", "x", 1);
    ("test/operations/env_var_deps.ml", "not_a_config_key", 1);
    ("test/operations/narrow_storage_compute.ml", "half_fma", 2);
    ("test/operations/narrow_storage_compute.ml", "vec_widen_bfloat16", 1);
    ("test/operations/narrow_storage_compute.ml", "vec_narrow_bfloat16", 1);
    ("test/operations/narrow_storage_compute.ml", "vec_widen_half", 2);
    ("test/operations/scope_over_materialized.ml", "virtualize_", 1);
    ("test/operations/tile_mma_narrow.ml", "vec_widen_bfloat16", 1);
    ("test/operations/tile_mma_narrow.ml", "vec_narrow_bfloat16", 1);
    ("test/operations/tile_mma_narrow.ml", "vec_widen_half", 1);
    ("test/operations/tile_mma_narrow.ml", "vec_narrow_half", 1);
    ("test/operations/tile_mma_narrow.ml", "half_fma", 1);
    ("arrayjit/lib/builtins_cc.ml", "has_avx2", 2);
    ("arrayjit/lib/builtins_cc.ml", "has_neon", 2);
    ("arrayjit/lib/builtins_cc.ml", "has_elementwise_fma", 3);
    ("arrayjit/lib/builtins_cc.ml", "has_convertvector", 11);
    ("arrayjit/lib/builtins_cc.ml", "vec_widen_bfloat16", 3);
    ("arrayjit/lib/builtins_cc.ml", "vec_narrow_bfloat16", 3);
    ("arrayjit/lib/builtins_cc.ml", "half_fma", 4);
    ("arrayjit/lib/builtins_cc.ml", "vec_widen_half", 3);
    ("arrayjit/lib/builtins_cc.ml", "vec_narrow_half", 3);
    ("arrayjit/lib/c_syntax.ml", "vec_widen_bfloat16", 2);
    ("arrayjit/lib/c_syntax.ml", "vec_narrow_bfloat16", 1);
    ("arrayjit/lib/c_syntax.ml", "vec_widen_half", 1);
    ("arrayjit/lib/c_syntax.ml", "vec_narrow_half", 1);
    ("arrayjit/lib/c_syntax.ml", "half_fma", 3);
    ("arrayjit/lib/c_syntax.ml", "has_elementwise_fma", 1);
    ("arrayjit/lib/cc_backend.ml", "half_fma", 1);
    ("arrayjit/lib/context.mli", "vec_widen_bfloat16", 1);
    ("arrayjit/lib/utils.ml", "not_a_key", 2);
    ("arrayjit/lib/utils.ml", "backedn", 2);
    ("arrayjit/lib/utils.ml", "print", 1);
  ]

(* These lowercase snake-case assignments are still ambiguous with other languages or report formats
   after the token grammar has excluded camelCase, uppercase environment-style names, and one-word
   mathematical notation. This is a judgment list rather than a restatement of a vocabulary owned
   elsewhere, and it is scoped by FILE as well as key. Each entry pins its occurrence count:
   becoming real, disappearing, or gaining a second same-file use all fail. *)
let non_config_assignment_mentions =
  [
    ("docs/agent-notes/scheduling-and-autotune.md", "max_chain", 1);
    ("docs/precision_inference.md", "top_down_prec", 6);
    ("docs/proposals/concat-forward-component-data-propagation.md", "a_mc", 1);
    ("docs/proposals/concat-forward-component-data-propagation.md", "b_mc", 1);
    ("docs/proposals/concat-forward-component-data-propagation.md", "c_mc", 1);
    ("docs/proposals/gh-ocannl-263.md", "seq_q", 1);
    ("docs/proposals/gh-ocannl-536.md", "private_seg_size", 1);
    ("docs/research/lean-attention-feasibility.md", "seq_q", 1);
    ("docs/syntax_extensions.md", "kernel_size", 2);
  ]

let mention_site path key = path ^ "\000" ^ key
let ambiguous_cli_value_site path spelling key = path ^ "\000" ^ spelling ^ "\000" ^ key

let kind_name = function
  | Cli_flag -> "command-line flag"
  | Prefix_free_cli_flag -> "prefix-free command-line flag"
  | Environment_assignment -> "environment assignment"
  | Standalone_environment_mention -> "environment-variable mention"
  | Markdown_assignment -> "documentation assignment"
  | Config_file_assignment -> "configuration-file assignment"

let has_ambiguous_cli_value occurrence =
  let qualified_cli =
    Poly.equal occurrence.kind Cli_flag
    || List.exists cli_name_prefixes ~f:(fun prefix -> String.is_prefix occurrence.spelling ~prefix)
  in
  if qualified_cli then
    match Utils.parse_config_token occurrence.spelling with
    | Some { token_shape = Utils.Command_line_token; token_key } ->
        not (String.equal token_key occurrence.key)
    | Some _ -> false
    | None -> true
  else false

let check ?(fail = Verdict.fail) ?(known_keys = Utils.known_config_keys)
    ?(control_non_config_environment_mentions = non_config_environment_mentions)
    ?(control_non_config_assignment_mentions = non_config_assignment_mentions)
    ?(control_historical_invalid_config_mentions = historical_invalid_config_mentions)
    ?(control_ambiguous_bare_config_mentions = ambiguous_bare_config_mentions)
    ?(control_prefix_free_config_mentions = prefix_free_config_mentions)
    ?(control_ambiguous_cli_value_mentions = ambiguous_cli_value_mentions) ~repository_census
    occurrences =
  let sites mentions =
    Set.of_list (module String)
    @@ List.map mentions ~f:(fun (path, key, _) -> mention_site path key)
  in
  let non_config_environment_sites = sites control_non_config_environment_mentions in
  let non_config_assignment_sites = sites control_non_config_assignment_mentions in
  let historical_invalid_config_sites = sites control_historical_invalid_config_mentions in
  let ambiguous_bare_config_sites = sites control_ambiguous_bare_config_mentions in
  let prefix_free_config_sites = sites control_prefix_free_config_mentions in
  let ambiguous_cli_value_sites =
    Set.of_list (module String)
    @@ List.map control_ambiguous_cli_value_mentions ~f:(fun (path, prefix, suffix, key, _) ->
        ambiguous_cli_value_site path (prefix ^ suffix) key)
  in
  let seen_non_config = Hashtbl.create (module String) in
  let seen_non_config_environment = Hashtbl.create (module String) in
  let seen_historical = Hashtbl.create (module String) in
  let seen_ambiguous_bare_config = Hashtbl.create (module String) in
  let seen_prefix_free = Hashtbl.create (module String) in
  let seen_ambiguous_cli_value = Hashtbl.create (module String) in
  List.iter occurrences ~f:(fun occurrence ->
      let ambiguous_site =
        ambiguous_cli_value_site occurrence.path occurrence.spelling occurrence.key
      in
      if has_ambiguous_cli_value occurrence then
        if Set.mem ambiguous_cli_value_sites ambiguous_site then
          Hashtbl.incr seen_ambiguous_cli_value ambiguous_site
        else
          fail
            (Printf.sprintf
               "%s:%d: ambiguous command-line value `%s` for `%s` lacks a file/token/key/count \
                entry in ambiguous_cli_value_mentions"
               occurrence.path occurrence.line occurrence.spelling occurrence.key);
      (if occurrence.ambiguous_bare then
         let site = mention_site occurrence.path occurrence.key in
         if Set.mem ambiguous_bare_config_sites site then
           Hashtbl.incr seen_ambiguous_bare_config site
         else
           fail
             (Printf.sprintf
                "%s:%d: ambiguous bare config mention `%s` lacks a file/key/count entry in \
                 ambiguous_bare_config_mentions"
                occurrence.path occurrence.line occurrence.spelling));
      (match occurrence.kind with
      | Prefix_free_cli_flag ->
          let site = mention_site occurrence.path occurrence.key in
          if Set.mem prefix_free_config_sites site then Hashtbl.incr seen_prefix_free site
          else if repository_census then
            fail
              (Printf.sprintf
                 "%s:%d: prefix-free config flag `%s` lacks a file/key/count entry in \
                  prefix_free_config_mentions"
                 occurrence.path occurrence.line occurrence.spelling)
      | _ -> ());
      if Set.mem known_keys occurrence.key then ()
      else if Set.mem non_config_environment_sites (mention_site occurrence.path occurrence.key)
      then Hashtbl.incr seen_non_config_environment (mention_site occurrence.path occurrence.key)
      else if Set.mem non_config_assignment_sites (mention_site occurrence.path occurrence.key) then
        Hashtbl.incr seen_non_config (mention_site occurrence.path occurrence.key)
      else if Set.mem historical_invalid_config_sites (mention_site occurrence.path occurrence.key)
      then Hashtbl.incr seen_historical (mention_site occurrence.path occurrence.key)
      else
        fail
          (Printf.sprintf "%s:%d: %s `%s` names `%s`, absent from Utils.known_config_keys"
             occurrence.path occurrence.line (kind_name occurrence.kind) occurrence.spelling
             occurrence.key));
  if repository_census then (
    let newly_real_environment =
      List.filter control_non_config_environment_mentions ~f:(fun (_, key, _) ->
          Set.mem known_keys key)
    in
    if not (List.is_empty newly_real_environment) then
      fail
        (Printf.sprintf
           "non-config environment exemptions now name registered config keys -- remove: %s"
           (newly_real_environment
           |> List.map ~f:(fun (path, key, _) -> path ^ ":" ^ key)
           |> String.concat ~sep:", "));
    let drifted_non_config_environment =
      List.filter_map control_non_config_environment_mentions ~f:(fun (path, key, expected) ->
          let actual =
            Hashtbl.find seen_non_config_environment (mention_site path key)
            |> Option.value ~default:0
          in
          Option.some_if (not (Int.equal actual expected)) (path, key, expected, actual))
    in
    if not (List.is_empty drifted_non_config_environment) then
      fail
        (Printf.sprintf "non-config environment-mention occurrence counts drifted: %s"
           (drifted_non_config_environment
           |> List.map ~f:(fun (path, key, expected, actual) ->
               Printf.sprintf "%s:%s expected %d, saw %d" path key expected actual)
           |> String.concat ~sep:", "));
    let newly_real =
      List.filter control_non_config_assignment_mentions ~f:(fun (_, key, _) ->
          Set.mem known_keys key)
    in
    if not (List.is_empty newly_real) then
      fail
        (Printf.sprintf
           "non-config assignment exemptions now name registered config keys -- remove: %s"
           (newly_real
           |> List.map ~f:(fun (path, key, _) -> path ^ ":" ^ key)
           |> String.concat ~sep:", "));
    let drifted_non_config =
      List.filter_map control_non_config_assignment_mentions ~f:(fun (path, key, expected) ->
          let actual =
            Hashtbl.find seen_non_config (mention_site path key) |> Option.value ~default:0
          in
          Option.some_if (not (Int.equal actual expected)) (path, key, expected, actual))
    in
    if not (List.is_empty drifted_non_config) then
      fail
        (Printf.sprintf "non-config assignment exemption occurrence counts drifted: %s"
           (drifted_non_config
           |> List.map ~f:(fun (path, key, expected, actual) ->
               Printf.sprintf "%s:%s expected %d, saw %d" path key expected actual)
           |> String.concat ~sep:", "));
    let drifted_historical =
      List.filter_map control_historical_invalid_config_mentions ~f:(fun (path, key, expected) ->
          let actual =
            Hashtbl.find seen_historical (mention_site path key) |> Option.value ~default:0
          in
          Option.some_if (not (Int.equal actual expected)) (path, key, expected, actual))
    in
    if not (List.is_empty drifted_historical) then
      fail
        (Printf.sprintf "historical invalid-config exemption occurrence counts drifted: %s"
           (drifted_historical
           |> List.map ~f:(fun (path, key, expected, actual) ->
               Printf.sprintf "%s:%s expected %d, saw %d" path key expected actual)
           |> String.concat ~sep:", "));
    let drifted_ambiguous_bare =
      List.filter_map control_ambiguous_bare_config_mentions ~f:(fun (path, key, expected) ->
          let actual =
            Hashtbl.find seen_ambiguous_bare_config (mention_site path key)
            |> Option.value ~default:0
          in
          Option.some_if (not (Int.equal actual expected)) (path, key, expected, actual))
    in
    if not (List.is_empty drifted_ambiguous_bare) then
      fail
        (Printf.sprintf "ambiguous bare config-mention occurrence counts drifted: %s"
           (drifted_ambiguous_bare
           |> List.map ~f:(fun (path, key, expected, actual) ->
               Printf.sprintf "%s:%s expected %d, saw %d" path key expected actual)
           |> String.concat ~sep:", "));
    let drifted_prefix_free =
      List.filter_map control_prefix_free_config_mentions ~f:(fun (path, key, expected) ->
          let actual =
            Hashtbl.find seen_prefix_free (mention_site path key) |> Option.value ~default:0
          in
          Option.some_if (not (Int.equal actual expected)) (path, key, expected, actual))
    in
    if not (List.is_empty drifted_prefix_free) then
      fail
        (Printf.sprintf "prefix-free config-mention occurrence counts drifted: %s"
           (drifted_prefix_free
           |> List.map ~f:(fun (path, key, expected, actual) ->
               Printf.sprintf "%s:%s expected %d, saw %d" path key expected actual)
           |> String.concat ~sep:", "));
    let drifted_ambiguous_cli_value =
      List.filter_map control_ambiguous_cli_value_mentions
        ~f:(fun (path, prefix, suffix, key, expected) ->
          let spelling = prefix ^ suffix in
          let actual =
            Hashtbl.find seen_ambiguous_cli_value (ambiguous_cli_value_site path spelling key)
            |> Option.value ~default:0
          in
          Option.some_if (not (Int.equal actual expected)) (path, spelling, key, expected, actual))
    in
    if not (List.is_empty drifted_ambiguous_cli_value) then
      fail
        (Printf.sprintf "ambiguous command-line value occurrence counts drifted: %s"
           (drifted_ambiguous_cli_value
           |> List.map ~f:(fun (path, spelling, key, expected, actual) ->
               Printf.sprintf "%s:%s (%s) expected %d, saw %d" path spelling key expected actual)
           |> String.concat ~sep:", ")))

let direct_refusal_formats =
  [
    "%s:%d: ambiguous bare config mention `%s` lacks a file/key/count entry in \
     ambiguous_bare_config_mentions";
    "%s:%d: prefix-free config flag `%s` lacks a file/key/count entry in \
     prefix_free_config_mentions";
    "%s:%d: %s `%s` names `%s`, absent from Utils.known_config_keys";
    "%s:%d: ambiguous command-line value `%s` for `%s` lacks a file/token/key/count entry in \
     ambiguous_cli_value_mentions";
    "non-config environment exemptions now name registered config keys -- remove: %s";
    "non-config environment-mention occurrence counts drifted: %s";
    "non-config assignment exemptions now name registered config keys -- remove: %s";
    "non-config assignment exemption occurrence counts drifted: %s";
    "historical invalid-config exemption occurrence counts drifted: %s";
    "ambiguous bare config-mention occurrence counts drifted: %s";
    "prefix-free config-mention occurrence counts drifted: %s";
    "ambiguous command-line value occurrence counts drifted: %s";
  ]

let refusal_control grammar_fixture =
  let source = "test/operations/config_usage_scan.ml" in
  let observed = Hash_set.create (module String) in
  let unexpected = ref [] in
  let fail message =
    match
      List.find direct_refusal_formats ~f:(fun format ->
          Test_utils.Refusal_control_scan.format_matches ~format message)
    with
    | Some format ->
        Hash_set.add observed format;
        Refusal_manifest.observe_failure ~source ~format
    | None -> unexpected := message :: !unexpected
  in
  let known_keys =
    let add_first_key mentions keys =
      match mentions with (_, key, _) :: _ -> Set.add keys key | [] -> keys
    in
    Utils.known_config_keys
    |> add_first_key non_config_environment_mentions
    |> add_first_key non_config_assignment_mentions
  in
  let occurrence ?(ambiguous_bare = false) ~path ~key ~spelling ~kind () =
    { path; line = 1; key; spelling; kind; ambiguous_bare }
  in
  Verdict.p "a runtime value separator is resolved before equals inside its value"
    (Option.equal String.equal
       (cli_key_of_token ~path:"docs/agent-notes/build-and-test.md"
          ("--ocannl_" ^ "backend_cuda=true"))
       (Some "backend"));
  Verdict.p_all ~min:1 "every scanner command-line prefix belongs to the runtime grammar"
    cli_name_prefixes ~f:(fun prefix ->
      let sentinel = "configtokengrammarkey" in
      let sentinel =
        if String.equal prefix (String.uppercase prefix) then String.uppercase sentinel
        else sentinel
      in
      List.mem (Utils.cmdline_var_prefixes sentinel) (prefix ^ sentinel ^ "=") ~equal:String.equal);
  Verdict.p_all ~min:4 "runtime-generated command-line spellings remain config tokens"
    [
      "--ocannl_print_decimals_precision=7";
      "--ocannl-print-decimals-precision=7";
      "--OCANNL_PRINT_DECIMALS_PRECISION=7";
      "--OCANNL-PRINT-DECIMALS-PRECISION=7";
    ] ~f:(fun spelling -> Option.is_some (Utils.parse_config_token spelling));
  Verdict.p_none "runtime-rejected mixed command-line spellings are not config tokens"
    [
      ("--ocannl-", "print_decimals-precision=7");
      ("--ocannl_", "Print_Decimals_Precision=7");
      ("--OCANNL_", "print_decimals_precision=7");
    ]
    ~f:(fun (prefix, suffix) -> Option.is_some (Utils.parse_config_token (prefix ^ suffix)));
  Verdict.p "a counted alternate separator recovers the runtime key/value boundary"
    (Option.equal String.equal
       (cli_key_of_token ~path:"test/operations/config_usage_scan.ml"
          ("--ocannl_" ^ "print_decimals_precision-7"))
       (Some "print_decimals_precision"));
  Verdict.p "an undeclared alternate separator is classified so the scan can refuse it"
    (Option.equal String.equal
       (cli_key_of_token ~path:"undeclared-alternate.md"
          ("--ocannl_" ^ "print_decimals_precision-8"))
       (Some "print_decimals_precision"));
  Verdict.p "an unknown cross-style command-line token remains classifiable for refusal"
    (Option.equal String.equal
       (cli_key_of_token ~path:"unknown-cross-style.sh" ("--ocannl_" ^ "typo_key-7"))
       (Some "typo_key_7"));
  Verdict.p_all ~min:2 "counted one-word documentation assignments remain config tokens"
    [
      ("AGENTS.md", "profile=reproducible|performance|approximate", "profile");
      ("docs/agent-notes/backend-dialects-and-idents.md", "backend=cc", "backend");
    ]
    ~f:(fun (path, spelling, expected_key) ->
      match one_assignment ~path spelling with
      | Some (key, ambiguous_bare) -> String.equal key expected_key && ambiguous_bare
      | None -> false);
  Verdict.p "a new registered one-word assignment remains classifiable for a required judgment"
    (match one_assignment ~path:"new-one-word.md" "backend=metal" with
    | Some (key, ambiguous_bare) -> String.equal key "backend" && ambiguous_bare
    | None -> false);
  Verdict.p "a tracked historical one-word typo remains classifiable after registry removal"
    (match one_assignment ~path:"docs/proposals/gh-ocannl-409.md" "bacend=multicore_cc" with
    | Some (key, ambiguous_bare) -> String.equal key "bacend" && not ambiguous_bare
    | None -> false);
  let grammar_text = In_channel.read_all grammar_fixture in
  let grammar_candidates = complete_inline_code_candidates grammar_text in
  let grammar_occurrences =
    markdown_occurrences ~allow_bare:true
      ~path:(Stdlib.Filename.basename grammar_fixture)
      grammar_text
  in
  Verdict.p_all ~min:3
    "each promised non-OCANNL assignment is present in the fixture and rejected as a config token"
    [ "fastMathEnabled=false"; "mathMode=Safe"; "d=1" ] ~f:(fun spelling ->
      List.mem grammar_candidates spelling ~equal:String.equal
      && not
           (List.exists grammar_occurrences ~f:(fun occurrence ->
                String.equal occurrence.spelling spelling)));
  Verdict.p_exists "a real documented OCANNL assignment remains a config token" grammar_occurrences
    ~f:(fun occurrence ->
      String.equal occurrence.key "debug_log_from_routines"
      && String.equal occurrence.spelling "debug_log_from_routines=true");
  check ~fail ~known_keys ~repository_census:true
    [
      occurrence ~ambiguous_bare:true ~path:"missing-spaced.md" ~key:"backend"
        ~spelling:"backend = cc" ~kind:Markdown_assignment ();
      occurrence ~path:"missing-prefix-free.ml" ~key:"backend" ~spelling:"--backend=cc"
        ~kind:Prefix_free_cli_flag ();
      occurrence ~path:"unknown.sh" ~key:"definitely_missing"
        ~spelling:("--ocannl_" ^ "missing=true") ~kind:Cli_flag ();
      occurrence ~path:"undeclared-alternate.md" ~key:"print_decimals_precision"
        ~spelling:("--ocannl_" ^ "print_decimals_precision-8")
        ~kind:Cli_flag ();
      occurrence ~ambiguous_bare:true ~path:"new-one-word.md" ~key:"backend"
        ~spelling:"backend=metal" ~kind:Markdown_assignment ();
    ];
  Verdict.p_all ~min:11 "every config-usage direct refusal format is observed"
    direct_refusal_formats ~f:(Hash_set.mem observed);
  Verdict.p_empty "the config-usage refusal control emits no unexpected diagnostic"
    ~over:(Hash_set.to_list observed) !unexpected;
  Refusal_manifest.print source

let file_kind path =
  let basename = Stdlib.Filename.basename path in
  if String.is_suffix path ~suffix:".md" then `Markdown
  else if String.equal path "ocannl_config.reference" then `Reference
  else if String.equal basename "dune" then `Dune
  else if String.equal basename "ocannl_config" || String.equal basename "ocannl_config.for_debug"
  then `Config
  else `Script

let scanned_source path =
  (not (String.is_suffix path ~suffix:".pp.ml" || String.is_suffix path ~suffix:".pp.mli"))
  &&
  let basename = Stdlib.Filename.basename path in
  let under roots = List.exists roots ~f:(fun root -> String.is_prefix path ~prefix:root) in
  let script = String.is_suffix path ~suffix:".sh" || String.is_suffix path ~suffix:".py" in
  let ocaml = String.is_suffix path ~suffix:".ml" || String.is_suffix path ~suffix:".mli" in
  let markdown =
    String.is_suffix path ~suffix:".md"
    && (String.equal path "AGENTS.md" || String.equal path "README.md"
       || under [ ".claude/skills/"; "docs/"; "benchmarks/" ])
  in
  String.equal path "ocannl_config.reference"
  || String.equal path "ocannl_config.for_debug"
  || String.equal basename "dune"
  || String.equal basename "ocannl_config"
  || script
  || ocaml
     && under [ "tools/"; "benchmarks/"; "bin/"; "test/"; "arrayjit/lib/"; "tensor/"; "lib/" ]
  || markdown
  || under [ ".github/workflows/" ]
     && (String.is_suffix path ~suffix:".yml" || String.is_suffix path ~suffix:".yaml")

let dedup_occurrences occurrences =
  let seen = Hash_set.create (module String) in
  List.filter occurrences ~f:(fun occurrence ->
      let id =
        Printf.sprintf "%s\000%d\000%s\000%s" occurrence.path occurrence.line occurrence.key
          occurrence.spelling
      in
      if Hash_set.mem seen id then false
      else (
        Hash_set.add seen id;
        true))

let occurrences_of_file ~reported_path path =
  let content = In_channel.read_all path in
  let primary =
    match file_kind reported_path with
    | `Markdown ->
        let allow_bare =
          (not (String.is_prefix reported_path ~prefix:"benchmarks/"))
          || String.equal reported_path "benchmarks/README.md"
        in
        markdown_occurrences ~allow_bare ~path:reported_path content
    | `Config ->
        config_file_occurrences
          ~include_commented:(String.equal reported_path "ocannl_config.for_debug")
          ~path:reported_path content
    | `Reference ->
        dedup_occurrences
          (script_occurrences ~path:reported_path content
          @ markdown_occurrences ~allow_bare:true ~path:reported_path content)
    | `Dune | `Script -> script_occurrences ~path:reported_path content
  in
  primary
  @ prefix_free_occurrences_for ~path:reported_path
      ~keys:(tracked_prefix_free_keys reported_path)
      content

let fixture path config_path multiline_path =
  let reported_path = Stdlib.Filename.basename path in
  let content = In_channel.read_all path in
  check ~repository_census:false
    (script_occurrences ~path:reported_path content
    @ markdown_occurrences ~allow_bare:true ~path:reported_path content
    @ prefix_free_occurrences_for ~path:reported_path
        ~keys:[ "definitely_not_a_prefix_free_config_key" ]
        content
    @ config_file_occurrences ~include_commented:true
        ~path:(Stdlib.Filename.basename config_path)
        (In_channel.read_all config_path)
    @ markdown_occurrences ~allow_bare:true
        ~path:(Stdlib.Filename.basename multiline_path)
        (In_channel.read_all multiline_path))

let live workspace_root generated =
  let inventory = Inventory.of_dune_sandbox ~workspace_root ~generated in
  let files =
    Inventory.select inventory ~f:scanned_source
    |> List.map ~f:(fun (file : Inventory.file) -> (file.path, file.on_disk))
  in
  let scripts =
    List.filter files ~f:(fun (path, _) ->
        String.is_suffix path ~suffix:".sh" || String.is_suffix path ~suffix:".py")
  in
  Verdict.p_all "the source inventory supplies shell and Python files" [ ".sh"; ".py" ]
    ~f:(fun suffix -> List.exists scripts ~f:(fun (path, _) -> String.is_suffix path ~suffix));
  Verdict.p_all "the source inventory supplies OCaml implementation and interface files"
    [ ".ml"; ".mli" ] ~f:(fun suffix ->
      List.exists files ~f:(fun (path, _) -> String.is_suffix path ~suffix));
  Verdict.p_exists "the source inventory supplies Markdown" files ~f:(fun (path, _) ->
      String.is_suffix path ~suffix:".md");
  Verdict.p "the source inventory supplies workflow YAML"
    (List.exists files ~f:(fun (path, _) ->
         String.is_suffix path ~suffix:".yml" || String.is_suffix path ~suffix:".yaml"));
  Verdict.p "the scan reaches Dune actions"
    (List.exists files ~f:(fun (path, _) -> String.equal (Stdlib.Filename.basename path) "dune"));
  Verdict.p_exists "the source inventory supplies checked-in ocannl_config files" files
    ~f:(fun (path, _) -> String.equal (Stdlib.Filename.basename path) "ocannl_config");
  Verdict.p "the scan reaches ocannl_config.reference examples"
    (List.exists files ~f:(fun (path, _) -> String.equal path "ocannl_config.reference"));
  Verdict.p "the scan reaches the checked-in debug config template"
    (List.exists files ~f:(fun (path, _) -> String.equal path "ocannl_config.for_debug"));
  let occurrences =
    List.concat_map files ~f:(fun (reported_path, path) -> occurrences_of_file ~reported_path path)
  in
  check ~repository_census:true occurrences;
  let cli_count =
    List.count occurrences ~f:(fun o ->
        Poly.equal o.kind Cli_flag || Poly.equal o.kind Prefix_free_cli_flag)
  in
  let env_count = List.count occurrences ~f:(fun o -> Poly.equal o.kind Environment_assignment) in
  let standalone_env_count =
    List.count occurrences ~f:(fun o -> Poly.equal o.kind Standalone_environment_mention)
  in
  let markdown_count = List.count occurrences ~f:(fun o -> Poly.equal o.kind Markdown_assignment) in
  let config_count =
    List.count occurrences ~f:(fun o -> Poly.equal o.kind Config_file_assignment)
  in
  eprintf
    "config usage scan: %d files, %d command-line flags, %d environment assignments, %d standalone \
     environment mentions, %d documentation assignments, %d config-file assignments\n"
    (List.length files) cli_count env_count standalone_env_count markdown_count config_count;
  if not (Verdict.any_failed ()) then (
    printf
      "OK: qualified OCANNL command-line flags and OCANNL_<KEY> environment mentions in source \
       scripts and OCaml sources name registered keys.\n";
    printf
      "OK: inline key=value assignments in scanned Markdown name registered keys or explicit \
       non-config notation.\n";
    printf
      "OK: configuration tokens in Dune actions and checked-in ocannl_config files name registered \
       keys or counted intentional-invalid controls.\n");
  Refusal_manifest.print "config_usage_scan.ml"

let () =
  match Array.to_list argv with
  | _ :: [ "--refusal-control"; grammar_fixture ] -> refusal_control grammar_fixture
  | _ :: [ "--fixture"; path; config_path; multiline_path ] ->
      fixture path config_path multiline_path
  | _ :: workspace_root :: generated when not (List.is_empty generated) ->
      live workspace_root generated
  | argv ->
      eprintf
        "Usage: %s <workspace_root> <generated sandbox paths...> | %s --fixture <file> \
         <config-file> <multiline-markdown-file>\n"
        (List.hd_exn argv) (List.hd_exn argv);
      Stdlib.exit 1
