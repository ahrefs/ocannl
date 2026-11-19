open Base

(** The [@policy-invariance] family (gh-ocannl-1104): which tests the daily sweep's policy units run
    under every non-default fp16/bf16 mode, derived from what the tests' own sources name.

    A member is a TEST -- a [(test)] or [(tests)] program, an inline-test library, or an
    [(executable)] some rule runs on a [runtest-] alias ({!runs_as_test}) -- one of whose own
    modules names the numerics policy: the module [Ir.Numerics] itself, or a value of
    [test/support/test_utils.ml] whose definition does -- its [with_policy] wrappers, derived by
    {!wrappers} rather than listed. Those are the programs that set the process-wide policy for some
    of their legs and read the ambient one in others, which is what makes a default-labelled claim
    in them test another policy whenever the caller exported a mode ([OCANNL_FP16_ARITHMETIC],
    [OCANNL_BF16_ARITHMETIC]), with no run but the sweep's ever exporting one.

    What it reads, and deliberately does not:
    - a member's own modules, and every arm of a [(select …)] that produces one, under the scanner's
      lexical module model ({!Config_key_scan.module_reference_sites}): an alias or an [open] of
      [Ir] counts, a comment or a string literal spelling [Numerics] does not;
    - of [test/support], [test_utils.ml] alone. A support value that sets the policy and restores it
      inside its own computation -- [Ll_test.tensorized_capability]'s tf32 lift, which runs no code
      of its caller's under it -- makes no member of its callers;
    - no program that is not run as a test: a benchmark under [bin/] naming the policy measures
      under whatever mode it is given, which is its caller's choice rather than a claim. Nor an
      [(executable)] whose only runs write a target a separate rule diffs: the family's units are
      the ones [env_var_deps]' aggregate machinery credits with an alias, and [cc_march_census], the
      one such program today, forces every child's fp16 mode itself;
    - of the family stanza, its direct [(alias …)] entries, each with the stanza's [(enabled_if …)];
      what a listed member's own dependencies pull in is that member's to answer for;
    - nothing about the precisions a test computes at. A test computing in half without naming the
      policy is outside the family: naming it is the decision that makes a test's legs depend on the
      mode, and the remaining tests answer for the default mode in every other run.

    A member whose claims deliberately follow the DEFAULT policy -- a premise it states as a claim
    -- opts out inside its own stanza, the reason saying which claims and why:
    {v ; ocannl-policy-invariance: exempt -- <reason> v}
    and the alias leaves it out. The opt-out is the whole stanza under every mode: a member that
    holds under some modes and not others pins the legs that need the default instead
    ([Test_utils.with_policy]), which keeps the rest of it in the matrix. *)

module Dune = Dune_stanza_scan
module Sources = Config_key_scan

let family_alias = "policy-invariance"
let sentinel = "ocannl-policy-invariance:"
let declaration = "exempt"
let support_source = "test/support/test_utils.ml"
let support_module = "Test_utils"
let policy_module = [ "Ir"; "Numerics" ]

(** The values of [test_utils.ml] whose definitions reach [Ir.Numerics], directly or through another
    of its values, each as the path a test names it by below the module ([with_policy],
    [with_fp16_auto], ...). *)
let wrappers content =
  let tag = "<numerics policy>" in
  (Sources.top_level_key_reach
     ~external_keys:(fun path ->
       if List.is_prefix path ~prefix:policy_module ~equal:String.equal then [ tag ] else [])
     content)
    .reach
  |> List.filter_map ~f:(fun (name, keys) ->
      Option.some_if (List.mem keys tag ~equal:String.equal) name)
  |> List.dedup_and_sort ~compare:String.compare

(** The requested paths a member names: the policy module and each wrapper under [Test_utils]. *)
let paths ~wrappers =
  policy_module :: List.map wrappers ~f:(fun w -> support_module :: String.split w ~on:'.')

(** Which of {!paths} [content] names, as the source spells them. Raises what the parser raises, for
    the caller's {!Scan_attempt}. *)
let policy_references ~wrappers content =
  Sources.module_references_in_source content ~paths:(paths ~wrappers)

(** The stanza heads a member can be: the programs dune links modules into, and an inline-test
    library. *)
let runnable stanza =
  match Dune.head stanza with
  | Some ("test" | "tests" | "executable" | "executables") -> true
  | Some "library" -> Option.is_some (Dune.field stanza "inline_tests")
  | _ -> false

(** Whether a unit built by [stanza] and reached by [aliases] runs as a test: a [(test)]/[(tests)]
    program or inline-test library runs on its own alias; an [(executable)] counts when some rule
    runs it on a [runtest-] alias. *)
let runs_as_test stanza ~aliases =
  match Dune.head stanza with
  | Some ("test" | "tests" | "library") -> runnable stanza
  | Some ("executable" | "executables") ->
      List.exists aliases ~f:(String.is_prefix ~prefix:"runtest-")
  | _ -> false

(** The source files a module [m] of [stanza] is compiled from: [m.ml], or every arm of a
    [(select m.ml from …)] when dune generates it. *)
let module_sources stanza m =
  let target = String.lowercase m ^ ".ml" in
  match
    List.filter_map (Dune.select_arms stanza) ~f:(fun (t, arm) ->
        Option.some_if (String.equal (String.lowercase t) target) arm)
  with
  | [] -> [ target ]
  | arms -> arms

type opt_out = { opt_out_line : int; opt_out_reason : string }

let opt_out_contract content =
  Dune.contained_marker_contract content ~sentinel
    ~parse_declaration:(fun _ ~declaration:written ~reason ->
      if String.equal written declaration then Ok reason
      else
        Error
          [
            Printf.sprintf
              "`%s` is not `%s` -- the opt-out leaves the whole stanza out of the family under \
               every mode; a leg that needs the default pins it with Test_utils.with_policy"
              written declaration;
          ])
    ~belongs:(fun marked _ ->
      if runnable marked.Dune.marked_sexp then Ok ()
      else
        Error
          [
            Printf.sprintf
              "a `%s` stanza links no modules -- the opt-out goes on the `(test)` or \
               `(executable)` whose modules name the numerics policy"
              marked.Dune.marked_head;
          ])

(** What is wrong with the file's opt-out markers, one line each, phrased for their author. *)
let issues contract =
  let grammar = Printf.sprintf "%s %s -- <reason>" sentinel declaration in
  List.map contract.Dune.contract_issues ~f:(function
    | Dune.Malformed_marker { issue_line; issue_malformed; _ } ->
        Printf.sprintf "line %d: %s" issue_line
          (Dune.marker_malformed_reason ~sentinel ~separator_subject:"declaration" ~grammar
             issue_malformed)
    | Dune.Marker_outside_stanza { issue_line; _ } ->
        Printf.sprintf
          "line %d: a policy-invariance opt-out between stanzas declares nothing -- move it inside \
           the parentheses of the test it is about"
          issue_line
    | Dune.Marker_in_wrong_stanza { issue_line; issue_why; _ } ->
        Printf.sprintf "line %d: %s" issue_line issue_why
    | Dune.Marker_outside_comment { issue_text_occurrences; issue_comment_occurrences } ->
        Printf.sprintf
          "spells `%s` %d times and only %d of them are in a comment -- an opt-out outside a \
           comment declares nothing"
          sentinel issue_text_occurrences issue_comment_occurrences)
  @ List.filter_map contract.Dune.contract_stanzas ~f:(fun marked ->
      match marked.Dune.stanza_markers with
      | _ :: second :: _ ->
          Some
            (Printf.sprintf "line %d: a second policy-invariance opt-out on one stanza"
               second.Dune.contained_line)
      | _ -> None)

(** Every stanza carrying a well-formed opt-out, with its [(subdir …)] and the opt-out. *)
let opt_outs contract =
  List.concat_map contract.Dune.contract_stanzas ~f:(fun marked ->
      List.map marked.Dune.stanza_markers ~f:(fun marker ->
          ( marked.Dune.marker_stanza.Dune.marked_subdir,
            marked.Dune.marker_stanza.Dune.marked_sexp,
            {
              opt_out_line = marker.Dune.contained_line;
              opt_out_reason = marker.Dune.contained_value;
            } )))

(** The direct [(alias …)] dependencies of every [(alias (name policy-invariance) …)] stanza in one
    directory's [stanzas]. *)
let listed stanzas =
  List.concat_map stanzas ~f:(fun stanza ->
      match (Dune.head stanza, Dune.field stanza "name") with
      | Some "alias", Some [ Sexp.Atom name ] when String.equal name family_alias ->
          let condition = Dune.field stanza "enabled_if" in
          Option.value_map (Dune.field stanza "deps") ~default:[] ~f:(fun deps ->
              List.filter_map deps ~f:(function
                | Sexp.List [ Sexp.Atom "alias"; Sexp.Atom alias ] -> Some (condition, alias)
                | _ -> None))
      | _ -> [])

(** Whether the family stanza listing a member is enabled exactly when the member is: a member under
    [(enabled_if …)] listed by an unconditional stanza leaves the family undefined wherever its
    condition is false ([No rule found for alias]), and one listed under a narrower condition than
    its own drops out wherever the two differ. Compared as written. *)
let same_condition ~listing ~member =
  Option.equal (List.equal Sexp.equal) listing (Dune.field member "enabled_if")

(** The diagnostics, phrased here so the scan states each relationship in one place. *)
let reading_refused ~source ~where =
  Printf.sprintf
    "%s does not parse as OCaml, so whether it names the numerics policy cannot be established: %s"
    source where

let reading_raised ~source ~exn =
  Printf.sprintf
    "the policy-invariance reading raised %s on %s, which parses -- a defect in this scan, so its \
     family membership went unchecked"
    exn source

let support_lost ~found =
  if found then
    Printf.sprintf
      "%s defines no value reaching %s -- the wrapper derivation has stopped finding the \
       with_policy family, and a test reaching the policy only through one would drop out of @%s \
       silently"
      support_source
      (String.concat ~sep:"." policy_module)
      family_alias
  else
    Printf.sprintf
      "this check was handed no %s, so the policy wrappers a test may name instead of %s cannot be \
       derived"
      support_source
      (String.concat ~sep:"." policy_module)

let stale ~dune_file ~alias =
  Printf.sprintf
    "%s: the `%s` alias lists `(alias %s)`, and no test there names the numerics policy -- a \
     member that sets no policy runs under every mode for nothing and reads as coverage. Drop it, \
     or name the policy where the test depends on it"
    dune_file family_alias alias

let opted_out_listed ~dune_file ~identity ~line =
  Printf.sprintf
    "%s:%d opts the %s out of the numerics-mode matrix, and the `%s` alias still reaches it -- \
     drop it from the alias or drop the opt-out"
    dune_file line identity family_alias

let condition_mismatch ~dune_file ~alias =
  Printf.sprintf
    "%s: the `%s` stanza listing `(alias %s)` is not enabled under the member's own `(enabled_if \
     …)` -- list a conditional member in a family stanza carrying the same condition, and an \
     unconditional one in an unconditional stanza"
    dune_file family_alias alias

let opt_out_on_non_member ~dune_file ~line =
  Printf.sprintf
    "%s:%d carries a policy-invariance opt-out on a stanza none of whose programs names the \
     numerics policy -- it is no member to opt out; drop the marker"
    dune_file line
