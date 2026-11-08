(** Repository source readers, separate from the runtime helpers in {!Test_utils}.

    Linking ppxlib pulls in the compiler archive, whose units are forced into every native
    executable. Keep that dependency on the scanning side (gh-ocannl-2010). *)

module Config_key_scan = Config_key_scan
(** Scanning OCaml sources for the config keys they read, shared by the configuration-consistency
    tests. *)

module Lifecycle_scan = Lifecycle_scan
module Dune_stanza_scan = Dune_stanza_scan

module Config_dep_resolution = Config_dep_resolution
(** Which configuration each test-running site's process reads, and whether its dependencies name
    it: [config_dep_completeness]'s resolution, over a tree handed in as data (gh-ocannl-603). *)

(* Which backends a dune argv can reach a stanza naming (gh-ocannl-1004, gh-ocannl-1066): the
   reachability half of tools/batch-backends.sh's resolution of a batch's backends. *)
module Slot_kind = Slot_kind
(** Reading dune files for the stanzas that run a test executable, and whether they declare the
    shared [ocannl_config]. *)

module Cache_dir_scan = Cache_dir_scan
(** Scanning OCaml sources for the autotune schedule cache directories they name, so that the one
    root [.gitignore] glob over their shared prefix covers all of them. *)

module Refused_action_scan = Refused_action_scan
module Refusal_control_scan = Refusal_control_scan

module Refusal_section_heads = Refusal_section_heads
(** The fixed text the refusal-control section printer writes for readers outside its process,
    [tools/mutation-run.sh] among them, which reads it from the source (gh-ocannl-1840). *)

module Refusal_control_manifest = Refusal_control_manifest
(** Extracting static refusal-diagnostic fragments from repository scanners and relating them to
    permanent control goldens. *)

module Refusal_manifest_sync = Refusal_manifest_sync
(** Rewriting [Refusal_control_manifest]'s stale rows, keys and labels after a diagnostic is
    reworded: tools/refusal-manifest-sync.sh's engine (gh-ocannl-1429). *)

module Optional_arg_scan = Optional_arg_scan
(** Classifying whether [lib/] optional arguments affect their function bodies or are deliberately
    unimplemented, in which case the caller-visible label must begin with an underscore. *)

module Verdict_callee = Verdict_callee
(** Whose a claim name is -- [Verdict]'s, ambiguously so past an [open] of another module, or not:
    the one resolver every [Verdict] reader asks (gh-ocannl-1361). *)

module Verdict_scan = Verdict_scan
(** Scanning test sources for claims a test decides itself and prints outside [Verdict], where a
    failing one is [dune promote]-able into the golden. *)

module Verdict_provenance = Verdict_provenance
(** Scope-and-polarity provenance of the Boolean a [Verdict] claim receives: which quantifiers it
    can rest on, through bindings, helpers, wrappers, matches and modules, and which populations are
    witnessed non-empty when it holds. *)

module Doc_examples_scan = Doc_examples_scan

module Agent_notes_scan = Agent_notes_scan
(** Reading [docs/agent-notes.md] and [docs/agent-notes/] as structure: bullet integrity, index-hook
    agreement, table shape, reachability from the index, and repetition across files. *)

module Markdown_link_scan = Markdown_link_scan
(** Checking the closed inline-link dialect against repository source paths. *)

module Dead_export_scan = Dead_export_scan

module Api_drift = Api_drift
(** Enumerating source-declared values in modules without interfaces and conservatively counting
    external qualified, aliased, opened, and included references to them. *)

module Codegen_text_scan = Codegen_text_scan
(** Deciding what pins the TEXT of generated code: goldens holding emitted kernel or IR source, and
    test sources asserting on it from a string literal. *)

module Kernel_exec_scan = Kernel_exec_scan
(** Deciding which tests compile or execute kernel text they built themselves -- a staged callback,
    a [?prelowered] compile, or a support value reaching either -- the population a codegen change
    breaks without any of it pinning text (gh-ocannl-1685). *)

module Scan_attempt = Scan_attempt
(** Classifying a scan's failure on one source: the input's parse error, or the scanner's own
    exception with its text -- never the second reported as the first. *)

module Source_inventory = Source_inventory
(** A source-only repository inventory derived from a clean Dune sandbox. Repository scans select
    their corpus from this shared set instead of maintaining recursive source-root lists. *)

module Rerun_aliases = Rerun_aliases
(** Which aliases re-run a file that pins the text of generated code: read from the dune files, the
    rule diffing a golden or the stanza running a source (gh-ocannl-1533). *)

module Scan_argv = Scan_argv
(** Response-file expansion for the repository-wide scans: a [@<path>] argument stands for the words
    in that file. What a scan reads is handed to it on the command line, and Windows caps a command
    line at 32,767 characters. *)

module Scan_floors = Scan_floors
(** Floors over a scanned census: the tripwire that keeps a scanning test from passing vacuously,
    shared by the scans that glob the repository. *)

module Lexical_scope = Lexical_scope
(** The lexical scope model the source scans resolve names over: which binding an identifier reaches
    where it is spelled, parameterised over what a name denotes (gh-ocannl-1079). *)

module Ll_test_scan = Ll_test_scan

module Operand_key_scan = Operand_key_scan
(** Finding operand fixtures that mint values from a hand-rolled modulus of a multi-axis key instead
    of going through [Ll_test.cycle]'s blind-axis guard (gh-ocannl-1018). *)

module Exact_partials_scan = Exact_partials_scan
(** Prose claiming the exactness of partial sums, and whether an [Ll_test.exact_partials] derivation
    stands beside it (gh-ocannl-1729). *)

module Mutation_serial_scan = Mutation_serial_scan
(** Holding every write to the inference session's module-level mutable state to
    [Row.note_mutation], or to an in-place classification (gh-ocannl-1506). *)

module Open_node_scan = Open_node_scan
(** Holding every tensor node a module of [tensor/] creates to [Tensor.note_open_node], the uid list
    a rewind discards by (gh-ocannl-1579). *)

module Debug_name_scan = Debug_name_scan
(** Holding every test's use of a node's debug name to display, never lookup (gh-ocannl-1606). *)

module Label_dispatch_scan = Label_dispatch_scan
(** Holding the compiler's reads of a node's label or namespace, and of a backend's name, to
    display, a copied label or a marked site, never a decision (gh-ocannl-1783). *)

module Tensorized_census_scan = Tensorized_census_scan
(** Finding executed claims about a tensorized routine whose item judges no mma census, which parity
    alone cannot replace (gh-ocannl-1309). *)

module Provenance_tag_scan = Provenance_tag_scan
(** Where the virtualizer's rejection codes are minted, which files name them, and the boundary
    test's phase table checked against the minting functions (gh-ocannl-1015). *)
