(** Static census of a generated kernel's innermost loops, and the [-march] compile matrix that
    feeds it (gh-ocannl-650).

    Two questions about an emitted kernel are answerable without owning the hardware it targets, and
    neither was answerable by anything in the repository until this module:

    - {b does the guarded arm even compile where its guard is true?} Every [#elif] for a foreign
      target ({!C_syntax.vec_fma_builtin}'s AVX-512, AVX512-FP16 and NEON rows) is emitted
      unconditionally and preprocessed away on the build host, so a syntax error, a wrong arity or a
      type mismatch inside one ships silently. gh-ocannl-621 caught exactly that -- gcc's aarch64
      fp16 vector builtin is typed in [__fp16], not the [_Float16] the emission carries -- by
      compiling [build_files/*.c] under seven [-march] flags in a scratch directory that died with
      the session.
    - {b does it compile WELL?} The same sweep was gh-ocannl-621's actual measurement: instructions,
      vector and scalar arithmetic, and stack references in the innermost loop that carries the
      accumulator update. That is a compile-time property, so it needs no hardware either, and it is
      what separates "gcc accepted the arm" from "gcc kept it in registers as one vector operation".

    {1 The two false readings this module is shaped around}

    Re-deriving the census by hand cost gh-ocannl-621 three iterations and two wrong answers, both
    of which a naive implementation reproduces:

    - {b selecting an outer loop.} A reduction kernel nests: the accumulator update sits in the
      innermost loop, and counting an enclosing one dilutes every ratio. {!census} therefore selects
      the {e smallest-span} loop whose body carries the construct being censused, and it identifies
      that construct through DWARF line numbers ([-g] puts [.loc] directives in the assembly) rather
      than by guessing from the instruction mix. {!census_in}'s explicit outer-carrier selector is
      the narrow exception for a construct with its own nested codec loop. The caller supplies the
      source-line set; a loop is a candidate only if its body's [.loc] set meets it.
    - {b counting only the instructions the good outcome produces.} Scoring vector FMAs alone makes
      a fully scalarized loop read as "no FMA loop found" -- a pass, from the failure. So the counts
      are separated ({!counts.vector_ops} against {!counts.scalar_fp_ops} against
      {!counts.libm_calls}) and the SELECTION never consults them: it is anchored on the source, so
      a loop whose combine compiled to nothing but library calls is still found and still censused.
      {!census} answering [None] means the anchor was not reached at all, which callers report as a
      failure rather than as an absence of findings.

    {1 What the counts mean}

    Classification is by mnemonic, from GCC/GAS and Clang assembly output for x86-64 and aarch64:

    - {b vector_ops}: packed-SIMD instructions -- an x86 mnemonic ending in [ps]/[pd]/[ph], a packed
      integer mnemonic, an operand naming [%ymm]/[%zmm], an aarch64 operand in vector-lane form
      ([v0.2d], [q0]), or an aarch64 mnemonic carrying the arrangement instead ([fmla.4s]), which is
      how Apple's assembler spells the same instruction.
    - {b scalar_fp_ops}: an x86 mnemonic ending in [ss]/[sd]/[sh], or an aarch64 [f...] instruction
      on a scalar FP register. This is the count that gives away SLP scalarization, which leaves no
      stack traffic behind.
    - {b libm_calls}: a call to the censused class's library function ([fmax]/[fmin] and friends,
      [fma]/[fmaf]). One of these inside a vector loop is the worst outcome of all: an opaque call
      cannot be vectorized at any optimization level or grid size.
    - {b stack_refs}: instructions addressing through the stack or frame pointer -- the spill signal
      gh-ocannl-614 measured.
    - {b residual}: instructions matched by none of those classifiers. Loop-control and integer
      bookkeeping legitimately live there, so it is reported rather than bounded; its purpose is to
      make a newly encountered dialect visible instead of silently returning zeroes.

    Both aarch64 spellings are read, because both are compiled in CI and a dialect must not change
    the reading: a count is a fact about the instruction, and the same loop assembled two ways has
    to census the same. Both are exercised in [test/operations/cc_march_census]'s dialect probes.

    Moves are the fuzzy edge (an [%xmm] operand does not by itself say whether a [movq] moved a lane
    or a vector), so the mnemonic rule below is deliberately narrow and the counts are reported as a
    profile rather than compared against fitted thresholds. Every claim built on them is an
    inequality that a wrong classification of a move cannot flip. *)

open Base

(** {1 Toolchains} *)

type toolchain = {
  label : string;  (** how the census table names this column, e.g. ["x86-64-v3"] *)
  command : string;  (** the compiler command, possibly a cross compiler *)
  march : string;  (** the [-march=] value, or [""] for the compiler's default target *)
  note : string;  (** what this column is here to exercise, for the skip line *)
}

let flags t = if String.is_empty t.march then "" else "-march=" ^ t.march

(* Compiler invocations go through the shell with output captured to a file, the way [Cc_backend]'s
   own probes do: the exit status is what decides, and the output is only wanted when it is nonzero.
   Paths are quoted -- these sit under the build directory or the system temp directory, either of
   which can contain whitespace. *)
let run_capture cmdline =
  let log = Stdlib.Filename.temp_file "ocannl_census_" ".log" in
  let rc = Stdlib.Sys.command (Printf.sprintf "%s > %s 2>&1" cmdline (Stdlib.Filename.quote log)) in
  let out = try Stdio.In_channel.read_all log with _ -> "(unable to read compiler output)" in
  (try Stdlib.Sys.remove log with _ -> ());
  (rc, out)

(** [toolchain_identity t] asks the compiler to identify the whole toolchain it will run: compiler
    build and version, configured target, and installation/configuration details. The result is
    deliberately the complete [-v] response rather than a version parsed from [t.command] or from a
    package name. A caller may digest it, but should not reinterpret it: the toolchain owns the
    identity and a changed answer is what invalidates compiler-derived data. *)
let toolchain_identity t =
  let rc, out = run_capture (Printf.sprintf "%s -v" t.command) in
  if rc = 0 then Ok out else Error out

(** [preprocessed_source t ~opt_level ~source] asks the toolchain for its complete resolved view of
    a translation unit under the target and optimization flags the measured compile uses. This
    includes the exact SDK/header inputs selected by conditional compilation and include-search
    environment, without parsing their versions or paths. *)
let preprocessed_source t ~opt_level ~source =
  let src = Stdlib.Filename.temp_file "ocannl_census_preprocess_" ".c" in
  Stdio.Out_channel.write_all src ~data:source;
  (* [-dD] retains macro definitions as well as declarations: a header or predefined macro can
     change generated assembly without changing any declaration text that survives ordinary
     preprocessing. [-P] removes temp-file line markers, so the probe is stable across runs. *)
  let rc, out =
    run_capture
      (Printf.sprintf "%s %s -O%d -E -P -dD %s" t.command (flags t) opt_level
         (Stdlib.Filename.quote src))
  in
  (try Stdlib.Sys.remove src with _ -> ());
  if rc = 0 then Ok out else Error out

(** [compilation_plan t ~opt_level ~source] asks the compiler driver to expand the effective
    compilation it would run, including response files, GCC specs, wrapper-injected options and
    target subtools. Probe-local paths are normalized out; they name scratch files rather than an
    input to code generation. *)
let compilation_plan t ~opt_level ~source =
  let src = Stdlib.Filename.temp_file "ocannl_census_plan_" ".c" in
  let asm = Stdlib.Filename.temp_file "ocannl_census_plan_" ".s" in
  Stdio.Out_channel.write_all src ~data:source;
  let rc, out =
    run_capture
      (Printf.sprintf "%s %s -O%d -g -S -### -o %s %s" t.command (flags t) opt_level
         (Stdlib.Filename.quote asm) (Stdlib.Filename.quote src))
  in
  let normalize_path out path marker =
    let variants =
      [
        path; Stdlib.Filename.basename path; String.substr_replace_all path ~pattern:"\\" ~with_:"/";
      ]
      |> List.dedup_and_sort ~compare:String.compare
      |> List.filter ~f:(fun path -> not (String.is_empty path))
    in
    List.fold variants ~init:out ~f:(fun out pattern ->
        String.substr_replace_all out ~pattern ~with_:marker)
  in
  let out =
    normalize_path out src "<SOURCE>" |> fun out ->
    normalize_path out asm "<ASSEMBLY>" |> fun out ->
    normalize_path out (Stdlib.Sys.getcwd ()) "<CWD>"
  in
  (try Stdlib.Sys.remove src with _ -> ());
  (try Stdlib.Sys.remove asm with _ -> ());
  if rc = 0 then Ok out else Error out

(** [accepts t] is whether this toolchain exists and accepts its [-march]. A column that answers
    [false] must be reported (see [Verdict.skipped]) rather than dropped: a silently missing column
    is indistinguishable from a passing one. *)
let accepts t =
  let src = Stdlib.Filename.temp_file "ocannl_census_probe_" ".c" in
  (* A temp file, NOT [-o /dev/null]: [Sys.command] hands the line to the platform shell, and on
     native Windows there is no [/dev/null] to hand it -- every probe would fail, every column would
     report itself [Verdict.skipped], and the golden would be green having compiled nothing. That is
     the exact failure this module's "a missing column is not a passing one" rule exists to prevent,
     so it must not be reintroduced by the probe that decides which columns exist (AGENTS.md names
     Windows a supported environment). Writing real assembly costs a few kilobytes and is
     portable. *)
  let out = Stdlib.Filename.temp_file "ocannl_census_probe_" ".s" in
  Stdio.Out_channel.write_all src ~data:"int ocannl_census_probe(void) { return 0; }\n";
  let rc, _ =
    run_capture
      (Printf.sprintf "%s %s -S -o %s %s" t.command (flags t) (Stdlib.Filename.quote out)
         (Stdlib.Filename.quote src))
  in
  (try Stdlib.Sys.remove src with _ -> ());
  (try Stdlib.Sys.remove out with _ -> ());
  rc = 0

(** [defines t] is the set of preprocessor macro NAMES this toolchain predefines for its target.

    This is how a column whose ISA its label does not spell out -- the host's own default target,
    which is whatever machine the run landed on -- can still be held to an ISA-scoped claim: the
    compiler is asked what it targets, instead of a hard-coded table asserting what a [-march]
    string implies. It answers for the named columns too, which is strictly better than the table
    was: [-march=x86-64-v3] implying [__FMA__] is gcc's fact to state, not this module's. *)
let defines t =
  let src = Stdlib.Filename.temp_file "ocannl_census_macros_" ".c" in
  Stdio.Out_channel.write_all src ~data:"";
  let _, out =
    run_capture (Printf.sprintf "%s %s -dM -E %s" t.command (flags t) (Stdlib.Filename.quote src))
  in
  (try Stdlib.Sys.remove src with _ -> ());
  String.split_lines out
  |> List.filter_map ~f:(fun l ->
      match String.split_on_chars (String.strip l) ~on:[ ' '; '\t' ] with
      | "#define" :: name :: _ ->
          (* A function-like macro's name ends at its parenthesis. *)
          Some (match String.lsplit2 name ~on:'(' with Some (n, _) -> n | None -> name)
      | _ -> None)
  |> Set.of_list (module String)

(** [compile t ~opt_level ~src_path ~asm_path] compiles [src_path] to assembly, with [-g] so that
    {!census} can anchor on source lines. [Error output] carries the compiler's diagnostics.

    Assembly rather than an object file: [-S] runs the whole front end and code generator, which is
    everything a generated kernel can trip, and it produces the census input in the same invocation.
*)
let compile t ~opt_level ~src_path ~asm_path =
  let rc, out =
    run_capture
      (Printf.sprintf "%s %s -O%d -g -S -o %s %s" t.command (flags t) opt_level
         (Stdlib.Filename.quote asm_path) (Stdlib.Filename.quote src_path))
  in
  if rc = 0 then Ok () else Error out

(** {1 Counting} *)

type op_class = Fma | Max_min

type counts = {
  instructions : int;
  vector_ops : int;
  scalar_fp_ops : int;
  libm_calls : int;
  stack_refs : int;
  residual : int;
      (** Instructions recognized by none of the classifiers above. This is descriptive rather than
          a failure threshold: ordinary loop-control instructions live here too, but a dialect
          surprise can no longer disappear into passing-looking zeroes. *)
}

type t = { loop_label : string; span : int; counts : counts }

let libm_names = function
  | Fma -> [ "fma"; "fmaf"; "fmal"; "fmaf16" ]
  | Max_min -> [ "fmax"; "fmaxf"; "fmaxl"; "fmaxf16"; "fmin"; "fminf"; "fminl"; "fminf16"; "fdim" ]

(* A GAS line, already stripped of its trailing comment. *)
type line =
  | Label of string
  | Directive of string list
  | Insn of { mnemonic : string; rest : string }

let strip_comment s =
  (* x86 GAS comments start with '#', aarch64 GAS with "//", and Apple's arm64 assembler with ';'.
     None appears inside the operand syntaxes we look at, and compiler-generated assembly writes one
     instruction per line, so ';' is never the statement separator GAS also allows it to be.

     The ';' case is not cosmetic: Apple clang annotates loop headers as [LBB0_7: ; =>This Inner
     Loop Header: Depth=1], and a line that does not END in ':' is not classified as a label, so
     every backward branch to it is discarded and macOS reports no recognizable loops -- the third
     distinct Mach-O detail to silence this census wholesale, after the dot-less labels and the
     checksummed [.file]. *)
  let s =
    match String.substr_index s ~pattern:"//" with Some i -> String.prefix s i | None -> s
  in
  let s = match String.index s ';' with Some i -> String.prefix s i | None -> s in
  match String.index s '#' with Some i -> String.prefix s i | None -> s

let classify_line raw =
  let s = strip_comment raw in
  let trimmed = String.strip s in
  if String.is_empty trimmed then None
  else if
    (not (Char.is_whitespace raw.[0])) && String.is_suffix trimmed ~suffix:":"
    (* A label, including the compiler-internal [.L]/[.LVL]/[.LBB] ones, is written at column 0;
       directives and instructions are indented. *)
  then Some (Label (String.drop_suffix trimmed 1))
  else if String.is_prefix trimmed ~prefix:"." then
    Some
      (Directive
         (String.split_on_chars trimmed ~on:[ ' '; '\t' ]
         |> List.filter ~f:(fun w -> not (String.is_empty w))))
  else
    match String.lsplit2 trimmed ~on:'\t' with
    | Some (m, rest) -> Some (Insn { mnemonic = String.strip m; rest = String.strip rest })
    | None -> (
        match String.lsplit2 trimmed ~on:' ' with
        | Some (m, rest) -> Some (Insn { mnemonic = String.strip m; rest = String.strip rest })
        | None -> Some (Insn { mnemonic = trimmed; rest = "" }))

(* {2 Mnemonic classification}

   [v]-prefixed VEX/EVEX forms are stripped once, and an EVEX mask suffix ([{%k1}], [{z}]) lives in
   the operands rather than the mnemonic, so the suffix test below is on the plain mnemonic. *)

let x86_fp_suffix m =
  let m = if String.is_prefix m ~prefix:"v" then String.drop_prefix m 1 else m in
  (* [push]/[pop] would otherwise read as scalar-single-precision by their last two characters. *)
  if String.is_prefix m ~prefix:"push" || String.is_prefix m ~prefix:"pop" then None
  else if String.length m < 3 then None
  else
    match String.suffix m 2 with
    | ("ps" | "pd" | "ph") as s -> Some (`Packed, s)
    | ("ss" | "sd" | "sh") as s -> Some (`Scalar, s)
    | _ -> None

let packed_integer_mnemonic m =
  let m = if String.is_prefix m ~prefix:"v" then String.drop_prefix m 1 else m in
  String.is_prefix m ~prefix:"movdq"
  || List.exists
       [ "pand"; "pandn"; "por"; "pxor"; "pcmp"; "punpck"; "pblend"; "pshuf"; "pinsr"; "pextr" ]
       ~f:(fun p -> String.is_prefix m ~prefix:p)

let has_substr s ~sub = String.is_substring s ~substring:sub

(* An aarch64 operand in vector-lane form ([v3.2d], [v10.16b]) or a full 128-bit register ([q7]).
   Written as a scan rather than a regex to keep this module's dependencies at [base]. *)
let aarch64_vector_operand rest =
  let n = String.length rest in
  let rec scan i =
    if i >= n then false
    else if Char.equal rest.[i] 'v' && i + 1 < n && Char.is_digit rest.[i + 1] then
      (* v<digits>.<digits><lane letter> *)
      let rec digits j = if j < n && Char.is_digit rest.[j] then digits (j + 1) else j in
      let j = digits (i + 1) in
      if j < n && Char.equal rest.[j] '.' then true else scan j
    else if
      Char.equal rest.[i] 'q'
      && i + 1 < n
      && Char.is_digit rest.[i + 1]
      && (i = 0 || not (Char.is_alphanum rest.[i - 1]))
    then true
    else scan (i + 1)
  in
  scan 0

(* Apple's arm64 assembler writes a NEON instruction's ARRANGEMENT on the mnemonic ([fmla.4s v0, v1,
   v2]) where GAS writes it on the registers ([fmla v0.4s, v1.4s, v2.4s]), so on that dialect
   {!aarch64_vector_operand} sees three plain [v] names and reports nothing. That silences the
   COUNTERS -- not the selection, and not the loop edges, which is why it survived the three earlier
   Mach-O rounds: the loop was found, its span and instruction total were right, and only the
   vector/scalar split read zero. CI's macos-latest leg censused a register-tiled bf16 k-loop of 24
   [fmla.4s], 6 [shll.4s] and 4 [dup.4s] as [vector=0 scalar_fp=0] and failed the vector-majority
   claim on [0 > 0]; the two tile rows beside it PASSED that claim on 3 and 10 instructions the
   [q<n>] and [v<n>.<arr>] rules happened to catch out of 28 and 78 real ones -- a pass arrived at
   from the same blindness (gh-ocannl-752).

   The suffix has to be an arrangement and not merely a dot: aarch64 spells its conditional branches
   [b.ne], and counting those as vector work would make every loop in every listing report at least
   one. *)
let aarch64_arrangements = [ "8b"; "16b"; "4h"; "8h"; "2s"; "4s"; "1d"; "2d" ]

let aarch64_arrangement_mnemonic m =
  match String.rsplit2 m ~on:'.' with
  | Some (_, suffix) -> List.mem aarch64_arrangements suffix ~equal:String.equal
  | None -> false

let aarch64_scalar_fp rest =
  let n = String.length rest in
  let rec scan i =
    if i >= n then false
    else
      let c = rest.[i] in
      if
        (Char.equal c 's' || Char.equal c 'd' || Char.equal c 'h')
        && i + 1 < n
        && Char.is_digit rest.[i + 1]
        && (i = 0 || not (Char.is_alphanum rest.[i - 1]))
      then true
      else scan (i + 1)
  in
  scan 0

let is_vector_insn ~mnemonic ~rest =
  match x86_fp_suffix mnemonic with
  | Some (`Packed, _) -> true
  | Some (`Scalar, _) -> false
  | None ->
      packed_integer_mnemonic mnemonic || has_substr rest ~sub:"%ymm" || has_substr rest ~sub:"%zmm"
      || aarch64_arrangement_mnemonic mnemonic
      || aarch64_vector_operand rest

let is_scalar_fp_insn ~mnemonic ~rest =
  match x86_fp_suffix mnemonic with
  | Some (`Scalar, _) -> true
  | Some (`Packed, _) -> false
  | None ->
      (* aarch64: an [f...] instruction whose operands are scalar FP registers. [fcmp s0, s0] and
         [fcsel s1, s2, s3, ne] are what a scalarized lane loop leaves behind. An Apple-dialect
         [fmla.4s v0, v1, v2] is excluded by the mnemonic, the way its GAS spelling is excluded by
         the operands: neither count may claim it twice, and it is the packed one. *)
      String.is_prefix mnemonic ~prefix:"f"
      && (not (aarch64_arrangement_mnemonic mnemonic))
      && (not (aarch64_vector_operand rest))
      && aarch64_scalar_fp rest

let is_stack_ref ~rest =
  List.exists [ "%rsp"; "%rbp"; "%esp"; "%ebp"; "[sp"; "[x29"; "sp,"; "x29," ] ~f:(fun p ->
      has_substr rest ~sub:p)

let call_target ~mnemonic ~rest =
  if String.equal mnemonic "call" || String.equal mnemonic "callq" || String.equal mnemonic "bl"
  then
    let target = String.strip rest in
    let target =
      match String.substr_index target ~pattern:"@" with
      | Some i -> String.prefix target i
      | None -> target
    in
    let target = String.strip target in
    (* Mach-O prefixes every C symbol with an underscore, so a libm call Apple clang leaves in an
       accumulator loop is [callq _fmaxf] and would not match [libm_names]' [fmaxf]. Left
       unstripped, the macOS column reports ZERO libm calls and passes the central gh-ocannl-649
       claim while containing exactly the regression it exists to catch -- a green that checked
       nothing, which is the failure mode this whole test is shaped against. *)
    Some (String.chop_prefix_if_exists target ~prefix:"_")
  else None

(* The aarch64 condition codes, as an explicit list rather than a prefix test, because the mnemonics
   a prefix test would sweep in are the ones that must NOT count: [bl] and [blr] are calls, [br] is
   an indirect jump with no label operand. GNU as accepts both [b.ne] and [bne] and gcc emits the
   SECOND -- which is how this started out recognizing no loop at all on either aarch64 column,
   every anchor reported missing across three widths and two optimization levels (the "no loop
   carried the anchor" reading is a failure precisely so that a gap like this one surfaces as a red
   run rather than as a quiet column of absences). *)
let aarch64_conds =
  [
    "eq";
    "ne";
    "cs";
    "hs";
    "cc";
    "lo";
    "mi";
    "pl";
    "vs";
    "vc";
    "hi";
    "ls";
    "ge";
    "lt";
    "gt";
    "le";
    "al";
  ]

let is_cond mnemonic ~prefix =
  match String.chop_prefix mnemonic ~prefix with
  | Some c -> List.mem aarch64_conds c ~equal:String.equal
  | None -> false

let is_branch ~mnemonic =
  (* x86 conditional and unconditional jumps; aarch64 [b], [b<cc>]/[b.<cc>] and the
     compare-and-branch forms. *)
  String.is_prefix mnemonic ~prefix:"j"
  || String.equal mnemonic "b" || is_cond mnemonic ~prefix:"b." || is_cond mnemonic ~prefix:"b"
  || List.mem [ "cbz"; "cbnz"; "tbz"; "tbnz" ] mnemonic ~equal:String.equal

(* The branch's label operand is its last one ([b .L3], [cbnz w0, .L5], [jne .L2]). It is returned
   UNFILTERED and {!backward_edges} decides whether it names a label, rather than being screened
   here for a leading [.]: that prefix is an ELF-assembler convention, and Apple's assembler spells
   its compiler-local labels [LBB0_9] with no dot at all. Screening on it rejected every branch
   target on Darwin, which does not read as "this ISA is unsupported" -- it reads as a file with no
   loops in it, i.e. every anchor missing at once, on a platform AGENTS.md supports and CI builds.
   The label table is the honest filter: an operand that is not a label is not in it. *)
let branch_target ~rest =
  String.split_on_chars rest ~on:[ ','; ' '; '\t' ]
  |> List.filter ~f:(fun w -> not (String.is_empty w))
  |> List.last

(** {1 Source anchoring} *)

(** [anchor_lines ~source ~patterns] is the set of 1-based line numbers of [source] containing any
    of [patterns]. This is how a caller names the construct to census -- the accumulator combine,
    the FMA update -- without depending on which preprocessor arm the target selected: list a
    substring of every arm and the one that survived is the one whose lines the assembly mentions.
*)
let anchor_lines ~source ~patterns =
  String.split_lines source
  |> List.filter_mapi ~f:(fun i l ->
      if List.exists patterns ~f:(fun p -> has_substr l ~sub:p) then Some (i + 1) else None)
  |> Set.of_list (module Int)

(** [anchor_block_lines ~source ~after_pattern ~patterns] is the smallest brace-delimited source
    range containing every occurrence of [patterns] after [after_pattern]. The generated-kernel
    census uses a range rather than only the exact matching lines because clang may attribute an
    inlined conversion to another line of the construct. Braces in comments and literals are
    ignored, so prose or macro strings cannot silently widen the range.

    An absent marker, absent anchor, or unbalanced enclosing block returns the empty set. Callers
    already treat an empty anchor as a failed census rather than guessing at a wider range. *)
let anchor_block_lines ~source ~after_pattern ~patterns =
  let lines = String.split_lines source |> Array.of_list in
  let after =
    Array.find_mapi lines ~f:(fun i line ->
        if has_substr line ~sub:after_pattern then Some i else None)
  in
  match after with
  | None -> Set.empty (module Int)
  | Some after -> (
      let matches =
        Array.filter_mapi lines ~f:(fun i line ->
            if i > after && List.exists patterns ~f:(fun p -> has_substr line ~sub:p) then Some i
            else None)
        |> Array.to_list
      in
      match
        (List.min_elt matches ~compare:Int.compare, List.max_elt matches ~compare:Int.compare)
      with
      | Some first, Some last ->
          let stack = ref [] and intervals = ref [] and in_comment = ref false in
          Array.iteri lines ~f:(fun line_no line ->
              let rec scan i quote escaped =
                if i >= String.length line then ()
                else if !in_comment then
                  if
                    i + 1 < String.length line
                    && Char.equal line.[i] '*'
                    && Char.equal line.[i + 1] '/'
                  then (
                    in_comment := false;
                    scan (i + 2) quote false)
                  else scan (i + 1) quote false
                else
                  match quote with
                  | Some q ->
                      if escaped then scan (i + 1) quote false
                      else if Char.equal line.[i] '\\' then scan (i + 1) quote true
                      else if Char.equal line.[i] q then scan (i + 1) None false
                      else scan (i + 1) quote false
                  | None ->
                      if
                        i + 1 < String.length line
                        && Char.equal line.[i] '/'
                        && Char.equal line.[i + 1] '/'
                      then ()
                      else if
                        i + 1 < String.length line
                        && Char.equal line.[i] '/'
                        && Char.equal line.[i + 1] '*'
                      then (
                        in_comment := true;
                        scan (i + 2) None false)
                      else if Char.equal line.[i] '"' || Char.equal line.[i] '\'' then
                        scan (i + 1) (Some line.[i]) false
                      else if Char.equal line.[i] '{' then (
                        stack := line_no :: !stack;
                        scan (i + 1) None false)
                      else if Char.equal line.[i] '}' then (
                        (match !stack with
                        | opening :: rest ->
                            stack := rest;
                            intervals := (opening, line_no) :: !intervals
                        | [] -> ());
                        scan (i + 1) None false)
                      else scan (i + 1) None false
              in
              scan 0 None false);
          List.filter !intervals ~f:(fun (opening, closing) -> opening <= first && last <= closing)
          |> List.min_elt ~compare:(fun (o1, c1) (o2, c2) -> Int.compare (c1 - o1) (c2 - o2))
          |> Option.value_map
               ~default:(Set.empty (module Int))
               ~f:(fun (opening, closing) ->
                 List.range opening (closing + 1)
                 |> List.map ~f:(fun zero_based -> zero_based + 1)
                 |> Set.of_list (module Int))
      | _ -> Set.empty (module Int))

(** {1 The census itself} *)

(* The DWARF file number standing for [basename]: [.file N "path"] entries name every header the
   kernel included, and an inline function from [math.h] would otherwise contribute anchor lines
   that collide with the kernel's own numbering. *)
let source_file_numbers lines ~basename =
  List.filter_map lines ~f:(function
    | Some (Directive (".file" :: num :: rest)) -> (
        match Int.of_string_opt num with
        | None -> None
        | Some n ->
            (* Every QUOTED field is a candidate path, and any one of them ending in the basename
               settles it. Not [List.last_exn rest]: a [.file] directive has three shapes here --
               [.file N "path"], gcc's two-field [.file 0 "dir" "path"], and clang's [.file 0 "dir"
               "path" md5 0x<digest>] -- and taking the last token reads the DIGEST as the path on
               the third, leaving no file number recognized, hence no anchor line matched, hence
               [census] answering [None] for every row on any clang-based host. Filtering on the
               quote is what tells a path field from the trailing metadata; the checksum and the
               [md5] keyword are unquoted. (A directory containing a space splits into tokens that
               no longer parse as quoted, which costs nothing: the FILENAME field is the one the
               basename test needs, and it is a separate field in every shape.) *)
            let quoted =
              List.filter_map rest ~f:(fun tok ->
                  if String.is_prefix tok ~prefix:"\"" then
                    Some (String.strip tok ~drop:(fun c -> Char.equal c '"'))
                  else None)
            in
            if List.exists quoted ~f:(fun path -> String.is_suffix path ~suffix:basename) then
              Some n
            else None)
    | _ -> None)
  |> Set.of_list (module Int)

let classify_asm asm = String.split_lines asm |> List.map ~f:classify_line |> Array.of_list

(* Every backward branch is a loop edge; its body is the span from the target label to the branch.
   Nested loops both span an anchor, so the smallest span is the innermost. *)
let backward_edges lines =
  let label_at = Hashtbl.create (module String) in
  Array.iteri lines ~f:(fun i -> function
    | Some (Label l) -> Hashtbl.set label_at ~key:l ~data:i
    | _ -> ());
  Array.foldi lines ~init:[] ~f:(fun i acc -> function
    | Some (Insn { mnemonic; rest }) when is_branch ~mnemonic -> (
        match branch_target ~rest with
        | Some t -> (
            match Hashtbl.find label_at t with Some j when j < i -> (t, j, i) :: acc | _ -> acc)
        | None -> acc)
    | _ -> acc)

type parsed = {
  lines : line option array;
  edges : (string * int * int) list;  (** (label, body start, backward branch) *)
  files : Set.M(Int).t;  (** the DWARF file numbers naming the censused source *)
}
(** One assembly file, classified once.

    Everything below the line classification -- the loop edges, and the DWARF file numbers standing
    for the kernel's own source -- is a property of the FILE, not of the construct being censused,
    while a run asks about several dozen constructs per file. Parsing per question made the driver
    re-scan every one of a 40000-line listing's lines once per anchor, which was the larger half of
    [test/operations/cc_march_census]'s wall clock once gh-ocannl-752 raised the loop count. Hence
    the split: {!parse} once per compiled file, {!census_in} once per anchor. *)

let parse ~asm ~source_basename =
  let lines = classify_asm asm in
  {
    lines;
    edges = backward_edges lines;
    files = source_file_numbers (Array.to_list lines) ~basename:source_basename;
  }

(** [edge_count p] is how many loop edges the branch vocabulary recognized in [p] at all, which is
    what separates the two readings of a {!census_in} answering [None]: this construct was hoisted
    or folded away (some other loop was still found), or {!is_branch} does not know this ISA's
    spelling and no loop in the file was found. The second is a defect in this module and reports
    every anchor missing at once, so it is worth a claim of its own. *)
let edge_count p = List.length p.edges

let loop_edges ~asm = List.length (backward_edges (classify_asm asm))

(* The instruction profile of one span of classified lines. Shared by {!census_in}, which asks it
   about a loop body, and {!profile_all}, which asks it about a whole listing. *)
let count_range lines op_class ~from_ ~to_ =
  let libm = libm_names op_class in
  let counts =
    ref
      {
        instructions = 0;
        vector_ops = 0;
        scalar_fp_ops = 0;
        libm_calls = 0;
        stack_refs = 0;
        residual = 0;
      }
  in
  for k = from_ to to_ do
    match lines.(k) with
    | Some (Insn { mnemonic; rest }) ->
        let c = !counts in
        let vector = is_vector_insn ~mnemonic ~rest in
        let scalar_fp = is_scalar_fp_insn ~mnemonic ~rest in
        let libm_call =
          match call_target ~mnemonic ~rest with
          | Some t -> List.mem libm t ~equal:String.equal
          | None -> false
        in
        let stack_ref = is_stack_ref ~rest in
        let c = { c with instructions = c.instructions + 1 } in
        let c = if vector then { c with vector_ops = c.vector_ops + 1 } else c in
        let c = if scalar_fp then { c with scalar_fp_ops = c.scalar_fp_ops + 1 } else c in
        let c = if libm_call then { c with libm_calls = c.libm_calls + 1 } else c in
        let c = if stack_ref then { c with stack_refs = c.stack_refs + 1 } else c in
        let c =
          if vector || scalar_fp || libm_call || stack_ref then c
          else { c with residual = c.residual + 1 }
        in
        counts := c
    | _ -> ()
  done;
  !counts

(** [profile_all op_class ~asm] is the instruction profile of an ENTIRE listing, with no loop
    selection and no source anchoring.

    For a listing that is one small function, which is what a toolchain PROBE compiles: the question
    there is what the compiler made of one expression, and there is no enclosing loop to find. A
    census of a generated kernel wants {!census_in} instead -- counting a whole kernel would average
    every loop in it together. *)
let profile_all op_class ~asm =
  let lines = classify_asm asm in
  count_range lines op_class ~from_:0 ~to_:(Array.length lines - 1)

type selection = Innermost | Smallest_outer_anchor_carrier

(** [census_in p op_class ~anchor] is a loop of [p] whose body carries a source line in [anchor],
    with that loop's instruction profile. [Innermost], the default, selects the smallest span.
    [Smallest_outer_anchor_carrier] takes the smallest candidate that contains another
    anchor-carrying candidate. The latter reads a vector reduction's immediate accumulator loop
    rather than either its per-lane codec or a more distant compiler loop that happens to share
    source attribution. If no nested pair exists it falls back to the ordinary candidates.

    [None] means no loop carried the anchor -- the construct was hoisted, folded away, or never
    emitted. That is a finding, not an absence of one: report it as a failure. *)
let census_in ?(selection = Innermost) ({ lines; edges = loops; files } : parsed) op_class ~anchor =
  let carries (_, j, i) =
    let rec go k =
      if k > i then false
      else
        match lines.(k) with
        | Some (Directive (".loc" :: fno :: lno :: _)) -> (
            match (Int.of_string_opt fno, Int.of_string_opt lno) with
            | Some f, Some l when Set.mem files f && Set.mem anchor l -> true
            | _ -> go (k + 1))
        | _ -> go (k + 1)
    in
    go j
  in
  let candidates = List.filter loops ~f:carries in
  let candidates =
    match selection with
    | Innermost -> candidates
    | Smallest_outer_anchor_carrier ->
        let contains_another =
          List.filter candidates ~f:(fun (_, outer_j, outer_i) ->
              List.exists candidates ~f:(fun (_, j, i) ->
                  (outer_j < j || i < outer_i) && outer_j <= j && i <= outer_i))
        in
        if List.is_empty contains_another then candidates else contains_another
  in
  let best =
    List.min_elt candidates ~compare:(fun (_, j1, i1) (_, j2, i2) ->
        Int.compare (i1 - j1) (i2 - j2))
  in
  Option.map best ~f:(fun (label, j, i) ->
      { loop_label = label; span = i - j; counts = count_range lines op_class ~from_:j ~to_:i })

(** Prefer exact source anchors; only when none carries a loop, try the smallest enclosing construct
    after [after_pattern]. Trying the range eagerly can select unrelated compiler loops nested
    beside the accumulator (gh-ocannl-844). Shared by the real census and its pure controls. *)
let census_source_in ?(selection = Innermost) parsed op_class ~source ~patterns ~after_pattern =
  let read anchor = census_in ~selection parsed op_class ~anchor in
  match read (anchor_lines ~source ~patterns) with
  | Some _ as profile -> profile
  | None -> read (anchor_block_lines ~source ~patterns ~after_pattern)

(** {!census_in} over an assembly listing parsed for this one question. Convenient where a caller
    asks about one construct in one file; a caller asking about many should {!parse} once. *)
let census op_class ~asm ~source_basename ~anchor =
  census_in (parse ~asm ~source_basename) op_class ~anchor

(** A one-line profile, for the stderr table a census run prints. *)
let to_line
    {
      loop_label;
      span;
      counts = { instructions; vector_ops; scalar_fp_ops; libm_calls; stack_refs; residual };
    } =
  Printf.sprintf "%s span=%d insns=%d vector=%d scalar_fp=%d libm_calls=%d stack=%d residual=%d"
    loop_label span instructions vector_ops scalar_fp_ops libm_calls stack_refs residual
