(* gh-ocannl-686: a routine whose name collides with a reserved word or a builtin.

   Routine names come from user-facing surfaces -- an [Assignments.Block_comment] label, the [~name]
   argument of [Context.compile] and of the autotune drop-ins (gh-ocannl-669) -- and used to reach
   the emitted [void <name>(] verbatim. [Block_comment ("asm", ...)] -- a natural abbreviation, and
   the one that found this -- therefore emitted [void asm(], which every C front end rejects, and
   the rejection surfaced as [Invalid_argument "... This is a bug in OCANNL. Please file an issue
   with the generated .c file ..."]: the actual cause, the name, appeared nowhere in the message.

   [C_syntax.kernel_ident] now normalizes the name once, at each backend's [compile] entry, against
   the same [ident_blacklist] the tensor-node names already avoid. What is pinned here:

   - The colliding routines COMPILE AND EXECUTE, with their values checked. A structural check on
   the emitted header would pass on a kernel that computes nothing; the point of the fix is that the
   routine works, not merely that it parses. - The emitted symbol is the mangled one, and the
   artifact names it -- the deterministic scheme is observable, not just its absence of failure. -
   The identity half: a name that is already a legal, non-reserved identifier is emitted unchanged.
   This is what keeps schedule-cache identities and every existing golden from churning, and it is
   the half a mangling scheme can silently break.

   The names probed are chosen to be reserved on EVERY backend: [asm] is a keyword for gcc, clang,
   nvrtc, hiprtc and the Metal front end alike (which is why it is in [C_syntax.c_keywords] even
   though C89/C99 reserve it only as an extension), and [arrayjit_threefry4x32] is a builtin all
   four backends' tables define. The C++-only keywords ([class], [new], ...) are deliberately NOT
   probed for mangling: they are legal identifiers in plain C, so the cc backend must and does leave
   them alone -- asserting a uniform mangling for them would be asserting something false. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Asgns = Ir.Assignments
module Idx = Ir.Indexing

let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

let approx_tol _ = 1e-4
let approx a b = Float.(abs (a - b) < approx_tol b)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name
let n = 16
let av = Array.init n ~f:(fun i -> Float.of_int (i % 5) *. 0.5)
let bv = Array.init n ~f:(fun i -> Float.of_int (i % 3) -. 1.)
let expected = Array.init n ~f:(fun i -> av.(i) +. bv.(i))

(* One compile+run of [a + b] under [~name], with fresh leaves each time so that no two cases share
   a comp (a comp's forward code is consumable once). Answers the values the routine produced. *)
let run_named ~label ~name =
  let a = TDSL.ndarray av ~label:[ label ^ "_a" ] ~output_dims:[ n ] () in
  let b = TDSL.ndarray bv ~label:[ label ^ "_b" ] ~output_dims:[ n ] () in
  let%op c = a + b in
  Train.set_materialized c.Tensor.value;
  let comp = Train.forward c in
  let ctx = Context.auto () in
  let ctx, routine = Context.compile ~name ctx comp Idx.Empty in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx c.Tensor.value, routine.Context.name)

(* [claim] for a name that must be mangled to [emitted]: the routine runs correctly, the artifact
   the compile wrote defines the mangled symbol, and the caller's own record still carries the name
   it asked for (mangling is a codegen concern, not a rename of the user's routine). *)
let colliding ~label ~name ~emitted =
  let values, routine_name = run_named ~label ~name in
  p_close
    (Printf.sprintf "routine named %S computes correct values" name)
    values expected ~tol:approx_tol;
  p
    (Printf.sprintf "routine named %S keeps its given name in the routine record" name)
    (String.equal routine_name name);
  Generated.assert_emits ~routine:emitted
    ~contains:("void " ^ emitted ^ "(")
    (Printf.sprintf "routine named %S emits the mangled symbol %S" name emitted)

(* --- 1. The reported case: a C keyword. --- *)
let () = colliding ~label:"gh686_kw" ~name:"asm" ~emitted:"asm__"

(* --- 2. A name that is a builtin every backend's table defines. Before the fix this both shadowed
   the definition (duplicate C symbol) and, since [filter_and_prepend_builtins] selects entries by
   searching the kernel for their key, was special-cased there to stop the name from dragging the
   definition in. Mangled, the routine's token is no longer the builtin's, so neither happens.
   --- *)
let () =
  colliding ~label:"gh686_bi" ~name:"arrayjit_threefry4x32" ~emitted:"arrayjit_threefry4x32__"

(* --- 2b. A name the prelude's unconditional includes declare. A routine DEFINITION takes file
   scope, so [void printf(...)] after [<stdio.h>] and [void exp(...)] after [<math.h>] are
   conflicting declarations -- errors whether or not the kernel calls either. These are not in
   [ident_blacklist] proper, because a tensor node named [exp] is a parameter or a local and legally
   shadows the declaration; putting them there would rename nodes (Tensor.unop's op_labels ARE these
   words) to prevent a collision that cannot happen. Hence a routine-only table. --- *)
let () = colliding ~label:"gh686_io" ~name:"printf" ~emitted:"printf__"
let () = colliding ~label:"gh686_mt" ~name:"exp" ~emitted:"exp__"

(* --- 3. Characters C does not admit in an identifier, and a leading digit. [Context.compile]'s
   [~name] bypasses [Assignments.get_name_exn]'s punctuation pass entirely, so these reached the
   emitted header as-is too. --- *)
let () = colliding ~label:"gh686_ch" ~name:"gh686 bad/name!" ~emitted:"gh686_bad_name_"
let () = colliding ~label:"gh686_dg" ~name:"686_leading_digit" ~emitted:"k_686_leading_digit"

(* --- 4. The identity half: an ordinary name is emitted verbatim. Without this, a mangling scheme
   that renamed everything would pass every claim above while churning every cache identity and
   every generated-source golden in the repository. --- *)
let () =
  let name = "gh686_ordinary_name" in
  let values, routine_name = run_named ~label:"gh686_ok" ~name in
  p_close "an ordinary routine name computes correct values" values expected ~tol:approx_tol;
  p "an ordinary routine name reaches the routine record unchanged" (String.equal routine_name name);
  Generated.assert_emits ~routine:name
    ~contains:("void " ^ name ^ "(")
    "an ordinary routine name is emitted unmangled"

(* --- 5. The Block_comment surface the issue was filed from: no [~name] at all, the name derived
   from the block comment. This is the exact shape that produced [void asm(]. --- *)
let () =
  let a = TDSL.ndarray av ~label:[ "gh686_bc_a" ] ~output_dims:[ n ] () in
  let b = TDSL.ndarray bv ~label:[ "gh686_bc_b" ] ~output_dims:[ n ] () in
  let%op c = a + b in
  Train.set_materialized c.Tensor.value;
  let comp = Tensor.consume_forward_code c in
  let comp = { comp with Asgns.asgns = Asgns.Block_comment ("asm", comp.Asgns.asgns) } in
  p "the block comment derives the reserved name"
    (String.equal (Asgns.get_name_exn comp.Asgns.asgns) "asm");
  let ctx = Context.auto () in
  let ctx, routine = Context.compile ctx comp Idx.Empty in
  let ctx = Context.run ctx routine in
  p_close "a Block_comment-named reserved routine computes correct values"
    (Context.get_values ctx c.Tensor.value)
    expected ~tol:approx_tol

(* --- 6. The other half of the same table: a tensor NODE whose label is a reserved word. Node names
   already avoid [ident_blacklist] by construction (a collision forces the disambiguated
   [n<id>_<label>] form), so this pins the table rather than the mechanism -- specifically the C++
   entries, which the GPU dialects need and plain C does not.

   [Tensor.unop ~op_label:"not"] mints exactly that: [not] is an alternative token for [!] in C++,
   so on MSL/CUDA/HIP a bare [not] is a parse error where in C it is an ordinary identifier. The
   claim is deliberately just the values: which spelling the node gets is backend-dependent (bare on
   cc, disambiguated on the C++ dialects) and so cannot appear in a backend-uniform golden -- the
   teeth here are that the kernel COMPILES AND RUNS under OCANNL_BACKEND=metal. *)
let () =
  let a = TDSL.ndarray av ~label:[ "gh686_not_a" ] ~output_dims:[ n ] () in
  let t = NTDSL.not a () in
  Train.set_materialized t.Tensor.value;
  let ctx = Context.auto () in
  let ctx = Train.forward_once ctx t in
  let expected_not = Array.map av ~f:(fun x -> if Float.(x = 0.) then 1. else 0.) in
  p_close "a node labeled by a C++ alternative token computes correct values"
    (Context.get_values ctx t.Tensor.value)
    expected_not ~tol:approx_tol

(* --- 7. gh-ocannl-707: beyond the C floor. The implementation's reserved namespace ([__x], [_X]),
   where the vendors put most of what their headers declare, is closed by rule rather than by table:
   such a name gains a [k] prefix on every backend. [main] is the C standard's own reserved entry
   point, which clang refuses as a [void] kernel and MSL reserves outright. --- *)
let () = colliding ~label:"gh707_ir" ~name:"__gh707_reserved" ~emitted:"k__gh707_reserved"
let () = colliding ~label:"gh707_mn" ~name:"main" ~emitted:"main__"

(* --- 8. Names a GPU backend's VENDOR headers declare at file scope: a global type, a macro, a
   dialect keyword, a C-linkage function -- each reaches a kernel definition [void <name>(] and
   collides with it ("redefinition of 'float4' as different kind of symbol"). Each backend's
   exemplars come from its own [vendor_routine_idents] classes, one per class, plus one name of the
   implementation namespace a vendor actually declares. The per-name outcome goes to stderr (not
   part of the golden); the golden carries one claim per property, evaluated on the GPU backends and
   skipped on the C ones, whose headers stop at the C floor sections 1-2b already probe. --- *)
let metal_exemplars =
  [
    ("float4", "float4__");
    ("packed_half2", "packed_half2__");
    ("M_PI_F", "M_PI_F__");
    ("METAL_FUNC", "METAL_FUNC__");
    ("as_type", "as_type__");
    ("metal", "metal__");
  ]

(* The CUDA and HIP tables are measured per toolchain (gh-ocannl-1338, [tools/vendor_idents.exe]),
   which is why the two lists differ: nvrtc accepts a kernel named [float4] (CUDA's vector types are
   structs, which a function of the same name hides) and rejects [rsqrtf] (a C-linkage math
   function), while hiprtc does the opposite. *)
let cuda_exemplars =
  [
    ("half2", "half2__");
    ("nv_bfloat16", "nv_bfloat16__");
    ("CUDARTAPI", "CUDARTAPI__");
    ("rsqrtf", "rsqrtf__");
    ("__hadd", "k__hadd");
  ]

let hip_exemplars =
  [
    ("half2", "half2__");
    ("float4", "float4__");
    ("HIPRT_INF_FP16", "HIPRT_INF_FP16__");
    ("FLT_MAX", "FLT_MAX__");
    ("__hadd", "k__hadd");
  ]

(* The reserved-namespace rule's boundary: a single underscore before a lowercase letter
   ([Assignments.get_name_exn]'s [_then_] joins) and an interior double underscore stay outside.
   (The CUDA and HIP tables' effect is pinned GPU-free in arrayjit/test/test_cuda_like_config.ml,
   through the instantiated configs, since only the GPU legs below can execute them.) *)
let () =
  p_all "names in the implementation's reserved namespace are recognized"
    [ "__half"; "__nv_bfloat16"; "__syncthreads"; "_Float16"; "__builtin_expf" ]
    ~f:Ir.C_syntax.implementation_reserved;
  p_none "names outside the implementation's reserved namespace are left alone"
    [ "_then_b"; "half_step"; "a__b"; "k__half"; "gh707" ]
    ~f:Ir.C_syntax.implementation_reserved

let vendor_exemplars =
  match backend_name with
  | "metal" -> metal_exemplars
  | "cuda" -> cuda_exemplars
  | "hip" -> hip_exemplars
  | _ -> []

(* Each exemplar compiled, run and read back once: [(values correct, given name kept, mangled symbol
   emitted)]. A compile that raises answers all-false, with the exception on stderr, so one failing
   name does not hide the others. *)
let vendor_results =
  List.mapi vendor_exemplars ~f:(fun i (name, emitted) ->
      let result =
        match run_named ~label:(Printf.sprintf "gh707_v%d" i) ~name with
        | values, routine_name ->
            let correct = Array.length values = n && Array.for_all2_exn values expected ~f:approx in
            let source = Generated.read emitted in
            ( correct,
              String.equal routine_name name,
              String.is_substring source ~substring:("void " ^ emitted ^ "(") )
        | exception exn ->
            Stdio.eprintf "vendor identifier %S as a routine name raised: %s\n%!" name
              (Exn.to_string exn);
            (false, false, false)
      in
      let ok, kept, emits = result in
      Stdio.eprintf
        "vendor identifier %S -> %S on %s: values %b, name kept %b, symbol emitted %b (not part of \
         the golden)\n\
         %!"
        name emitted backend_name ok kept emits;
      result)

let vendor_gate = List.mem [ "metal"; "cuda"; "hip" ] backend_name ~equal:String.equal

let () =
  gated_all ~when_:vendor_gate ~on:backend_name
    "every vendor-header identifier probed as a routine name computes correct values" vendor_results
    ~f:(fun (ok, _, _) -> ok);
  gated_all ~when_:vendor_gate ~on:backend_name
    "every vendor-header identifier probed keeps its given name in the routine record"
    vendor_results ~f:(fun (_, kept, _) -> kept);
  gated_all ~when_:vendor_gate ~on:backend_name
    "every vendor-header identifier probed emits its mangled symbol" vendor_results
    ~f:(fun (_, _, emits) -> emits)

(* --- 9. The backstop: what no table covers fails as the NAME's failure. The tables are a floor
   harvested at one SDK version, so a name a later header declares, or a platform extension outside
   the C standard, still reaches the compiler. [Compile_driver] then reads the compiler's report
   and, when it points at a routine's declaration, re-raises the failure as a [Utils.User_error]
   naming the routine -- never as the "bug in OCANNL" a codegen fault is.

   First the attribution itself, over the report shapes the toolchains produce: the quoted
   declaration line (gcc and clang print the source line; nvrtc and hiprtc too), and an error line
   quoting the name. The negatives are what must NOT blame the name: an error in the routine's body
   that gcc prefixes with its "In function" context line, and a report naming some other routine. *)
let () =
  let attributed report =
    Ir.C_syntax.routine_name_in_diagnostics ~routine_names:[ "other_fwd"; "random" ] report
  in
  let blames_random report = Option.equal String.equal (attributed report) (Some "random") in
  p_all "compiler reports pointing at the routine's declaration blame its name"
    [
      (* clang (cc on macOS), from a real run *)
      "k.c:6:6: error: conflicting types for 'random'\n\
      \    6 | void random(\n\
      \      |      ^\n\
       _stdlib.h:250:7: note: previous declaration is here";
      (* gcc's typographic quotes *)
      "k.c:6:6: error: conflicting types for \u{2018}random\u{2019}; have \u{2018}void(float \
       *)\u{2019}";
      (* the Metal front end: the error names nothing, the quoted line does *)
      "program_source:3:13: error: expected unqualified-id\n\
       kernel void random(device char* __pool0 [[buffer(0)]],\n\
      \            ^";
      (* nvrtc's double-quoted names *)
      "default_program(12): error: invalid redeclaration of type name \"random\"";
    ]
    ~f:blames_random;
  p_none "compiler reports about a routine's body or another routine do not blame the name"
    [
      "k.c: In function 'random':\nk.c:9:5: error: 'x' undeclared (first use in this function)";
      "k.c:9:5: error: use of undeclared identifier 'random_ptr'";
      (* a body error quoting a node or local that shares the routine's spelling *)
      "k.c:9:5: error: use of undeclared identifier 'random'";
      "default_program(14): error: identifier \"random\" is undefined";
      "k.c:3:6: error: conflicting types for 'other_fwd_v2'\n    3 | void other_fwd_v2(";
    ] ~f:(fun report -> Option.is_some (attributed report))

(* Then executed. [random] is a POSIX/BSD extension, not C: the macOS and glibc [<stdlib.h>] the cc
   prelude includes declare it ([long random(void)]), so [void random(float *...)] is a conflicting
   declaration there, outside every table on purpose. Mingw's headers do not declare it, and no GPU
   prelude does, so elsewhere the routine simply runs. Either way the claim below holds: the name
   never fails anonymously. The second claim is the one with teeth, gated to the hosts whose headers
   make it collide. *)
let () =
  let outcome =
    match run_named ~label:"gh707_rn" ~name:"random" with
    | values, _ -> `Ran values
    | exception Utils.User_error msg -> `Attributed msg
    | exception exn -> `Other (Exn.to_string exn)
  in
  (match outcome with
  | `Ran _ -> Stdio.eprintf "routine named \"random\" ran\n%!"
  | `Attributed msg -> Stdio.eprintf "routine named \"random\" failed as its name's: %s\n%!" msg
  | `Other msg -> Stdio.eprintf "routine named \"random\" failed UNATTRIBUTED: %s\n%!" msg);
  let names_random msg =
    String.is_substring msg ~substring:"\"random\""
    && String.is_substring msg ~substring:"gh-ocannl-707"
  in
  let label = "a routine named after a header name no table holds runs or fails as its name's" in
  (match outcome with
  | `Ran values -> p_close label values expected ~tol:approx_tol
  | `Attributed msg -> p label (names_random msg)
  | `Other _ -> p label false);
  let collides_here =
    List.mem [ "cc"; "multidev_cc" ] backend_name ~equal:String.equal
    && not (String.equal Stdlib.Sys.os_type "Win32")
  in
  gated ~aggregation:`Environment ~when_:collides_here
    ~on:(backend_name ^ " on " ^ Stdlib.Sys.os_type)
    "where the C headers declare random, the routine's compile failure names the routine"
    (match outcome with `Attributed msg -> names_random msg | `Ran _ | `Other _ -> false);
  (* The message blames the name, so the cc report it quotes must not go on to call itself a bug in
     OCANNL -- while still quoting what the compiler said. *)
  gated ~aggregation:`Environment ~when_:collides_here
    ~on:(backend_name ^ " on " ^ Stdlib.Sys.os_type)
    "where the C headers declare random, the name's failure quotes the compiler without the \
     codegen-bug claim"
    (match outcome with
    | `Attributed msg ->
        (not (String.is_substring msg ~substring:Ir.C_syntax.codegen_bug_claim_prefix))
        && String.is_substring msg ~substring:"Compiler output:"
        && String.is_substring msg ~substring:"random("
    | `Ran _ | `Other _ -> false)

(* --- 10. The two cases side by side, on the report the cc backend actually raises
   ([Cc_backend.compile_failure_detail]) and on every host: attributed to the routine's name, the
   compiler's output is quoted without the codegen-bug claim; a body error is not attributed and
   propagates unchanged, claim included. The second half is what keeps the first from passing
   vacuously: it pins that cc's report carries the claim the attribution drops. --- *)
let () =
  let attribute compiler_output =
    let detail =
      Context.Cc_backend.compile_failure_detail ~exit_code:1 ~source_note:"the generated .c file"
        ~command:"cc k.c -o k.so" ~compiler_output
    in
    let exn =
      Ir.Schedule_outcome.Cause_at
        ( Ir.Schedule_outcome.Backend_compile,
          Ir.Schedule_outcome.Backend_rejected
            {
              backend = "cc";
              stage = "compiler";
              severity = Ir.Schedule_outcome.Compiler_bug;
              detail;
            } )
    in
    let raised =
      try
        Ir.C_syntax.attribute_routine_name_failure
          ~routine_names:[ ("random", "random") ]
          exn (Stdlib.Printexc.get_callstack 1)
      with exn -> exn
    in
    (detail, raised)
  in
  let claims text = String.is_substring text ~substring:Ir.C_syntax.codegen_bug_claim_prefix in
  let collision = "k.c:6:6: error: conflicting types for 'random'\n    6 | void random(" in
  (match attribute collision with
  | _, Utils.User_error msg ->
      p "a name-attributed cc compile failure drops the codegen-bug claim" (not (claims msg));
      p "a name-attributed cc compile failure keeps the compiler's output"
        (String.is_substring msg ~substring:collision)
  | _, exn ->
      Stdio.eprintf "name-attributed cc report raised: %s\n%!" (Exn.to_string exn);
      p "a name-attributed cc compile failure drops the codegen-bug claim" false;
      p "a name-attributed cc compile failure keeps the compiler's output" false);
  let body_error =
    "k.c: In function 'random':\nk.c:9:5: error: 'x' undeclared (first use in this function)"
  in
  match attribute body_error with
  | detail, Ir.Schedule_outcome.Cause_at (_, cause) ->
      let propagated = Ir.Schedule_outcome.detail_of_cause cause in
      p "an unattributed cc compile failure keeps the codegen-bug claim" (claims propagated);
      p "an unattributed cc compile failure propagates its report unchanged"
        (String.equal propagated detail)
  | _, exn ->
      Stdio.eprintf "unattributed cc report raised: %s\n%!" (Exn.to_string exn);
      p "an unattributed cc compile failure keeps the codegen-bug claim" false;
      p "an unattributed cc compile failure propagates its report unchanged" false
