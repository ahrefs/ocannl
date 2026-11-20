(* Harvest the file-scope names a GPU routine must avoid (gh-ocannl-1338). The backend's runtime
   compiler checks a kernel definition and subsequent unqualified vendor type uses after the
   prelude. The uses catch C++ struct names a kernel definition can legally hide. The type
   candidates come from GPU builtin definitions and the selected dialect's scalar and vector type
   spellings, restricted to identifiers in the SDK headers or the routine-name table beyond the
   routine-name floor. Each prelude's compiler checks them as unqualified type aliases. This uses
   the GPU builtin tables together because some vendor types, such as CUDA's [float4], are spelled
   by another GPU dialect's emitter.

   Usage: vendor_idents.exe <cuda|hip|metal|cc> [--include-dir DIR]... [--batch N] [--names A,B,...]

   Run it on a box with the toolchain: [tools/test-run.sh run exec tools/vendor_idents.exe -- cuda].
   [--names] measures just the listed names, each alone, and prints the compiler's first error line
   for each that fails: a spot check of one entry, not a harvest.

   The harvest reads literal, single-line quoted and angled [#include] and [#include_next] lines,
   starting at the backend's prelude and implicit runtime headers. It resolves those spellings as
   paths and tokenizes header text without comments and literals. It does not preprocess macro
   include operands or evaluate conditionals. HIP follows only canonical paths inside the [hip],
   [hsa] and [rocwmma] subtrees of the include directories, including explicit [--include-dir]
   roots. The compiler still sees its ordinary headers and options. Candidate names are the tokens
   and current table entries, less the names another layer covers: the implementation's reserved
   namespace ([C_syntax.implementation_reserved], a rule) and the routine-name floor
   ([routine_floor_idents]: the node-name blacklist and the C library's names). The table's own
   entries are candidates too, so an entry that no longer collides is noticed. Each candidate is
   compiled as a kernel name after a prelude, in batches. Failing batches are settled by compiling
   the names their reports blame one at a time, or by bisection where the reports blame none. HIP
   tests the ordinary prelude first and retests survivors with the rocWMMA prelude. Each reported
   collision failed alone in at least one kernel configuration the backend uses.

   What it prints: the measurement, the names that fail although the backend's [kernel_ident] passes
   them through (missing from the table), the table entries that compile as kernel names
   (superfluous), and the generated table module between BEGIN and END lines. Exit status 0 when the
   table holds -- a generated table exactly, a hand-kept one (Metal's grammar, a deliberate
   superset) at least -- 1 when it does not, 2 when the measurement could not be taken.

   A point-in-time measurement of ONE toolchain on ONE box: a name an SDK adds later, or a header
   only another configuration includes, fails through [C_syntax.attribute_routine_name_failure]
   instead (gh-ocannl-707). Vendor calls live in the arms ([vendor_idents_arm_<backend>]), linked
   only where the vendor library is installed and type-checked everywhere else against the stub
   libraries of arrayjit/vendor_stubs (gh-ocannl-1688); everything else lives here, which every box
   links. *)

open Base
open Vendor_idents_headers

(* The vendor knowledge each backend's run needs beyond its arm. *)
type vendor = {
  backend : string;
  built : bool;
  probes : unit -> (string * Ir.C_syntax.routine_name_probe) list;
  include_dirs : unit -> string list;
  capabilities : unit -> (string * string) list;
  implicit_entries : string list;
      (** Headers the runtime compiler includes without the source asking (nvrtc and hiprtc
          pre-include their runtime headers' equivalents): followed like the prelude's own. *)
  generated : string option;
      (** The generated table module the measurement regenerates, when the table is generated. *)
}

let vendors =
  [
    {
      backend = "cuda";
      built = Vendor_idents_arm_cuda.built;
      probes = (fun () -> [ ("CUDA", Vendor_idents_arm_cuda.probe ()) ]);
      include_dirs = Vendor_idents_arm_cuda.include_dirs;
      capabilities = Vendor_idents_arm_cuda.capabilities;
      implicit_entries = [ "cuda_runtime.h" ];
      generated = Some "arrayjit/lib/vendor_idents_cuda.ml";
    };
    {
      backend = "hip";
      built = Vendor_idents_arm_hip.built;
      probes = Vendor_idents_arm_hip.probes;
      include_dirs = Vendor_idents_arm_hip.include_dirs;
      capabilities = Vendor_idents_arm_hip.capabilities;
      implicit_entries = [ "hip/hip_runtime.h" ];
      generated = Some "arrayjit/lib/vendor_idents_hip.ml";
    };
    {
      backend = "metal";
      built = Vendor_idents_arm_metal.built;
      probes = (fun () -> [ ("Metal", Vendor_idents_arm_metal.probe ()) ]);
      include_dirs = Vendor_idents_arm_metal.include_dirs;
      capabilities = Vendor_idents_arm_metal.capabilities;
      implicit_entries = [];
      (* Metal's table is the grammar in [Metal_backend] (gh-ocannl-707); this measures it. *)
      generated = None;
    };
  ]

(* {2 Measuring} *)

let line_count s = 1 + String.count s ~f:(Char.equal '\n')

(* The source lines a compiler report blames: nvrtc's [name.cu(LINE): error], clang's
   [file:LINE:COL: error] (hiprtc, Metal). *)
let blamed_lines report =
  let digits_before s j =
    let i = ref j in
    while !i > 0 && Char.is_digit s.[!i - 1] do
      Int.decr i
    done;
    if !i < j then Int.of_string_opt (String.sub s ~pos:!i ~len:(j - !i)) else None
  in
  List.filter_map (String.split_lines report) ~f:(fun line ->
      match String.substr_index line ~pattern:"): error" with
      | Some j -> digits_before line j
      | None -> (
          match String.substr_index line ~pattern:": error" with
          | None -> None
          | Some j ->
              (* [...:LINE:COL: error]: skip the column. *)
              let i = ref j in
              while !i > 0 && Char.is_digit line.[!i - 1] do
                Int.decr i
              done;
              if !i > 0 && !i < j && Char.equal line.[!i - 1] ':' then digits_before line (!i - 1)
              else None))

let () =
  let args = List.tl_exn (Array.to_list (Sys.get_argv ())) in
  let rec parse ~backend ~dirs ~batch ~names = function
    | "--include-dir" :: d :: rest -> parse ~backend ~dirs:(dirs @ [ d ]) ~batch ~names rest
    | "--batch" :: b :: rest -> parse ~backend ~dirs ~batch:(Int.of_string b) ~names rest
    | "--names" :: ns :: rest ->
        parse ~backend ~dirs ~batch ~names:(Some (String.split ns ~on:',')) rest
    | b :: rest when Option.is_none backend -> parse ~backend:(Some b) ~dirs ~batch ~names rest
    | [] -> (backend, dirs, batch, names)
    | arg :: _ ->
        Stdio.eprintf "vendor_idents: unexpected argument %s\n" arg;
        Stdlib.exit 2
  in
  let backend, extra_dirs, batch, spot_names =
    parse ~backend:None ~dirs:[] ~batch:256 ~names:None args
  in
  let backend = String.lowercase (Option.value backend ~default:"cc") in
  let vendor =
    match List.find vendors ~f:(fun v -> String.equal v.backend backend) with
    | Some v -> v
    | None ->
        Stdio.printf
          "vendor_idents: backend %s has no vendor headers beyond the C library (nothing to harvest)\n"
          backend;
        Stdlib.exit 0
  in
  if not vendor.built then (
    Stdio.printf
      "vendor_idents: the %s arm is not built on this box (its vendor library is absent)\n" backend;
    Stdlib.exit 2);
  let probes = vendor.probes () in
  let _, probe = List.hd_exn probes in
  Stdio.printf "vendor_idents: backend %s\n" backend;
  let capabilities = vendor.capabilities () in
  List.iter capabilities ~f:(fun (k, v) -> Stdio.printf "  %s: %s\n" k v);
  let dirs = extra_dirs @ vendor.include_dirs () in
  let entries =
    List.concat_map probes ~f:(fun (_, probe) ->
        List.map (includes probe.Ir.C_syntax.prelude) ~f:(fun (_, t) -> (`Angled, t)))
    @ List.map vendor.implicit_entries ~f:(fun t -> (`Angled, t))
  in
  let header_identity =
    if String.equal backend "hip" then hip_sdk_header_identity dirs
    else fun path -> Some (path, Stdlib.Filename.dirname path)
  in
  let n_headers, unresolved, tokens = header_closure ~dirs ~entries ~header_identity in
  Stdio.printf "headers: %d read from [%s]; %d include targets unresolved%s\n" n_headers
    (String.concat ~sep:"; " dirs) (List.length unresolved)
    (match List.take (List.sort unresolved ~compare:String.compare) 8 with
    | [] -> ""
    | some -> " (" ^ String.concat ~sep:", " some ^ ", ...)");
  let floor = Hash_set.of_list (module String) probe.routine_floor_idents in
  let covered name = Ir.C_syntax.implementation_reserved name || Hash_set.mem floor name in
  let table = List.dedup_and_sort probe.vendor_routine_idents ~compare:String.compare in
  let candidates =
    List.dedup_and_sort ~compare:String.compare (Hash_set.to_list tokens @ table)
    |> List.filter ~f:(fun name -> not (covered name))
  in
  Stdio.printf "candidates: %d (of %d header tokens and %d table entries)\n%!"
    (List.length candidates) (Hash_set.length tokens) (List.length table);
  let builtin_source (_, source, _) = source in
  let emitted =
    List.map
      (Builtins_cuda.builtins @ Builtins_hip.builtins @ Builtins_metal.builtins)
      ~f:builtin_source
    @ probe.type_spellings
    |> List.concat_map ~f:(fun text -> identifiers (blank_comments_and_literals text))
    |> Hash_set.of_list (module String)
  in
  let type_candidates = List.filter candidates ~f:(Hash_set.mem emitted) in
  let type_use typ = Printf.sprintf " using __ocannl_probe_type_%s = %s;" typ typ in
  let compiles = ref 0 in
  let measurements =
    List.map probes ~f:(fun (label, (probe : Ir.C_syntax.routine_name_probe)) ->
        let first_line = line_count probe.prelude + 1 in
        let types =
          List.filter type_candidates ~f:(fun typ ->
              Result.is_ok (probe.compile (probe.prelude ^ "\n" ^ type_use typ ^ "\n")))
        in
        Stdio.printf "%s: %d emitted type candidates checked, %d admitted%s\n%!" label
          (List.length type_candidates) (List.length types)
          (if List.is_empty types then "" else " (" ^ String.concat ~sep:", " types ^ ")");
        (* Keep each kernel's type uses on its diagnostic line. *)
        let type_uses = String.concat (List.map types ~f:type_use) in
        let source names =
          probe.prelude ^ "\n"
          ^ String.concat ~sep:"\n"
              (List.map names ~f:(fun name -> probe.kernel_definition name ^ type_uses))
          ^ "\n"
        in
        let compile names =
          Int.incr compiles;
          probe.compile (source names)
        in
        (* The controls: the prelude compiles alone, and with a fresh name; a keyword does not. *)
        (match (compile [], compile [ "ocannl_probe_control" ], compile [ "int" ]) with
        | Ok (), Ok (), Error _ -> ()
        | prelude, control, keyword ->
            let say what = function
              | Ok () -> what ^ " compiled"
              | Error r -> what ^ " failed: " ^ r
            in
            Stdio.printf "vendor_idents: %s controls failed, no measurement taken\n%s\n%s\n%s\n"
              label (say "the prelude" prelude)
              (say "a fresh kernel name" control)
              (say "[int]" keyword);
            Stdlib.exit 2);
        let failed_alone name = Result.is_error (compile [ name ]) in
        let rec settle names =
          match names with
          | [] -> []
          | [ name ] -> if failed_alone name then [ name ] else []
          | _ -> (
              match compile names with
              | Ok () -> []
              | Error report -> (
                  let arr = Array.of_list names in
                  let blamed =
                    List.filter_map (blamed_lines report) ~f:(fun line ->
                        let k = line - first_line in
                        if k >= 0 && k < Array.length arr then Some arr.(k) else None)
                    |> List.dedup_and_sort ~compare:String.compare
                  in
                  match List.filter blamed ~f:failed_alone with
                  | [] ->
                      let left, right = List.split_n names (List.length names / 2) in
                      settle left @ settle right
                  | failed ->
                      let failed_set = Set.of_list (module String) failed in
                      failed @ settle (List.filter names ~f:(fun n -> not (Set.mem failed_set n)))))
        in
        ( label,
          compile,
          fun candidates -> List.chunks_of candidates ~length:batch |> List.concat_map ~f:settle ))
  in
  Option.iter spot_names ~f:(fun names ->
      List.iter names ~f:(fun name ->
          match
            List.find_map measurements ~f:(fun (_, compile, _) ->
                match compile [ name ] with Ok () -> None | Error report -> Some report)
          with
          | None -> Stdio.printf "%s: compiles as a kernel name\n" name
          | Some report ->
              let first_error =
                List.find (String.split_lines report) ~f:(fun l ->
                    String.is_substring l ~substring:"error")
              in
              Stdio.printf "%s: fails: %s\n" name
                (Option.value first_error ~default:"(no error line)"));
      Stdlib.exit 0);
  let _, failed =
    List.fold measurements ~init:(candidates, []) ~f:(fun (remaining, failed) (label, _, measure) ->
        let before = !compiles in
        let collisions = measure remaining in
        if List.length measurements > 1 then
          Stdio.printf "measured %s: %d compiles; %d collisions\n%!" label (!compiles - before)
            (List.length collisions);
        let collisions_set = Hash_set.of_list (module String) collisions in
        let remaining =
          List.filter remaining ~f:(fun name -> not (Hash_set.mem collisions_set name))
        in
        (remaining, List.rev_append collisions failed))
  in
  let failed = List.sort failed ~compare:String.compare in
  Stdio.printf "measured: %d compiles; %d names fail as a kernel name\n" !compiles
    (List.length failed);
  let failed_set = Set.of_list (module String) failed in
  let missing = List.filter failed ~f:(fun name -> String.equal (probe.kernel_ident name) name) in
  let compile_fine, covered_elsewhere =
    List.filter table ~f:(fun name -> not (Set.mem failed_set name))
    |> List.partition_tf ~f:(Fn.non covered)
  in
  let show what names =
    Stdio.printf "%s: %d%s\n" what (List.length names)
      (if List.is_empty names then "" else "\n  " ^ String.concat ~sep:" " names)
  in
  show "fail, yet kernel_ident passes them through (missing from the table)" missing;
  show "in the table, yet compile as kernel names (superfluous)" compile_fine;
  show "in the table, and covered by the reserved-namespace rule or the floor as well"
    covered_elsewhere;
  Option.iter vendor.generated ~f:(fun path ->
      Stdio.printf "BEGIN %s\n" path;
      List.iter
        ~f:(fun line -> Stdio.printf "%s\n" line)
        [
          Printf.sprintf
            "(* GENERATED by tools/vendor_idents.exe for the %s backend (gh-ocannl-1338) -- do not \
             edit."
            backend;
          "   Regenerate on a box with the toolchain: [tools/test-run.sh run exec";
          Printf.sprintf
            "   tools/vendor_idents.exe -- %s], then replace this file with the module that run \
             prints"
            backend;
          "   between its BEGIN and END lines.";
          "";
          "   The names that failed to compile as a kernel name after the backend's prelude, out \
           of the";
          Printf.sprintf
            "   identifier tokens of the %d headers it reaches; names the reserved-namespace rule \
             or the"
            n_headers;
          "   routine-name floor already covers are not candidates. Measured with:";
        ];
      List.iter capabilities ~f:(fun (k, v) -> Stdio.printf "   - %s: %s\n" k v);
      Stdio.printf "*)\n\nlet names =\n  [\n";
      List.iter failed ~f:(fun name -> Stdio.printf "    %S;\n" name);
      Stdio.printf "  ]\nEND %s\n" path);
  if n_headers = 0 then
    Stdio.printf
      "note: no header was found under the include directories, so only the table's own entries \
       were measured: an entry can be shown superfluous, a missing name cannot be shown (pass \
       --include-dir)\n";
  if n_headers = 0 && Option.is_some vendor.generated then (
    Stdio.printf "verdict: no measurement of the %s headers was taken\n" backend;
    Stdlib.exit 2
    (* A generated table is the measurement itself, so it must be exact; a hand-kept one (Metal's
       grammar, a deliberate superset) must only miss nothing. *))
  else if List.is_empty missing && (Option.is_none vendor.generated || List.is_empty compile_fine)
  then (
    Stdio.printf "verdict: the %s table holds for this toolchain\n" backend;
    Stdlib.exit 0)
  else (
    Stdio.printf "verdict: the %s table differs from this toolchain's measurement\n" backend;
    Stdlib.exit 1)
