(* Harvest the file-scope names a GPU routine must avoid (gh-ocannl-1338). The backend's runtime
   compiler checks a kernel definition and subsequent unqualified vendor type uses after the
   prelude. The uses catch C++ struct names a kernel definition can legally hide.

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

(* {2 Reading headers} *)

let is_ident_start c = Char.is_alpha c || Char.equal c '_'
let is_ident_char c = Char.is_alphanum c || Char.equal c '_'

(* [text] with comments, string and character literals blanked, newlines kept. *)
let blank_comments_and_literals text =
  let n = String.length text in
  let out = Bytes.of_string text in
  let blank i = if not (Char.equal (Bytes.get out i) '\n') then Bytes.set out i ' ' in
  let rec code i =
    if i >= n then ()
    else
      match text.[i] with
      | '/' when i + 1 < n && Char.equal text.[i + 1] '/' -> line_comment i
      | '/' when i + 1 < n && Char.equal text.[i + 1] '*' ->
          blank i;
          blank (i + 1);
          block_comment (i + 2)
      | ('"' | '\'') as q -> literal q (i + 1)
      | _ -> code (i + 1)
  and line_comment i =
    if i >= n || Char.equal text.[i] '\n' then code i
    else (
      blank i;
      line_comment (i + 1))
  and block_comment i =
    if i >= n then ()
    else if Char.equal text.[i] '*' && i + 1 < n && Char.equal text.[i + 1] '/' then (
      blank i;
      blank (i + 1);
      code (i + 2))
    else (
      blank i;
      block_comment (i + 1))
  and literal q i =
    if i >= n then ()
    else if Char.equal text.[i] '\\' && i + 1 < n then (
      blank i;
      blank (i + 1);
      literal q (i + 2))
    else if Char.equal text.[i] q then code (i + 1)
    else if Char.equal text.[i] '\n' then code i
    else (
      blank i;
      literal q (i + 1))
  in
  code 0;
  Bytes.to_string out

(* The identifier tokens of [text] (already blanked): a number literal is consumed whole, so its
   exponent or suffix letters are not read as a name. *)
let identifiers text =
  let n = String.length text in
  let acc = ref [] in
  let rec scan i =
    if i >= n then ()
    else if is_ident_start text.[i] then (
      let j = ref i in
      while !j < n && is_ident_char text.[!j] do
        Int.incr j
      done;
      acc := String.sub text ~pos:i ~len:(!j - i) :: !acc;
      scan !j)
    else if Char.is_digit text.[i] then (
      let j = ref i in
      while !j < n && (is_ident_char text.[!j] || Char.equal text.[!j] '.') do
        Int.incr j
      done;
      scan !j)
    else scan (i + 1)
  in
  scan 0;
  !acc

(* The [#include] / [#include_next] targets of [text] (raw, so the quotes survive): [`Quoted] for
   ["x"], [`Angled] for [<x>]. *)
let includes text =
  List.filter_map (String.split_lines text) ~f:(fun line ->
      let line = String.strip line in
      match String.chop_prefix line ~prefix:"#" with
      | None -> None
      | Some rest -> (
          let rest = String.lstrip rest in
          let rest =
            match String.chop_prefix rest ~prefix:"include_next" with
            | Some r -> Some r
            | None -> String.chop_prefix rest ~prefix:"include"
          in
          match Option.map rest ~f:String.strip with
          | Some r when String.length r >= 2 -> (
              let close = function '"' -> Some '"' | '<' -> Some '>' | _ -> None in
              match close r.[0] with
              | None -> None
              | Some c -> (
                  match String.index_from r 1 c with
                  | Some j ->
                      let target = String.sub r ~pos:1 ~len:(j - 1) in
                      Some ((if Char.equal c '"' then `Quoted else `Angled), target)
                  | None -> None))
          | _ -> None))

let is_file p = Stdlib.Sys.file_exists p && not (Stdlib.Sys.is_directory p)

(* Every header the [entries] reach through [dirs], with the identifier tokens of all of them. *)
let header_closure ~dirs ~entries ~header_identity =
  let seen = Hash_set.Poly.create () in
  let unresolved = Hash_set.create (module String) in
  let tokens = Hash_set.create (module String) in
  let resolve ~from (kind, target) =
    let local =
      match (kind, from) with `Quoted, Some dir -> [ Stdlib.Filename.concat dir target ] | _ -> []
    in
    List.find (local @ List.map dirs ~f:(fun d -> Stdlib.Filename.concat d target)) ~f:is_file
    |> Option.bind ~f:(fun path -> Option.map (header_identity path) ~f:(fun key -> (path, key)))
  in
  let rec visit ~from inc =
    match resolve ~from inc with
    | None -> Hash_set.add unresolved (snd inc)
    | Some (_, key) when Hash_set.mem seen key -> ()
    | Some (path, key) ->
        Hash_set.add seen key;
        let raw = Stdio.In_channel.read_all path in
        List.iter (identifiers (blank_comments_and_literals raw)) ~f:(Hash_set.add tokens);
        List.iter (includes raw) ~f:(visit ~from:(Some (Stdlib.Filename.dirname path)))
  in
  List.iter entries ~f:(visit ~from:None);
  (Hash_set.length seen, Hash_set.to_list unresolved, tokens)

let hip_sdk_header_identity dirs =
  let canonical path = try Some (Unix.realpath path) with Unix.Unix_error _ -> None in
  let subtrees =
    List.concat_map dirs ~f:(fun dir ->
        List.filter_map [ "hip"; "hsa"; "rocwmma" ] ~f:(fun subtree ->
            let path = Stdlib.Filename.concat dir subtree in
            if Stdlib.Sys.file_exists path && Stdlib.Sys.is_directory path then canonical path
            else None))
  in
  fun path ->
    Option.bind (canonical path) ~f:(fun file ->
        if
          List.exists subtrees ~f:(fun root ->
              String.is_prefix file ~prefix:(Stdlib.Filename.concat root ""))
        then
          (* The same symlink target can get different quoted includes from different
             directories. *)
          Option.map (canonical (Stdlib.Filename.dirname path)) ~f:(fun dir -> (file, dir))
        else None)

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
  (* A kernel can hide a C++ struct name without conflicting with its declaration. Keep the
     unqualified type uses on the candidate's line so diagnostics still identify that candidate. *)
  let type_uses =
    List.mapi Ir.Cuda_like_config.Vendor_vector_types.probe_types ~f:(fun i typ ->
        Printf.sprintf " using __ocannl_probe_type_%d = %s;" i typ)
    |> String.concat
  in
  let compiles = ref 0 in
  let measurements =
    List.map probes ~f:(fun (label, (probe : Ir.C_syntax.routine_name_probe)) ->
        let first_line = line_count probe.prelude + 1 in
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
