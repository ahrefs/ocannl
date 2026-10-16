(* Every arm of the cc storage bridges, executed (gh-ocannl-1145).

   The bridges between 16-bit storage and f32 compute registers ([Builtins_cc]'s
   [OCANNL_VEC_WIDEN_*] and [OCANNL_VEC_NARROW_*]) are preprocessor-guarded macros, and a golden run
   only ever compiled the arms its own host selects:

   - the width-specific x86 arms ([..._X4], [..._X8], [..._X16]: one [vpmovzxwd], [vcvtph2ps] or
   [vcvtps2ph] each) are named by the f32 lane count, which follows [cc_vector_bytes], which is
   auto-probed. CI's ubuntu runners probe 32, so only X8 ran there; macOS takes the portable arm; X4
   ran nowhere, and X16 only on an AVX-512 box by hand; - the fallback arms, taken without
   [__builtin_convertvector], were compiled by no test at all: gcc and clang both have the builtin,
   so the [#else] arms are preprocessed away everywhere. A defect in one of them (gh-ocannl-1102's
   round-1 finding) was checked only by hand-forcing [#if 0].

   So this test re-executes the two tests whose kernels go through the bridges --
   [narrow_storage_compute] (every 16-bit code through the widening, every fp16 rounding boundary
   through the narrowing, each against its serial twin) and [tile_mma_geometry] (a register tile's
   partial bf16/fp16 column, the gh-ocannl-1102 site) -- once per (vector width, bridge set), with
   [OCANNL_CC_VECTOR_BYTES] forced per child as [cc_march_census] does, at each of 16, 32 and 64
   bytes up to the widest vector the target holds ({!Cc_backend.auto_vector_bytes}; a wider cell is
   skipped whole: no user runs it, and it reaches no arm a narrower one misses). The [portable] set
   adds [-DOCANNL_FORCE_PORTABLE_BRIDGES] to the compiler command, which every fallback arm honours.
   Each child must exit 0 and print exactly its own committed golden: the forced run says what the
   host-width run says, so the two tests' claims are this test's claims too, and nothing about them
   is restated here.

   Which arm a kernel actually compiled into is not inferred from the settings. It is read from the
   retained kernel sources, preprocessed ([-dM -E]) under the kernel's own compiler command and
   {!Cc_backend.kernel_compiler_flags}: the definition each bridge name a kernel calls resolves to,
   through the renames, is the arm that ran. An arm the cc target cannot take (X16 without AVX-512,
   any x86 arm on aarch64) is a {!Verdict.gated} skip with [`Environment] aggregation, never a pass:
   [grep SKIPPED] over a run lists exactly the arms that host did not execute, and the forced
   sweep's skip coverage lists the ones no box did. The per-kernel table goes to stderr. *)

open Base
open Ocannl.Operation.DSL_modules
module Cc_backend = Context.Cc_backend

let widths = [ 16; 32; 64 ]

type bridges = Native | Portable

let bridges_name = function Native -> "native" | Portable -> "portable"
let portable_define = "-DOCANNL_FORCE_PORTABLE_BRIDGES"
let children = [ "narrow_storage_compute"; "tile_mma_geometry" ]
let base_command = Cc_backend.compiler_command ()

let command_for = function
  | Native -> base_command
  | Portable -> base_command ^ " " ^ portable_define

(* Native-Windows safe: no shell spellings ([mkdir -p], [rm -rf]), as in [cc_march_census]. *)
let rec mkdir_p dir =
  if not (Stdlib.Sys.file_exists dir) then (
    let parent = Stdlib.Filename.dirname dir in
    if not (String.equal parent dir) then mkdir_p parent;
    try Stdlib.Sys.mkdir dir 0o755 with Sys_error _ -> ())

let rec rm_rf path =
  if Stdlib.Sys.file_exists path then
    if Stdlib.Sys.is_directory path then (
      Array.iter (Stdlib.Sys.readdir path) ~f:(fun e -> rm_rf (Stdlib.Filename.concat path e));
      try Stdlib.Sys.rmdir path with Sys_error _ -> ())
    else try Stdlib.Sys.remove path with Sys_error _ -> ()

let read_file path = try Some (Stdio.In_channel.read_all path) with Sys_error _ -> None

(* {1 The children} *)

type cell = { bridges : bridges; width : int; test : string; dir : string }

(* This process's environment with every setting a child must not inherit replaced, as in
   [cc_march_census]: removed first, because [Unix.create_process_env] passes the array verbatim and
   a duplicate key resolves to whichever copy the child's getenv finds first.
   [OCANNL_BUILD_FILES_PREFIX] is removed and not replaced: the children read their kernels through
   [Test_utils.Generated], which refuses a configured prefix, and this test reads them from the
   default, executable-derived directory under the child's own working directory. *)
let child_env cell =
  let forced =
    [
      ("OCANNL_BACKEND", "cc");
      ("OCANNL_CC_VECTOR_BYTES", Int.to_string cell.width);
      ("OCANNL_CC_BACKEND_COMPILER_COMMAND", command_for cell.bridges);
    ]
  in
  let removed = "OCANNL_BUILD_FILES_PREFIX" :: List.map forced ~f:fst in
  let inherited =
    Unix.environment () |> Array.to_list
    |> List.filter ~f:(fun kv ->
        match String.lsplit2 kv ~on:'=' with
        | Some (k, _) -> not (List.mem removed k ~equal:String.equal)
        | None -> true)
  in
  Array.of_list (inherited @ List.map forced ~f:(fun (k, v) -> k ^ "=" ^ v))

let cwd = Stdlib.Sys.getcwd ()
let root = Stdlib.Filename.concat cwd "bridge_arm_coverage_runs"

(* Each child runs in a directory of its own below this one: its [build_files/<test>/] is then
   private to it, where the standalone run of the same test -- possibly concurrent, in this very
   directory -- keeps its own. [ocannl_config] is found in the ancestor directory. *)
let spawn cell =
  mkdir_p cell.dir;
  let exe = Stdlib.Filename.concat cwd (cell.test ^ ".exe") in
  let fd name =
    Unix.openfile (Stdlib.Filename.concat cell.dir name) [ O_WRONLY; O_CREAT; O_TRUNC ] 0o644
  in
  let out = fd "stdout.txt" and err = fd "stderr.txt" in
  (* [create_process_env] takes no working directory; the child inherits this process's at spawn. *)
  Stdlib.Sys.chdir cell.dir;
  let pid =
    Exn.protect
      ~f:(fun () -> Unix.create_process_env exe [| exe |] (child_env cell) Unix.stdin out err)
      ~finally:(fun () -> Stdlib.Sys.chdir cwd)
  in
  Unix.close out;
  Unix.close err;
  pid

(* At most this many children at once: each is a whole test with its own worker pool. *)
let parallelism = 3

(* Each cell paired with how its child exited, in [cells]' order. *)
let run_all cells =
  let status = Hashtbl.create (module Int) in
  let rec loop spawned pending running =
    match (pending, running) with
    | [], [] -> List.rev spawned
    | cell :: rest, _ when List.length running < parallelism ->
        let pid = spawn cell in
        loop ((cell, pid) :: spawned) rest (pid :: running)
    | _ ->
        (* The oldest running child, by its pid: native Windows has no [Unix.wait] for any child. *)
        let oldest = List.last_exn running in
        let pid, st = Unix.waitpid [] oldest in
        Hashtbl.set status ~key:pid ~data:st;
        loop spawned pending (List.filter running ~f:(fun p -> p <> pid))
  in
  List.map (loop [] cells []) ~f:(fun (cell, pid) -> (cell, Hashtbl.find_exn status pid))

(* {1 Reading which arm each kernel compiled into} *)

type arm =
  | Packed  (** a width-specific x86 arm: an [__builtin_ia32_*] conversion *)
  | Convertvector  (** the portable arm, through [__builtin_convertvector] *)
  | Fallback  (** neither: the per-lane (or per-row) arm the builtin's absence selects *)
  | Undefined  (** a name the kernel calls that its preprocessed source does not define *)

let arm_name = function
  | Packed -> "packed x86"
  | Convertvector -> "convertvector"
  | Fallback -> "fallback"
  | Undefined -> "UNDEFINED"

let is_ident_char c = Char.is_alphanum c || Char.equal c '_'

(* The kernel's code with its comments and preprocessor directives (continuations included) removed:
   what is left calls the bridges, where the builtin prelude only defines them. *)
let code_lines src =
  let buf = Buffer.create (String.length src) in
  let n = String.length src in
  let rec go i =
    if i < n then
      if i + 1 < n && Char.equal src.[i] '/' && Char.equal src.[i + 1] '*' then
        match String.substr_index src ~pos:(i + 2) ~pattern:"*/" with
        | Some j -> go (j + 2)
        | None -> ()
      else (
        Buffer.add_char buf src.[i];
        go (i + 1))
  in
  go 0;
  let rec drop_directives in_directive acc = function
    | [] -> List.rev acc
    | line :: rest ->
        let continued = String.is_suffix (String.rstrip line) ~suffix:"\\" in
        if in_directive || String.is_prefix (String.lstrip line) ~prefix:"#" then
          drop_directives continued acc rest
        else drop_directives false (line :: acc) rest
  in
  drop_directives false [] (String.split_lines (Buffer.contents buf))

(* The bridge names a kernel calls: [OCANNL_VEC_WIDEN_*] and [OCANNL_VEC_NARROW_*] followed by an
   argument list. *)
let bridge_calls src =
  let found = Hash_set.create (module String) in
  List.iter (code_lines src) ~f:(fun line ->
      List.iter [ "OCANNL_VEC_WIDEN_"; "OCANNL_VEC_NARROW_" ] ~f:(fun prefix ->
          List.iter (String.substr_index_all line ~may_overlap:false ~pattern:prefix) ~f:(fun i ->
              let j = ref (i + String.length prefix) in
              while !j < String.length line && is_ident_char line.[!j] do
                Int.incr j
              done;
              let rec next_nonblank k =
                if k < String.length line && Char.is_whitespace line.[k] then next_nonblank (k + 1)
                else k
              in
              let k = next_nonblank !j in
              if
                (i = 0 || not (is_ident_char line.[i - 1]))
                && k < String.length line
                && Char.equal line.[k] '('
              then Hash_set.add found (String.sub line ~pos:i ~len:(!j - i)))));
  Hash_set.to_list found |> List.sort ~compare:String.compare

(* [-dM -E] output: each macro's name, whether it takes arguments, and its body. *)
let parse_defines dump =
  List.filter_map (String.split_lines dump) ~f:(fun line ->
      match String.chop_prefix line ~prefix:"#define " with
      | None -> None
      | Some rest ->
          let len = String.length rest in
          let rec name_end i = if i < len && is_ident_char rest.[i] then name_end (i + 1) else i in
          let e = name_end 0 in
          let name = String.prefix rest e in
          if e < len && Char.equal rest.[e] '(' then
            match String.index_from rest e ')' with
            | Some close -> Some (name, (true, String.drop_prefix rest (close + 1)))
            | None -> None
          else Some (name, (false, String.drop_prefix rest e)))
  |> Map.of_alist_reduce (module String) ~f:(fun _ last -> last)

(* Follows object-like renames ([#define OCANNL_VEC_WIDEN_HALF_X8 OCANNL_VEC_WIDEN_HALF]) to the
   definition that is expanded, and says which arm that is. *)
let resolve defines name =
  let rec go name depth =
    match Map.find defines name with
    | None -> Undefined
    | Some (function_like, body) ->
        let body = String.strip body in
        if (not function_like) && String.for_all body ~f:is_ident_char && depth < 8 then
          go body (depth + 1)
        else if String.is_substring body ~substring:"__builtin_ia32_" then Packed
        else if String.is_substring body ~substring:"__builtin_convertvector" then Convertvector
        else Fallback
  in
  go name 0

(* The kernel's macro table as the kernel's own compile saw it: its compiler command (the child's,
   [-D] included) and the backend's kernel flags. [None] when the preprocessor fails. *)
let preprocess ~command src_path =
  let out = Stdlib.Filename.temp_file "ocannl_bridge_arm_" ".txt" in
  let log = Stdlib.Filename.temp_file "ocannl_bridge_arm_" ".log" in
  let cmd =
    Printf.sprintf "%s %s -dM -E %s > %s 2> %s" command
      (Cc_backend.kernel_compiler_flags ())
      (Stdlib.Filename.quote src_path) (Stdlib.Filename.quote out) (Stdlib.Filename.quote log)
  in
  let result =
    if Stdlib.Sys.command cmd = 0 then Option.map (read_file out) ~f:parse_defines
    else (
      Stdio.eprintf "bridge-arm: preprocessing failed: %s\n%s\n%!" cmd
        (Option.value (read_file log) ~default:"");
      None)
  in
  List.iter [ out; log ] ~f:(fun f -> try Stdlib.Sys.remove f with Sys_error _ -> ());
  result

type use = { cell : cell; kernel : string; bridge : string; arm : arm }

(* Every bridge call in every kernel a child left in its [build_files/<test>/], with its arm. *)
let uses_of cell =
  let kdir = Stdlib.Filename.concat (Stdlib.Filename.concat cell.dir "build_files") cell.test in
  let files =
    try Stdlib.Sys.readdir kdir |> Array.to_list |> List.sort ~compare:String.compare
    with Sys_error _ -> []
  in
  List.concat_map files ~f:(fun file ->
      if not (String.is_suffix file ~suffix:".c") then []
      else
        let path = Stdlib.Filename.concat kdir file in
        let kernel = String.chop_suffix_exn file ~suffix:".c" in
        match Option.map (read_file path) ~f:bridge_calls with
        | None | Some [] -> []
        | Some bridges ->
            let defines =
              Option.value
                (preprocess ~command:(command_for cell.bridges) path)
                ~default:(Map.empty (module String))
            in
            List.map bridges ~f:(fun bridge ->
                { cell; kernel; bridge; arm = resolve defines bridge }))

(* {1 The claims} *)

let equal_arm (a : arm) b = Poly.equal a b

(* The kernels whose bridge is the subject of [narrow_storage_compute]'s sweeps, with the bridge the
   renderer names for them at [lanes] f32 lanes -- asked of {!Ir.C_syntax}, which renders the call,
   rather than spelled here. *)
let sweeps ~lanes =
  let name f store_prec = Option.value_exn (f ~store_prec ~prec:Ir.Ops.single ~lanes) in
  [
    ( "nsc_wvec_bf16",
      name Ir.C_syntax.vec_widen_macro Ir.Ops.bfloat16,
      "the every-code bf16 widening sweep" );
    ( "nsc_wvec_half",
      name Ir.C_syntax.vec_widen_macro Ir.Ops.half,
      "the every-code fp16 widening sweep" );
    ( "nsc_nvec_half",
      name Ir.C_syntax.vec_narrow_macro Ir.Ops.half,
      "the fp16 rounding-boundary narrowing sweep" );
  ]

(* The widest vector the cc target holds, as [cc_vector_bytes] auto-selects it. A wider width is not
   a configuration anyone runs -- the compiler splits the vector and spills -- and it takes no arm a
   narrower one does not, so its cells are skipped rather than run. *)
let host_width = Cc_backend.auto_vector_bytes ()
let fits width = width <= host_width
let too_wide = Printf.sprintf "a cc target whose widest vector is %d bytes" host_width

let () =
  rm_rf root;
  let cells =
    List.concat_map [ Native; Portable ] ~f:(fun bridges ->
        List.concat_map widths ~f:(fun width ->
            List.map children ~f:(fun test ->
                let dir =
                  Stdlib.Filename.concat root
                    (Printf.sprintf "%s-w%d/%s" (bridges_name bridges) width test)
                in
                { bridges; width; test; dir })))
  in
  (* Only the widths the target holds are run; the others' claims are skipped below, and their
     kernel directories stay absent. *)
  let results = run_all (List.filter cells ~f:(fun c -> fits c.width)) in
  let clean cell =
    let status = List.Assoc.find results cell ~equal:Poly.equal in
    (* Line endings normalized on both sides: on native Windows a child's text-mode stdout ends its
       lines in CRLF, and the goldens are LF. *)
    let lf text = String.substr_replace_all text ~pattern:"\r\n" ~with_:"\n" in
    let stdout = Option.map ~f:lf (read_file (Stdlib.Filename.concat cell.dir "stdout.txt")) in
    let golden =
      Option.map ~f:lf (read_file (Stdlib.Filename.concat cwd (cell.test ^ ".expected")))
    in
    let ok =
      (match status with Some (Unix.WEXITED 0) -> true | _ -> false)
      && Option.is_some golden
      && Option.equal String.equal stdout golden
    in
    if not ok then
      Stdio.eprintf
        "bridge-arm: %s at cc_vector_bytes=%d, %s bridges, did not run clean; see %s\n%!" cell.test
        cell.width (bridges_name cell.bridges) cell.dir;
    ok
  in
  let uses = List.map cells ~f:(fun cell -> (cell, uses_of cell)) in
  (* The table, per child: each bridge it calls, the arm(s) it resolved to and in how many
     kernels. *)
  List.iter uses ~f:(fun (cell, us) ->
      let tally = Hashtbl.create (module String) in
      List.iter us ~f:(fun u ->
          Hashtbl.update tally
            (u.bridge ^ " -> " ^ arm_name u.arm)
            ~f:(function None -> 1 | Some n -> n + 1));
      List.iter
        (Hashtbl.to_alist tally |> List.sort ~compare:(fun (a, _) (b, _) -> String.compare a b))
        ~f:(fun (row, n) ->
          Stdio.eprintf "bridge-arm: %s w%d %s: %s (%d kernel%s)\n" (bridges_name cell.bridges)
            cell.width cell.test row n
            (if n = 1 then "" else "s")));
  Stdio.eprintf "%!";
  List.iter [ Native; Portable ] ~f:(fun bridges ->
      List.iter widths ~f:(fun width ->
          let here =
            List.filter cells ~f:(fun c -> Poly.equal c.bridges bridges && c.width = width)
          in
          let set = bridges_name bridges in
          let fit = fits width in
          let clean_of test =
            fit && List.find here ~f:(fun c -> String.equal c.test test) |> Option.exists ~f:clean
          in
          List.iter children ~f:(fun test ->
              Verdict.gated ~aggregation:`Environment ~when_:fit ~on:too_wide
                (Printf.sprintf
                   "%s bridges at cc_vector_bytes=%d: %s exits 0 printing its own golden" set width
                   test)
                (clean_of test));
          let calls =
            List.concat_map here ~f:(fun c -> List.Assoc.find_exn uses c ~equal:Poly.equal)
          in
          let nsc_calls =
            List.filter calls ~f:(fun u -> String.equal u.cell.test "narrow_storage_compute")
          in
          let lanes = width / 4 in
          let arm_of kernel bridge =
            List.find nsc_calls ~f:(fun u ->
                String.equal u.kernel kernel && String.equal u.bridge bridge)
            |> Option.map ~f:(fun u -> u.arm)
          in
          Verdict.gated_all ~aggregation:`Environment ~when_:fit ~on:too_wide
            (Printf.sprintf "%s bridges at cc_vector_bytes=%d: the sweeps call the %d-lane bridges"
               set width lanes) (sweeps ~lanes) ~f:(fun (kernel, bridge, _) ->
              Option.is_some (arm_of kernel bridge));
          Verdict.gated_all ~aggregation:`Environment ~when_:fit ~on:too_wide
            (Printf.sprintf
               "%s bridges at cc_vector_bytes=%d: every bridge a kernel calls is defined in it" set
               width) calls ~f:(fun u -> not (equal_arm u.arm Undefined));
          match bridges with
          | Portable ->
              Verdict.gated_all ~aggregation:`Environment ~when_:fit ~on:too_wide
                (Printf.sprintf
                   "portable bridges at cc_vector_bytes=%d: every bridge a kernel calls compiled \
                    to its fallback arm"
                   width) calls ~f:(fun u -> equal_arm u.arm Fallback)
          | Native ->
              let on_arm = if fit then "a cc target that does not select this arm" else too_wide in
              List.iter (sweeps ~lanes) ~f:(fun (kernel, bridge, sweep) ->
                  Verdict.gated ~aggregation:`Environment
                    ~when_:(Option.equal equal_arm (arm_of kernel bridge) (Some Packed))
                    ~on:on_arm
                    (Printf.sprintf
                       "native bridges at cc_vector_bytes=%d: the packed %s arm ran %s to its \
                        golden"
                       width bridge sweep)
                    (clean_of "narrow_storage_compute"));
              (* Per child, so the claim is about one run: whether that child's kernels took a
                 [convertvector] arm is a property of the target, read from what they compiled. *)
              List.iter children ~f:(fun test ->
                  Verdict.gated ~aggregation:`Environment
                    ~when_:
                      (List.exists calls ~f:(fun u ->
                           String.equal u.cell.test test && equal_arm u.arm Convertvector))
                    ~on:(if fit then "a cc target that selects no convertvector arm" else too_wide)
                    (Printf.sprintf
                       "native bridges at cc_vector_bytes=%d: the __builtin_convertvector arms ran \
                        %s to its golden"
                       width test)
                    (clean_of test))))
