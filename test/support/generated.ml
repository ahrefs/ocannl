(** Freshness-checked reads of the generated-kernel artifacts under [build_files/] (gh-ocannl-655).

    Tests that assert on emitted code read [build_files/<exe>/<routine>.<ext>]. The artifact is a
    side effect of a compile, not a value the test holds, and nothing in the path ties it to the
    compile it is supposed to describe:

    - [test/config/ocannl_config] sets [clean_up_build_files_on_startup=false], so an artifact
      outlives the run that produced it, indefinitely. A test asserting "the kernel contains
      [single_to_fp8]" keeps passing after the kernel stops being emitted at all -- folded to a
      constant, erased by precision inference, fissioned into a differently-named routine -- because
      it is reading the previous run's file. A structural check that has quietly stopped
      structurally checking is worth less than no check, because it is counted as coverage.
    - Within one run, two compiles under the same routine name overwrite each other's artifact. A
      loop that compiles several candidates under one name and credits each with what it finds on
      disk afterwards is crediting candidate N with candidate N-1's source whenever a compile skips
      emission (a schedule-cache hit, a reused artifact, a conditional write).

    So this module owns the read, and makes both cases fail rather than pass:

    - {!init} -- called once, before any compile -- empties this process's own [build_files/<exe>/]
      subdirectory. Every artifact present afterwards was written by this run, so existence *is*
      freshness and no clock is involved. What makes that safe is that the DEFAULT subdirectory is
      derived from the executable's own name and dune runs one process per executable, so nothing
      else writes there. The other two resolutions are not this process's to empty and get their own
      treatment: any configured [build_files_prefix], flat or named, is refused, because that
      directory is not this process's to empty (a second executable can be given the same prefix)
      and without deletion nothing distinguishes a re-emitted identical kernel from a stale one --
      deletion is the only write signal independent of timestamp granularity.
    - {!read} fails the run (through {!Verdict}, so the exit status carries it and no [dune promote]
      can bless it) when the artifact is missing. Silence must not be indistinguishable from a pass,
      which is what a [string option] returning [None] invited: some call sites recorded that as
      [false], others forgot it.
    - Reading one routine twice, with the artifact's contents having changed in between, is reported
      as an overwrite: within a run that is a same-named second routine, or an unarmed second
      candidate. {!arm} -- delete this routine's artifact, right before compiling the candidate that
      should produce it -- is how a loop says the overwrite is intended, and turns "was it really
      candidate N's kernel?" into a checked fact for each candidate in turn.

    A test whose leg the backend cannot evaluate (a GPU intrinsic on a CPU backend) must not reach
    {!read} at all: gate the leg and report it with [Verdict.skipped]. An absent artifact is a
    failure here, deliberately. *)

open Base

(* The run's backend name, lowercased, as passed to [init]. [None] until then, so that a read
   preceding initialization is caught rather than silently answered from a stale directory. *)
let backend : string option ref = ref None

(* Contents digest of each artifact path already read, so that a second read of a path whose
   contents changed is reported instead of silently attributed to the wrong compile. *)
let read_digests : (string, string) Hashtbl.t = Hashtbl.create (module String)

(* Whether wholesale cleanup of the artifact directory is this process's to do.

   Only the DEFAULT, executable-derived subdirectory is inherently process-private: dune runs one
   process per executable, so nothing else writes there. Every other resolution can be shared -- the
   flat layout by definition, and an explicit [build_files_prefix] because two executables can be
   given the same one -- and emptying a shared directory would delete a concurrently running test's
   kernel between its compile and its read. So the three cases get three behaviours rather than one:
   sweep, refuse, or establish freshness per routine through {!arm}. *)
type scoping = Private_to_this_exe | Shared_prefix of string | Flat

(* Read from the CONFIGURED VALUE, not from what the resolved directory happens to look like. The
   property is "nobody configured this directory, so it is mine by construction" -- and no
   comparison of the result can establish that: [build_files_prefix=<this exe's own basename>]
   resolves exactly where the default would, while remaining a prefix a second executable can be
   given too. The empty string is what [Utils.artifacts_subdir] itself treats as "unset". *)
let scoping () =
  match Utils.get_global_arg ~default:"" ~arg_name:"build_files_prefix" with
  | "" -> Private_to_this_exe
  | "." -> Flat
  | prefix -> Shared_prefix prefix

(* A refusal that returns is not a refusal. [Verdict.fail] only increments a counter, so a caller
   that reports an unusable artifact directory and carries on hands the rest of the test exactly the
   directory it just refused -- and {!arm}, which deletes, would then run against it. The exit
   status arriving eventually at teardown does not help: by then the deletions have happened. So
   refusing the DIRECTORY aborts the process, at the point of refusal, before any compile or cleanup
   can go through it. Exit 1 rather than an exception, matching what [Verdict]'s own teardown would
   produce: [Stdlib.exit] runs that teardown on the way out, so the failure is still reported
   normally. *)
let abort msg =
  Verdict.fail msg;
  Stdio.Out_channel.flush Stdio.stdout;
  Stdlib.exit 1

let uninitialized where =
  Verdict.fail
    (where ^ ": Test_utils.Generated.init ~backend_name must be called before any compile, so that "
   ^ "artifacts left by earlier runs cannot be mistaken for this run's")

(** The source extension, dot included, a backend writes its generated routines under: the name
    looked up exactly, case-insensitively and with a deprecated alias resolved, in
    [Config_reads.code_extension], which [marker_backend_vocabulary] holds equal to each backend's
    own [source_extension] fact. Raises [Invalid_argument] on a name that is no backend's: a
    misspelt backend must not read artifacts under some other backend's extension (the substring
    test this replaced read [xcuda] as CUDA and anything it did not recognize as C). *)
let extension_of_backend backend_name =
  match Config_reads.code_extension (Config_reads.canonical_backend backend_name) with
  | Some ext -> "." ^ ext
  | None ->
      invalid_arg
        (Printf.sprintf
           "Test_utils.Generated.extension_of_backend: %S is not a backend name, so it names no \
            source extension"
           backend_name)

let extension () =
  match !backend with
  | Some b -> extension_of_backend b
  | None ->
      uninitialized "Generated.extension";
      ".c"

(** [path ?ext routine] is where the backend writes [routine]'s generated source. [?ext] overrides
    the backend-derived extension, for a test pinning one backend's artifact specifically. *)
let path ?ext routine =
  let ext = match ext with Some e -> e | None -> extension () in
  Utils.build_file (routine ^ ext)

(* Deletion IS the freshness guarantee here, so a delete that does not happen cannot be swallowed: a
   surviving file is one a later {!read} would accept as this run's, which is the whole failure this
   module exists to prevent. Absence is therefore verified rather than inferred from the call not
   raising -- a Windows sharing violation, a read-only directory, or a file another process holds
   open all leave the path in place. Deleting something that was not there is not a failure, which
   is why the verdict is on [file_exists] afterwards rather than on the exception. *)
let remove_or_fail ~context p =
  let err =
    try
      Stdlib.Sys.remove p;
      None
    with Stdlib.Sys_error msg -> Some msg
  in
  if Stdlib.Sys.file_exists p then
    Verdict.fail
      (Printf.sprintf
         "%s: could not delete %s%s -- a stale artifact left in place would be read as this run's"
         context p
         (match err with Some msg -> " (" ^ msg ^ ")" | None -> ""))

(* [Stdlib.Sys.is_directory] follows symlinks, so a symlinked [build_files/] or [build_files/<exe>/]
   is accepted as an ordinary directory -- and a sweep would then [readdir] the LINK TARGET and
   unlink real files outside the artifact tree. [arrayjit/lib/utils.ml]'s startup cleanup uses
   [Unix.lstat] for exactly this reason; the same question has to be asked here, of the destructive
   operation this module performs. Unlinking an individual symlinked ENTRY is not the hazard (that
   removes the link, not its target, which is what the startup cleanup relies on too) -- the
   containing directory is. *)

(** [init ~backend_name] empties this executable's [build_files/] subdirectory, so that every
    artifact found later was emitted by this run. Call it once at the top of the test, before any
    compile -- [Verdict.fail]s if a read happens without it.

    [~backend_name] selects the source extension ({!extension_of_backend}; a name that is no
    backend's aborts the run here); tests already derive it as
    [String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")]. It is passed in
    rather than read here to keep the dependency at [arrayjit.utils]: this module resolves paths, it
    does not choose backends. *)
let is_symlink p =
  match Unix.lstat p with
  | { Unix.st_kind = Unix.S_LNK; _ } -> true
  | _ -> false
  | exception Unix.Unix_error _ -> false

let init ~backend_name =
  (* An unknown backend is refused here, before any compile, rather than at the first read. *)
  (match extension_of_backend backend_name with
  | (_ : string) -> ()
  | exception Invalid_argument msg -> abort ("Generated.init: " ^ msg));
  backend := Some (String.lowercase backend_name);
  Hashtbl.clear read_digests;
  (* The ROOT is inspected before anything that could create through it. [Utils.build_files_dir]
     creates the per-executable subdirectory when it is missing, and [Sys.is_directory] follows
     links, so calling it first would materialize that subdirectory inside a symlinked root's target
     -- an external write performed by the very check whose purpose is to refuse one. *)
  let root = "build_files" in
  if is_symlink root then
    abort
      (Printf.sprintf
         "Generated.init: the artifact root %s is a symbolic link; refusing to use it, since \
          resolving through it would create and delete files outside the artifact tree"
         root)
  else
    let dir = Utils.build_files_dir () in
    if is_symlink dir then
      (* Refused rather than skipped, and aborted rather than merely recorded: following the link
         would delete files that are not this test's -- strictly worse than any stale artifact --
         and a refusal the run continues past would do exactly that at the next {!arm}. *)
      abort
        (Printf.sprintf
           "Generated.init: %s is a symbolic link; refusing to use it, since following it would \
            delete files outside the artifact tree"
           dir)
    else
      match scoping () with
      | Private_to_this_exe ->
          (* Derived from this executable's own name, so nothing else writes here: emptying it makes
             existence mean "written by this run" for every routine at once.

             Entries are classified by [lstat], never by [Sys.is_directory]: that follows links and
             RAISES on a dangling one, so a "treat what we cannot stat as a directory" fallback
             would leave exactly the entries most worth removing. A surviving link is not merely
             stale -- the backend's next write to that routine name follows it and lands outside the
             artifact tree. Unlinking a symlink removes the link and not its target, which is what
             makes this safe and is the same distinction arrayjit's own startup cleanup draws. *)
          Array.iter (Stdlib.Sys.readdir dir) ~f:(fun entry ->
              let p = Stdlib.Filename.concat dir entry in
              match Unix.lstat p with
              | { Unix.st_kind = Unix.S_DIR; _ } -> ()
              | _ -> remove_or_fail ~context:"Generated.init" p
              | exception Unix.Unix_error _ -> remove_or_fail ~context:"Generated.init" p)
      | Flat | Shared_prefix _ ->
          (* Any configured prefix, flat or named, is a directory this process does not own: another
             executable can be given the same one, and dune schedules tests concurrently. Deleting
             there could take a running test's kernel out from under it, and NOT deleting leaves no
             way to tell this run's artifact from a previous run's -- deletion is the only write
             signal that does not depend on timestamp granularity, and two successive runs of a
             deterministic compile produce byte-identical files. Both halves of that were tried and
             both have corners that cannot be closed (gh-ocannl-655 review rounds 5-7), so the
             configuration is refused instead of being supported approximately.

             The remedy is to let the prefix default: it is derived from the executable's own name,
             which is what makes the directory this process's to empty. *)
          abort
            "Generated.init: build_files_prefix is set, so the artifact directory is not this \
             executable's own and provenance cannot be established in it -- another executable may \
             be configured with the same prefix, deletion there is unsafe, and without deletion a \
             re-emitted identical kernel is indistinguishable from a stale one. Unset \
             build_files_prefix for tests that assert on generated code."

(** [arm ?ext routine] deletes [routine]'s artifact, so that the next {!read} of it sees only what
    the *next* compile emits. Use it in a loop that compiles several candidates under one routine
    name: without it, the second read of a changed artifact is reported as an unattributed
    overwrite. *)
let arm ?ext routine =
  let p = path ?ext routine in
  Hashtbl.remove read_digests p;
  remove_or_fail ~context:("Generated.arm " ^ routine) p

(** [read ?ext routine] is the generated source [routine]'s compile emitted during this run.

    Fails the run, and answers [""], when the artifact is missing: the kernel was not emitted, the
    routine is named something else now, or the test forgot
    [Utils.settings.output_debug_files_in_build_directory <- true]. Answering rather than raising
    keeps the run's remaining checks reporting; the failure is already recorded, so the exit status
    holds whatever the caller concludes from the empty source. *)
let read ?ext routine =
  if Option.is_none !backend then (
    uninitialized ("Generated.read " ^ routine);
    "")
  else
    let p = path ?ext routine in
    if not (Stdlib.Sys.file_exists p) then (
      Verdict.fail
        (Printf.sprintf
           "no generated source for routine %s: %s was not emitted by this run (kernel folded \
            away, renamed, fissioned, or debug artifacts are off)"
           routine p);
      "")
    else
      let src = Stdio.In_channel.read_all p in
      let digest = Stdlib.Digest.string src in
      (match Hashtbl.find read_digests p with
      | Some previous when not (String.equal previous digest) ->
          Verdict.fail
            (Printf.sprintf
               "generated source for routine %s changed between reads: %s was overwritten by \
                another compile under the same name, so the earlier reading described a different \
                kernel (name the routines apart, or Generated.arm before each compile)"
               routine p)
      | Some _ | None -> ());
      Hashtbl.set read_digests ~key:p ~data:digest;
      src

(** [kept_libraries ?owner routine] lists the paths of the libraries kept for [routine] under
    [output_dlls_in_build_directory] (gh-ocannl-1247) or [output_debug_files_in_build_directory]
    (gh-ocannl-1335), in owner then compile order: those of this process ([`This_process], the
    default), of the process with OS id [p] ([`Pid p]), or of every process ([`Any]). Names are read
    with the backend's own {!Utils.parse_kept_library_name}, and the kernel name they carry is
    {!Utils.library_kernel_name}'s, so neither a renaming nor a routine name too long to carry whole
    can leave this listing looking for the wrong ones. A child process of this executable writes
    into the same directory, which is how a test sees another process's libraries; it must not call
    {!init} itself, or it would sweep the parent's.

    Like {!read}, it answers from this run only: {!init} emptied the directory first. An empty list
    is an answer here, not a failure, so a caller states the count it expects. *)
let kept_libraries ?(owner = `This_process) routine =
  if Option.is_none !backend then (
    uninitialized ("Generated.kept_libraries " ^ routine);
    [])
  else
    (* This process is its OS id AND its nonce: a gone process, or one in another PID namespace, can
       have reported the same id. *)
    let wanted { Utils.owner_pid; owner_nonce; _ } =
      match owner with
      | `This_process ->
          owner_pid = Utils.os_process_id () && String.equal owner_nonce (Utils.process_nonce ())
      | `Pid p -> owner_pid = p
      | `Any -> true
    in
    let dir = Utils.build_files_dir () in
    Stdlib.Sys.readdir dir |> Array.to_list
    |> List.filter_map ~f:(fun entry ->
        match Utils.parse_kept_library_name entry with
        | Some ({ kernel_name; owner_pid; library_run_id; _ } as kept)
          when String.equal kernel_name (Utils.library_kernel_name routine) && wanted kept ->
            Some ((owner_pid, library_run_id), Stdlib.Filename.concat dir entry)
        | Some _ | None -> None)
    |> List.sort ~compare:(fun ((p1, r1), _) ((p2, r2), _) ->
        match Int.compare p1 p2 with 0 -> Int.compare r1 r2 | c -> c)
    |> List.map ~f:snd

(** [plant_kept_library ?run_id ~pid routine] writes a placeholder under the name of a library the
    process [pid] kept for [routine]'s compile [run_id] (default 1), and answers its path: a fixture
    for what a startup cleanup does with a library it finds, without the test naming the artifact
    directory itself. *)
let plant_kept_library ?(run_id = 1) ~pid routine =
  if Option.is_none !backend then (
    uninitialized ("Generated.plant_kept_library " ^ routine);
    "")
  else
    let path =
      Stdlib.Filename.concat (Utils.build_files_dir ())
        (Utils.kept_library_name ~pid ~nonce:(Utils.process_nonce ()) ~run_id routine)
    in
    Stdio.Out_channel.write_all path ~data:"planted";
    path

(** [assert_emits ~routine ~contains claim] records [claim] as holding exactly when [routine]'s
    generated source contains [contains] -- the common case, as an assertion rather than a [match]
    whose [None] arm has to be remembered. *)
let assert_emits ?ext ~routine ~contains claim =
  Verdict.p claim (String.is_substring (read ?ext routine) ~substring:contains)

(** [assert_omits ~routine ~contains claim] is {!assert_emits} for a substring that must NOT appear.
    Note that it still requires the artifact to exist: "the kernel does not contain [x]" is not a
    fact a missing kernel establishes. *)
let assert_omits ?ext ~routine ~contains claim =
  Verdict.p claim (not (String.is_substring (read ?ext routine) ~substring:contains))

(** [node_ident ~facts label] is the identifier a tensor node whose code name is [label] takes in
    this run's emitted kernels: [label] behind the backend dialect's node-identifier prefix, as the
    backend's [facts] state it ([node_ident_prefix] of {!Ir.Backend_intf.backend_facts},
    gh-ocannl-1338, gh-ocannl-1723) -- [Context.backend_facts_of_name backend_name] for the run's
    backend. A claim pinning a node's identifier in emitted text spells it through here, so it
    states the same identifier on every backend. *)
let node_ident ~(facts : Ir.Backend_intf.backend_facts) label =
  match !backend with
  | Some _ -> facts.node_ident_prefix ^ label
  | None ->
      uninitialized "Generated.node_ident";
      label

(* An identifier spelled with a number a compile mints afresh: a loop symbol ([i<digits>],
   [Indexing.symbol_ident]), a node identifier spelling its id ([n<digits>], optionally followed by
   [_<label>]: [Tnode.styled_ident] for an unlabeled node or one whose label repeats in the kernel),
   likewise a schedule-minted tile in the schedule's namespace ([tile__n<digits>_<label>]), or a
   scope-local scalar spelling its scope id ([v<digits>_<ident>]), or a pool-chunked [Grid] loop's
   chunk index and bounds, which derive from its loop symbol ([Ir.C_syntax.grid_chunk_derivations],
   gh-ocannl-1877). A type or builtin whose name ends in a digit ([uint3], [float4], [__half2]) is
   none of these shapes and stays as written. *)
let tile_ident_prefix = Ir.Tnode.ident_prefix Ir.Schedule.tile_namespace

let rec is_minted_ident ~(facts : Ir.Backend_intf.backend_facts) ident =
  let numbered ~prefix =
    match String.chop_prefix ident ~prefix with
    | None -> None
    | Some rest ->
        let num = Option.value_map (String.lsplit2 rest ~on:'_') ~default:rest ~f:fst in
        Option.some_if ((not (String.is_empty num)) && String.for_all num ~f:Char.is_digit) rest
  in
  (match numbered ~prefix:"i" with
    | Some rest -> String.for_all rest ~f:Char.is_digit
    | None -> false)
  || (let chunk, lo, hi = Ir.C_syntax.grid_chunk_derivations in
      List.exists [ chunk; lo; hi ] ~f:(fun (prefix, suffix) ->
          match Option.bind (String.chop_prefix ident ~prefix) ~f:(String.chop_suffix ~suffix) with
          | Some base ->
              String.is_prefix base ~prefix:"i"
              && (not (String.equal base ident))
              && is_minted_ident ~facts base
          | None -> false))
  || Option.is_some (numbered ~prefix:"n")
  (* A schedule-minted tile or accumulator, spelling its id in the schedule's own namespace
     ([tile__n24_tile_relu]): a process-wide counter, so a second compile of one program numbers its
     tiles afresh (gh-ocannl-1877). *)
  || Option.is_some (numbered ~prefix:tile_ident_prefix)
  (* A node identifier behind a GPU dialect's prefix ([t_n12_x], gh-ocannl-1338). *)
  || (match !backend with
    | Some _ ->
        Option.is_some (numbered ~prefix:(facts.node_ident_prefix ^ "n"))
        || Option.is_some (numbered ~prefix:(facts.node_ident_prefix ^ tile_ident_prefix))
    | None -> false)
  || Ir.Low_level.in_scope_local_namespace ident

(** [normalize_minted ~facts text] is a generated source with every identifier a compile spells with
    a freshly minted number -- loop symbols [i<digits>], node identifiers [n<digits>[_<label>]]
    (behind the backend's {!node_ident} prefix where it has one), scope-local scalars
    [v<digits>_<ident>], a pool-chunked loop's [i<digits>_chunk] and bounds -- renamed [@0], [@1],
    ... by first appearance, and every run of whitespace collapsed to one space (a longer identifier
    moves the pretty-printer's line breaks). Two compiles of one program differ in nothing else --
    not even across fresh builds of it, which mint new node ids (gh-ocannl-1473) -- so their
    normalized sources are equal, and a difference left is one in the code. Number literals are
    copied whole, so a hex float's letters are not read as an identifier. *)
let normalize_minted ~facts text =
  let names = Hashtbl.create (module String) in
  let out = Buffer.create (String.length text) in
  let n = String.length text in
  let is_ident_char c = Char.is_alphanum c || Char.equal c '_' in
  let rec scan i =
    if i < n then
      let c = text.[i] in
      if Char.is_whitespace c then (
        let j = ref i in
        while !j < n && Char.is_whitespace text.[!j] do
          Int.incr j
        done;
        Buffer.add_char out ' ';
        scan !j)
      else if Char.is_alpha c || Char.equal c '_' then (
        let j = ref i in
        while !j < n && is_ident_char text.[!j] do
          Int.incr j
        done;
        let ident = String.sub text ~pos:i ~len:(!j - i) in
        if is_minted_ident ~facts ident then
          let fresh = Hashtbl.length names in
          Buffer.add_string out
            ("@" ^ Int.to_string (Hashtbl.find_or_add names ident ~default:(fun () -> fresh)))
        else Buffer.add_string out ident;
        scan !j)
      else if Char.is_digit c then (
        let j = ref i in
        while !j < n && (is_ident_char text.[!j] || Char.equal text.[!j] '.') do
          Int.incr j
        done;
        Buffer.add_string out (String.sub text ~pos:i ~len:(!j - i));
        scan !j)
      else (
        Buffer.add_char out c;
        scan (i + 1))
  in
  scan 0;
  Buffer.contents out

(** [brace_intervals lines] is every matched brace pair of a C source given as its lines, as
    [(opening, closing)] 0-based line indices, ordered by their closing brace. Braces in comments
    and in string or character literals are not braces, so prose or a macro string cannot open or
    close a construct; an unmatched closing brace is dropped. *)
let brace_intervals lines =
  let stack = ref [] and intervals = ref [] and in_comment = ref false in
  Array.iteri lines ~f:(fun line_no line ->
      let len = String.length line in
      let rec scan i quote escaped =
        if i >= len then ()
        else if !in_comment then
          if i + 1 < len && Char.equal line.[i] '*' && Char.equal line.[i + 1] '/' then (
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
              if i + 1 < len && Char.equal line.[i] '/' && Char.equal line.[i + 1] '/' then ()
              else if i + 1 < len && Char.equal line.[i] '/' && Char.equal line.[i + 1] '*' then (
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
  List.rev !intervals

(* The callee of a line that is one call statement, [f(args);]. *)
let called_function line =
  let line = String.strip line in
  match String.lsplit2 line ~on:'(' with
  | Some (callee, _) when String.is_suffix line ~suffix:");" ->
      let callee = String.rstrip callee in
      Option.some_if
        ((not (String.is_empty callee))
        && String.for_all callee ~f:(fun c -> Char.is_alphanum c || Char.equal c '_'))
        callee
  | _ -> None

(* The outlined parts' definitions of a source given as its lines and their brace pairs, by name,
   each as the 0-based range of its body lines. *)
let part_bodies lines intervals =
  let n = Array.length lines in
  (* A line opening several braces: its first opens the construct the line's header declares, and
     that one closes last. *)
  let closing_of = Hashtbl.create (module Int) in
  List.iter intervals ~f:(fun (opening, closing) ->
      Hashtbl.update closing_of opening ~f:(function
        | Some c -> Int.max c closing
        | None -> closing));
  (* The outlined parts' definitions, by name: a header carrying the part attribute (its
     preprocessor guard is not one) that opens a body before any [;] ends it as a declaration. *)
  let parts = Hashtbl.create (module String) in
  Array.iteri lines ~f:(fun h line ->
      if
        String.is_substring line ~substring:Ir.Kernel_outline.part_attribute
        && not (String.is_prefix (String.lstrip line) ~prefix:"#")
      then
        match String.lsplit2 line ~on:'(' with
        | None -> ()
        | Some (prefix, _) -> (
            let rec body_opening k =
              if k >= n || String.mem lines.(k) ';' then None
              else if String.mem lines.(k) '{' then Some k
              else body_opening (k + 1)
            in
            match body_opening h with
            | None -> ()
            | Some opening ->
                Option.iter (Hashtbl.find closing_of opening) ~f:(fun closing ->
                    Hashtbl.set parts
                      ~key:(Ir.Kernel_outline.trailing_ident prefix)
                      ~data:(opening + 1, closing - 1))));
  parts

(* The 0-based numbers of the lines {!main_logic_lines} reads, each with the outlined part whose
   body holds it ([None]: the kernel's own main logic). *)
let statement_lines source =
  let lines = Array.of_list (String.split_lines source) in
  let intervals = brace_intervals lines in
  let parts = part_bodies lines intervals in
  let reached = Hash_set.create (module String) in
  let markers =
    Array.filter_mapi lines ~f:(fun i line ->
        Option.some_if (String.is_substring line ~substring:Ir.C_syntax.main_logic_marker) i)
  in
  if Array.is_empty markers && not (String.is_empty (String.strip source)) then
    Verdict.fail
      ("Generated.main_logic_lines: the source has no " ^ Ir.C_syntax.main_logic_marker
     ^ " marker, so no kernel's statements can be told from the rest of it");
  let numbered =
    Array.to_list markers
    |> List.concat_map ~f:(fun marker ->
        (* The kernel's definition: the innermost brace pair around its marker. *)
        let enclosing =
          List.filter intervals ~f:(fun (opening, closing) -> opening < marker && marker < closing)
          |> List.min_elt ~compare:(fun (o1, c1) (o2, c2) -> Int.compare (c1 - o1) (c2 - o2))
        in
        match enclosing with
        | None ->
            Verdict.fail
              (Printf.sprintf
                 "Generated.main_logic_lines: the marker on line %d is in no definition" (marker + 1));
            []
        | Some (_, closing) ->
            List.range (marker + 1) closing
            |> List.concat_map ~f:(fun k ->
                match
                  Option.bind
                    (called_function lines.(k))
                    ~f:(fun f -> Option.map (Hashtbl.find parts f) ~f:(fun body -> (f, body)))
                with
                | Some (f, (first, last)) ->
                    Hash_set.add reached f;
                    List.map (List.range first (last + 1)) ~f:(fun l -> (l, Some f))
                | None -> [ (k, None) ]))
  in
  Hashtbl.iter_keys parts ~f:(fun f ->
      if not (Hash_set.mem reached f) then
        Verdict.fail
          (Printf.sprintf
             "Generated.main_logic_lines: the outlined part %s is called from no kernel's main \
              logic, so a reading of the kernels' statements would miss its body"
             f));
  numbered

(** [main_logic_lines source] is the 1-based numbers of the lines of a generated source that hold
    its kernels' statements, in program order: each kernel's lines after its
    {!Ir.C_syntax.main_logic_marker}, up to the closing brace of its definition -- with, where the
    kernel was rendered OUTLINED (gh-ocannl-1603), each call to one of its parts replaced by that
    part's body lines. So a reading of a kernel's statements does not depend on whether a size
    threshold moved them into parts: the parts are found from what the renderer emitted -- the
    definitions carrying {!Ir.Kernel_outline.part_attribute}, and the calls the main logic makes to
    them -- not from how they are named. The kernel's signature, its local declarations and the
    prepended builtins are none of these lines.

    A source with no marker, a marker outside any definition, or a part definition no kernel's main
    logic calls (a reading that would silently miss its statements) fails the run through
    {!Verdict}. The empty source -- what {!read} answers after already failing -- reads as no lines.
*)
let main_logic_lines source = List.map (statement_lines source) ~f:(fun (k, _part) -> k + 1)

(** [main_logic source] is the text of {!main_logic_lines}: the kernels' statements, an outlined
    kernel's parts spliced in at their calls, one line per line. *)
let main_logic source =
  let lines = Array.of_list (String.split_lines source) in
  String.concat ~sep:"\n" (List.map (main_logic_lines source) ~f:(fun k -> lines.(k - 1)))

(** [outlined_parts source] is the names of the outlined parts a generated source defines (the
    definitions carrying {!Ir.Kernel_outline.part_attribute}), sorted: empty when no kernel of it
    was rendered outlined (gh-ocannl-1603). What a run forcing outlining claims, so that a threshold
    change leaving its kernels whole cannot turn the run into a second copy of the whole one
    (gh-ocannl-1708). *)
let outlined_parts source =
  let lines = Array.of_list (String.split_lines source) in
  Hashtbl.keys (part_bodies lines (brace_intervals lines)) |> List.sort ~compare:String.compare

(** Where a {!declared_array} is declared. *)
type declaration_scope =
  | Function_scope
      (** In a kernel's body ahead of its {!Ir.C_syntax.main_logic_marker}: its function-scope
          declarations, which stay in the kernel's own frame when it is outlined -- its parts take
          them as pointers. *)
  | Block_scope of { part : string option }
      (** Among its statements ({!main_logic_lines}) -- a pool-parallel [Grid] chunk's privatized
          tile, say -- with the outlined part whose body holds it ([None]: the kernel's own main
          logic). *)

type declared_array = {
  line : int;  (** 1-based; for a heap array, the line of its allocation. *)
  typ : string;  (** The element type as spelled, qualifiers included ([__shared__ float]). *)
  ident : string;  (** The identifier the declaration binds. *)
  elems : int;
  heap : bool;
      (** Declared as a pointer into a heap allocation ([C_syntax.heap_local_lines], gh-ocannl-1669)
          rather than as a stack array. *)
  scope : declaration_scope;
}

let c_keywords = [ "return"; "else"; "case"; "goto"; "do"; "if"; "for"; "while"; "sizeof" ]

let is_ident s =
  (not (String.is_empty s)) && String.for_all s ~f:(fun c -> Char.is_alphanum c || Char.equal c '_')

(* [<type words> <ident>[<n>]<attributes>< = {0}>;], the stack form of [C_syntax.local_array_decl]
   and of a workgroup-shared tile; a leading [__attribute__((...))] is dropped. As [(typ, ident,
   elems)]. *)
let stack_array_decl line =
  let line = String.strip line in
  let line =
    if String.is_prefix line ~prefix:"__attribute__((" then
      match String.substr_index line ~pattern:")) " with
      | Some i -> String.drop_prefix line (i + 3)
      | None -> line
    else line
  in
  match String.lsplit2 line ~on:'[' with
  | None -> None
  | Some (decl, rest) -> (
      let words = String.split decl ~on:' ' |> List.filter ~f:(Fn.non String.is_empty) in
      match (List.rev words, String.lsplit2 rest ~on:']') with
      | ident :: (_ :: _ as typ_rev), Some (n, after)
        when List.for_all words ~f:is_ident
             && (not (List.mem c_keywords (List.last_exn typ_rev) ~equal:String.equal))
             && (not (String.is_empty n))
             && String.for_all n ~f:Char.is_digit
             && String.is_suffix after ~suffix:";"
             && (String.equal after ";"
                || String.is_prefix after ~prefix:" __attribute__"
                || String.is_prefix after ~prefix:" = {0}") ->
          Some (String.concat ~sep:" " (List.rev typ_rev), ident, Int.of_string n)
      | _ -> None)

(* The heap form ([C_syntax.heap_local_lines]): the [<typ> * const <ident> = ...] line naming the
   node, whose allocation -- [void * const <raw> = malloc((size_t)<n> * sizeof(<typ>) ...] or
   [calloc(1, ...)], [<raw>] the renderer's derivation of [<ident>] -- is among [scope_lines]. As
   [(alloc_line, typ, ident, elems)]. *)
let heap_array_decl ~lines ~scope_lines line =
  let line = String.strip line in
  match String.substr_index line ~pattern:" * const " with
  | None -> None
  | Some i -> (
      let typ = String.prefix line i in
      let rest = String.drop_prefix line (i + String.length " * const ") in
      match String.lsplit2 rest ~on:' ' with
      | Some (ident, init)
        when is_ident ident && String.is_prefix init ~prefix:("= (" ^ typ ^ " *)(((uintptr_t)") ->
          let raw = Ir.C_syntax.derived_name Ir.C_syntax.heap_local_derivation ident in
          let alloc = "void * const " ^ raw ^ " = " in
          let size = "(size_t)" in
          List.find_map scope_lines ~f:(fun k ->
              let l = String.strip lines.(k) in
              if String.is_prefix l ~prefix:alloc then
                match String.substr_index l ~pattern:size with
                | None -> None
                | Some j -> (
                    let after = String.drop_prefix l (j + String.length size) in
                    match String.lsplit2 after ~on:' ' with
                    | Some (n, tail)
                      when String.for_all n ~f:Char.is_digit
                           && (not (String.is_empty n))
                           && String.is_prefix tail ~prefix:("* sizeof(" ^ typ ^ ")") ->
                        Some (k, typ, ident, Int.of_string n)
                    | _ -> None)
              else None)
      | _ -> None)

(** [declared_arrays source] is every array a generated source's kernels declare: first every
    kernel's function-scope arrays -- its body ahead of its {!Ir.C_syntax.main_logic_marker} -- then
    the arrays its statements declare, read over {!main_logic_lines}, each group in program order.
    So an OUTLINED kernel's block-scope arrays (a pool-parallel [Grid] chunk's privatized tiles,
    which move into a part with their loop) are read from its parts' bodies, and its function-scope
    ones from the kernel, where outlining leaves them (gh-ocannl-1778). Both the stack form and the
    heap form (gh-ocannl-1669) are read; a [const] pointer alias of a stack array is not a second
    array. The sources {!main_logic_lines} refuses fail the run through {!Verdict} here too. *)
let declared_arrays source =
  let lines = Array.of_list (String.split_lines source) in
  let intervals = brace_intervals lines in
  let in_scope ~scope scope_lines =
    List.filter_map scope_lines ~f:(fun k ->
        match stack_array_decl lines.(k) with
        | Some (typ, ident, elems) -> Some { line = k + 1; typ; ident; elems; heap = false; scope }
        | None ->
            Option.map
              (heap_array_decl ~lines ~scope_lines lines.(k))
              ~f:(fun (alloc, typ, ident, elems) ->
                { line = alloc + 1; typ; ident; elems; heap = true; scope }))
  in
  let function_scope =
    Array.to_list lines
    |> List.filter_mapi ~f:(fun i line ->
        Option.some_if (String.is_substring line ~substring:Ir.C_syntax.main_logic_marker) i)
    |> List.concat_map ~f:(fun marker ->
        let enclosing =
          List.filter intervals ~f:(fun (opening, closing) -> opening < marker && marker < closing)
          |> List.min_elt ~compare:(fun (o1, c1) (o2, c2) -> Int.compare (c1 - o1) (c2 - o2))
        in
        match enclosing with
        | Some (opening, _) -> in_scope ~scope:Function_scope (List.range (opening + 1) marker)
        (* A marker in no definition is [statement_lines]' failure, reported there. *)
        | None -> [])
  in
  let block_scope =
    List.group (statement_lines source) ~break:(fun (_, p1) (_, p2) ->
        not (Option.equal String.equal p1 p2))
    |> List.concat_map ~f:(fun group ->
        let part = snd (List.hd_exn group) in
        in_scope ~scope:(Block_scope { part }) (List.map group ~f:fst))
  in
  function_scope @ block_scope

(** {1 Declared identifiers (gh-ocannl-2102)}

    Every identifier a generated source's kernels DECLARE, read off the text, so a test can hold the
    names the renderer mints against the rules that keep a node label from shadowing them.

    What is read: each kernel's signature (its parameters), its body ahead of its
    {!Ir.C_syntax.main_logic_marker} (function scope) and its statements ({!main_logic_lines}, an
    outlined kernel's parts spliced in at their calls), and the body of every function-like macro
    the source [#define]s that a kernel statement calls (a macro body can declare: the register
    tile's bf16 row widening declares its [dst] argument). What is deliberately not read: the
    prepended builtins outside every kernel -- their functions' locals are another function's --
    whose typedef names are read only as TYPE names; an outlined part's own signature (it re-binds
    names its kernel declares); comments, string and character literals; the preprocessor
    conditionals [#if] / [#ifdef] / [#ifndef] / [#elif] / [#else] / [#endif] and [#pragma], whose
    lines are skipped, so BOTH arms of a conditional are read and every macro definition under them
    counts.

    The reading is a fail-closed allowlist. A statement that starts with a declaration specifier (a
    C type or qualifier, [__attribute__((...))], [__typeof__(...)], [typedef], or a name the source
    typedefs), or with an identifier followed by a declarator (an unknown type name: in C nothing
    else starts a statement so), must take one of these forms, or it is
    {!declarations.unrecognized}:

    - [<specifiers> <declarator>;] and [<specifiers> <declarator> = <initializer>;], one declarator,
      where a declarator is [*]s (each optionally followed by [const] / [restrict] / [volatile] /
      [__restrict]), the identifier, [[<digits>]] subscripts and trailing [__attribute__((...))]s;
    - [typedef <specifiers> <declarator>;];
    - a [for] header's first clause in either variable form, the loop's body read as a statement;
    - a signature parameter: [<specifiers> <declarator>], no initializer;
    - a statement-level call of a macro the source defines, [<macro>(<args>);]: what each of its
      definitions declares (read by these same rules, the macro's parameters admitted as type
      names), the arguments its declared parameters receive among them, each a bare identifier.

    Also unrecognized, as a declaration this reading cannot classify: a second declarator; a
    function declaration; [struct] / [union] / [enum]; a label, [goto], [case] or [default]; a call
    of a declaring macro anywhere but as a whole statement, a declared argument that is not a bare
    identifier, and a macro whose body pastes or stringifies ([#], [##]) or holds a statement these
    rules refuse; any other preprocessor directive in a kernel ([#define] there declares a macro).
    Everything else -- an expression statement, a call, [if] / [else] / [while] / [do] / [switch]
    heads (their bodies read as statements), [return] / [break] / [continue], braces -- declares
    nothing. *)

type declaration_site =
  | Parameter  (** A kernel signature's parameter. *)
  | Typedef  (** A [typedef] in a kernel's body, or in a macro body a statement expands. *)
  | Variable
      (** A variable declared in a kernel's body -- a [for] header's included, and one a macro call
          declares. *)

type declaration = { decl_line : int;  (** 1-based. *) ident : string; site : declaration_site }

type declarations = {
  declared : declaration list;  (** Each kernel's parameters, then the bodies, in source order. *)
  unrecognized : (int * string) list;
      (** A 1-based line and the statement's tokens: a statement outside the allowlisted forms. *)
}

type token = Ident of string | Punct of char | Literal

let tokenize (lines : (int * string) list) =
  let toks = ref [] and in_comment = ref false and directives = ref [] in
  (* A directive's line ending in a backslash continues it: a multi-line macro body is the
     directive's text, not code. *)
  let continued = ref false in
  List.iter lines ~f:(fun (line_no, line) ->
      let n = String.length line in
      let emit t = toks := (line_no, t) :: !toks in
      let stripped = String.lstrip line in
      let ends_continued = String.is_suffix (String.rstrip line) ~suffix:"\\" in
      let body () = String.strip (String.chop_suffix_if_exists (String.rstrip line) ~suffix:"\\") in
      if !continued then (
        (match !directives with
        | (l, text) :: rest -> directives := (l, text ^ " " ^ body ()) :: rest
        | [] -> ());
        continued := ends_continued)
      else if (not !in_comment) && String.is_prefix stripped ~prefix:"#" then (
        directives := (line_no, body ()) :: !directives;
        continued := ends_continued)
      else
        let rec scan i =
          if i >= n then ()
          else if !in_comment then
            if i + 1 < n && Char.equal line.[i] '*' && Char.equal line.[i + 1] '/' then (
              in_comment := false;
              scan (i + 2))
            else scan (i + 1)
          else
            let c = line.[i] in
            if i + 1 < n && Char.equal c '/' && Char.equal line.[i + 1] '*' then (
              in_comment := true;
              scan (i + 2))
            else if i + 1 < n && Char.equal c '/' && Char.equal line.[i + 1] '/' then ()
            else if Char.is_whitespace c then scan (i + 1)
            else if Char.is_alpha c || Char.equal c '_' then (
              let j = ref i in
              while !j < n && (Char.is_alphanum line.[!j] || Char.equal line.[!j] '_') do
                Int.incr j
              done;
              emit (Ident (String.sub line ~pos:i ~len:(!j - i)));
              scan !j)
            else if Char.is_digit c || (Char.equal c '.' && i + 1 < n && Char.is_digit line.[i + 1])
            then (
              let j = ref (i + 1) in
              while
                !j < n
                && (Char.is_alphanum line.[!j]
                   || Char.equal line.[!j] '_'
                   || Char.equal line.[!j] '.'
                   || Char.(line.[!j] = '+' || line.[!j] = '-')
                      && Char.(
                           line.[!j - 1] = 'e'
                           || line.[!j - 1] = 'E'
                           || line.[!j - 1] = 'p'
                           || line.[!j - 1] = 'P'))
              do
                Int.incr j
              done;
              emit Literal;
              scan !j)
            else if Char.equal c '"' || Char.equal c '\'' then (
              let j = ref (i + 1) in
              while !j < n && not (Char.equal line.[!j] c) do
                if Char.equal line.[!j] '\\' then j := !j + 2 else Int.incr j
              done;
              emit Literal;
              scan (!j + 1))
            else (
              emit (Punct c);
              scan (i + 1))
        in
        scan 0);
  (List.rev !toks, List.rev !directives)

(* Statements at bracket depth 0: a [;] ends one, and so does a brace that opens or closes a block.
   A [{] opens a block when the statement so far is empty, a control head or a function definition's
   head; anywhere else it opens an initializer or compound literal, which the statement runs through
   to its matching [}]. *)
let statements toks =
  let control = [ "for"; "if"; "else"; "while"; "do"; "switch" ] in
  let out = ref [] and cur = ref [] and depth = ref 0 and init_braces = ref 0 in
  let flush () =
    if not (List.is_empty !cur) then out := List.rev !cur :: !out;
    cur := []
  in
  List.iter toks ~f:(fun ((_, t) as tok) ->
      match t with
      | Punct ('(' | '[') ->
          Int.incr depth;
          cur := tok :: !cur
      | Punct (')' | ']') ->
          Int.decr depth;
          cur := tok :: !cur
      | Punct ';' when !depth = 0 && !init_braces = 0 ->
          cur := tok :: !cur;
          flush ()
      | Punct '{' when !depth = 0 && !init_braces = 0 -> (
          match List.last !cur with
          | None -> ()
          | Some (_, Ident kw) when List.mem control kw ~equal:String.equal -> flush ()
          (* A function definition's head: [... name(...) {], no [=] in it. *)
          | Some _
            when (match !cur with (_, Punct ')') :: _ -> true | _ -> false)
                 && not (List.exists !cur ~f:(function _, Punct '=' -> true | _ -> false)) ->
              flush ()
          | Some _ ->
              Int.incr init_braces;
              cur := tok :: !cur)
      | Punct '{' ->
          Int.incr init_braces;
          cur := tok :: !cur
      | Punct '}' when !init_braces > 0 ->
          Int.decr init_braces;
          cur := tok :: !cur
      | Punct '}' when !depth = 0 -> flush ()
      | _ -> cur := tok :: !cur);
  flush ();
  List.rev !out

let base_type_words =
  [
    "void";
    "char";
    "short";
    "int";
    "long";
    "float";
    "double";
    "signed";
    "unsigned";
    "_Bool";
    "bool";
    "size_t";
    "ptrdiff_t";
    "intptr_t";
    "uintptr_t";
    "int8_t";
    "int16_t";
    "int32_t";
    "int64_t";
    "uint8_t";
    "uint16_t";
    "uint32_t";
    "uint64_t";
    "_Float16";
    "__fp16";
    "__bf16";
  ]

let qualifier_words =
  [ "const"; "volatile"; "restrict"; "__restrict"; "static"; "register"; "inline"; "__volatile__" ]

let token_text = function Ident s -> s | Punct c -> String.of_char c | Literal -> "<literal>"
let render_tokens toks = String.concat ~sep:" " (List.map toks ~f:(fun (_, t) -> token_text t))

(* [toks] after an opening parenthesis: the tokens up to its matching [)], and those after it. *)
let split_paren toks =
  let rec go depth acc = function
    | [] -> (List.rev acc, [])
    | ((_, Punct ')') as t) :: rest ->
        if depth = 0 then (List.rev acc, rest) else go (depth - 1) (t :: acc) rest
    | ((_, Punct '(') as t) :: rest -> go (depth + 1) (t :: acc) rest
    | t :: rest -> go depth (t :: acc) rest
  in
  go 0 [] toks

let split_commas toks =
  let rec go depth cur acc = function
    | [] -> List.rev (List.rev cur :: acc)
    | (_, Punct ',') :: rest when depth = 0 -> go depth [] (List.rev cur :: acc) rest
    | ((_, Punct ('(' | '[' | '{')) as t) :: rest -> go (depth + 1) (t :: cur) acc rest
    | ((_, Punct (')' | ']' | '}')) as t) :: rest -> go (depth - 1) (t :: cur) acc rest
    | t :: rest -> go depth (t :: cur) acc rest
  in
  go 0 [] [] toks

(* Drop the balanced [__attribute__((...))] / [__typeof__(...)] groups at the head of [toks]. *)
let rec drop_groups = function
  | (_, Ident ("__attribute__" | "__typeof__" | "typeof")) :: (_, Punct '(') :: rest ->
      drop_groups (snd (split_paren rest))
  | toks -> toks

(* The single declarator of a declaration whose specifiers are already dropped: its identifier, or
   [None] when the rest is not an allowlisted form. [~with_init] admits [= ...]. *)
let declarator ~with_init toks =
  let rec stars = function
    | (_, Punct '*') :: rest -> stars rest
    | (_, Ident ("const" | "restrict" | "volatile" | "__restrict")) :: rest -> stars rest
    | toks -> toks
  in
  match stars toks with
  | (_, Ident name) :: rest -> (
      let rec suffixes = function
        | (_, Punct '[') :: (_, Literal) :: (_, Punct ']') :: rest -> suffixes rest
        | (_, Ident "__attribute__") :: _ as toks -> suffixes (drop_groups toks)
        | toks -> toks
      in
      let rest =
        match List.rev (suffixes rest) with (_, Punct ';') :: r -> List.rev r | r -> List.rev r
      in
      match rest with
      | [] -> Some name
      | (_, Punct '=') :: (_ :: _ as init) when with_init && List.length (split_commas init) = 1 ->
          Some name
      | _ -> None)
  | _ -> None

(* A macro definition read off a [#define] directive's text: function-like with its parameters, or
   object-like ([params = None]). *)
type macro = { params : string list option; body : (int * token) list }

let macro_of_directive (line, text) =
  let rest = String.lstrip (String.drop_prefix text 1) in
  match String.chop_prefix rest ~prefix:"define" with
  | None -> None
  | Some rest -> (
      let rest = String.lstrip rest in
      let name_len =
        Option.value ~default:(String.length rest)
          (String.lfindi rest ~f:(fun _ c -> not (Char.is_alphanum c || Char.equal c '_')))
      in
      let name = String.prefix rest name_len and after = String.drop_prefix rest name_len in
      let body_toks text = fst (tokenize [ (line, text) ]) in
      if String.is_empty name then None
      else
        match String.chop_prefix after ~prefix:"(" with
        | Some after -> (
            match String.lsplit2 after ~on:')' with
            | None -> None
            | Some (params, body) ->
                let params =
                  String.split params ~on:',' |> List.map ~f:String.strip
                  |> List.filter ~f:(Fn.non String.is_empty)
                in
                Some (name, { params = Some params; body = body_toks body }))
        | None -> Some (name, { params = None; body = body_toks after }))

(** [declarations source] is every identifier the kernels of a generated source declare, and every
    statement of theirs outside the allowlisted declaration forms (the boundary and the forms: the
    section head above). The sources {!main_logic_lines} refuses fail the run through {!Verdict}
    here too. *)
let declarations source =
  let lines = Array.of_list (String.split_lines source) in
  let numbered ks = List.map ks ~f:(fun k -> (k + 1, lines.(k))) in
  let all_toks, all_directives = tokenize (numbered (List.range 0 (Array.length lines))) in
  (* Every name the source typedefs, prelude included: read as type names only. *)
  let typedef_names =
    statements all_toks
    |> List.filter_map ~f:(function
      | (_, Ident "typedef") :: rest -> (
          let rec strip acc = function
            | [] -> List.rev acc
            | (_, Ident "__attribute__") :: _ as toks -> strip acc (drop_groups toks)
            | t :: rest -> strip (t :: acc) rest
          in
          match List.rev (strip [] rest) with
          | (_, Punct ';') :: (_, Ident name) :: _ | (_, Ident name) :: _ -> Some name
          | _ -> None)
      | _ -> None)
    |> Hash_set.of_list (module String)
  in
  let macros = Hashtbl.create (module String) in
  List.iter all_directives ~f:(fun d ->
      Option.iter (macro_of_directive d) ~f:(fun (name, m) ->
          Hashtbl.add_multi macros ~key:name ~data:m));
  let is_spec_word w =
    List.mem base_type_words w ~equal:String.equal
    || List.mem qualifier_words w ~equal:String.equal
    || Hash_set.mem typedef_names w
  in
  let line_of = function (l, _) :: _ -> l | [] -> 0 in
  (* What a macro's expansion declares: the index of a parameter it declares ([`Param]) or a name of
     its own ([`Fixed]), each with whether it is a typedef; [None] where the reading refuses one of
     its definitions. *)
  let effects_memo = Hashtbl.create (module String) in
  let rec effects ~depth name =
    match Hashtbl.find effects_memo name with
    | Some e -> e
    | None ->
        let e =
          if depth > 16 then None
          else
            Hashtbl.find_multi macros name
            |> List.fold ~init:(Some []) ~f:(fun acc m ->
                match (acc, m) with
                | None, _ -> None
                | Some acc, { params = None; body = [ (_, Ident target) ] }
                  when Hashtbl.mem macros target ->
                    Option.map (effects ~depth:(depth + 1) target) ~f:(( @ ) acc)
                | Some acc, { params; body } ->
                    if List.exists body ~f:(function _, Punct '#' -> true | _ -> false) then None
                    else
                      let params = Option.value params ~default:[] in
                      let found = ref [] and refused = ref false in
                      let decl ~site:_ ~typedef (_, ident) =
                        found :=
                          (match List.findi params ~f:(fun _ p -> String.equal p ident) with
                          | Some (i, _) -> (`Param i, typedef)
                          | None -> (`Fixed ident, typedef))
                          :: !found
                      in
                      scan ~depth:(depth + 1) ~params ~decl ~refuse:(fun _ -> refused := true) body;
                      if !refused then None else Some (acc @ List.rev !found))
        in
        Hashtbl.set effects_memo ~key:name ~data:e;
        e
  (* Read [toks], a run of statements. [decl] receives each declared identifier (with the line it is
     declared on), [refuse] each statement outside the allowlist; [params] are a macro body's
     parameters, admitted as type names. *)
  and scan ~depth ~params ~decl ~refuse toks =
    let is_spec w = is_spec_word w || List.mem params w ~equal:String.equal in
    let declaring_macro m =
      Hashtbl.mem macros m && match effects ~depth m with Some [] -> false | Some _ | None -> true
    in
    let declaration ~with_init toks =
      let is_typedef, body =
        match drop_groups toks with
        | (_, Ident "typedef") :: rest -> (true, rest)
        | rest -> (false, rest)
      in
      let rec specs acc = function
        | ((_, Ident w) as t) :: rest when is_spec w -> specs (t :: acc) rest
        | (_, Ident "__attribute__") :: _ as rest -> specs acc (drop_groups rest)
        | (_, Ident ("__typeof__" | "typeof")) :: _ as rest ->
            specs ((0, Literal) :: acc) (drop_groups rest)
        | rest -> (acc, rest)
      in
      let with_init = with_init && not is_typedef in
      let acc, rest = specs [] body in
      let found =
        match ((if List.is_empty acc then None else declarator ~with_init rest), rest, acc) with
        | (Some _ as found), _, _ -> found
        (* An unknown type name: the first identifier past the specifiers, its declarator after. *)
        | None, (_, Ident _) :: (_ :: _ as rest'), _
          when Option.is_some (declarator ~with_init rest') ->
            declarator ~with_init rest'
        (* The declarator spelled like a type name: a typedef's own name. *)
        | None, _, last :: _ :: _ -> declarator ~with_init (last :: rest)
        | None, _, _ -> None
      in
      match found with
      | Some ident -> decl ~site:Variable ~typedef:is_typedef (line_of toks, ident)
      | None -> refuse toks
    in
    let macro_call toks m args =
      match effects ~depth m with
      | None -> refuse toks
      | Some effs ->
          let args = if List.is_empty args then [] else split_commas args in
          List.iter effs ~f:(fun (eff, typedef) ->
              match eff with
              | `Fixed ident -> decl ~site:Variable ~typedef (line_of toks, ident)
              | `Param i -> (
                  match List.nth args i with
                  | Some [ (_, Ident a) ] -> decl ~site:Variable ~typedef (line_of toks, a)
                  | _ -> refuse toks))
    in
    let nested_declaring_call toks =
      List.exists toks ~f:(function _, Ident m -> declaring_macro m | _ -> false)
    in
    (* [<ident> <ident>] or [<ident> *... <ident>], then the end, [;], [=], [[], [,] or [(]: what
       only a declaration starts with, its type a name the reading does not know. *)
    let unknown_type_declaration rest =
      match
        List.drop_while rest ~f:(function
          | _, Punct '*' | _, Ident ("const" | "restrict" | "volatile" | "__restrict") -> true
          | _ -> false)
      with
      | (_, Ident _) :: ([] | (_, Punct (';' | '=' | '[' | ',' | '(')) :: _) -> true
      | _ -> false
    in
    let rec statement toks =
      match toks with
      | [] | [ (_, Punct ';') ] -> ()
      | (_, Ident "for") :: (_, Punct '(') :: rest ->
          let header, after = split_paren rest in
          statement (List.take_while header ~f:(function _, Punct ';' -> false | _ -> true));
          statement after
      | (_, Ident ("if" | "while" | "switch")) :: (_, Punct '(') :: rest ->
          let cond, after = split_paren rest in
          if nested_declaring_call cond then refuse toks;
          statement after
      | (_, Ident ("else" | "do")) :: rest -> statement rest
      (* Inline assembly declares nothing. *)
      | (_, Ident ("__asm__" | "asm" | "__asm")) :: _ -> ()
      | (_, Ident ("return" | "break" | "continue")) :: rest ->
          if nested_declaring_call rest then refuse toks
      | (_, Ident ("goto" | "case" | "default" | "struct" | "union" | "enum")) :: _ -> refuse toks
      | (_, Ident m) :: (_, Punct '(') :: rest when Hashtbl.mem macros m -> (
          let args, after = split_paren rest in
          match after with
          | [] | [ (_, Punct ';') ] ->
              if nested_declaring_call args then refuse toks else macro_call toks m args
          | _ -> if declaring_macro m || nested_declaring_call after then refuse toks)
      | (_, Ident w) :: _
        when is_spec w
             || List.mem
                  [ "typedef"; "__attribute__"; "__typeof__"; "typeof" ]
                  w ~equal:String.equal ->
          if nested_declaring_call toks then refuse toks else declaration ~with_init:true toks
      | (_, Ident _) :: (_, Punct ':') :: _ -> refuse toks
      | (_, Ident _) :: rest when unknown_type_declaration rest ->
          if nested_declaring_call toks then refuse toks else declaration ~with_init:true toks
      | _ -> if nested_declaring_call toks then refuse toks
    in
    List.iter (statements toks) ~f:statement
  in
  let declared = ref [] and unrecognized = ref [] in
  let refuse toks = unrecognized := (line_of toks, render_tokens toks) :: !unrecognized in
  let record ~site ~typedef (line, ident) =
    declared := { decl_line = line; ident; site = (if typedef then Typedef else site) } :: !declared
  in
  let directive (line, text) =
    let word =
      String.drop_prefix text 1 |> String.lstrip |> fun s ->
      match String.lfindi s ~f:(fun _ c -> not (Char.is_alpha c)) with
      | Some i -> String.prefix s i
      | None -> s
    in
    if
      not
        (List.mem
           [ "if"; "ifdef"; "ifndef"; "elif"; "else"; "endif"; "pragma" ]
           word ~equal:String.equal)
    then unrecognized := (line, text) :: !unrecognized
  in
  let read_body ks =
    let toks, directives = tokenize (numbered ks) in
    List.iter directives ~f:directive;
    scan ~depth:0 ~params:[] ~decl:record ~refuse toks
  in
  let intervals = brace_intervals lines in
  let markers =
    Array.to_list lines
    |> List.filter_mapi ~f:(fun i line ->
        Option.some_if (String.is_substring line ~substring:Ir.C_syntax.main_logic_marker) i)
  in
  let bodies =
    List.filter_map markers ~f:(fun marker ->
        List.filter intervals ~f:(fun (opening, closing) -> opening < marker && marker < closing)
        |> List.min_elt ~compare:(fun (o1, c1) (o2, c2) -> Int.compare (c1 - o1) (c2 - o2))
        (* None: [statement_lines]' failure, reported there. *)
        |> Option.map ~f:(fun (opening, _) -> (opening, marker)))
  in
  List.iter bodies ~f:(fun (opening, _) ->
      (* The signature: the lines from the one opening its parameter list to the brace. *)
      let rec start k =
        if k <= 0 then 0
        else
          let prev = String.strip lines.(k - 1) in
          if
            String.is_empty prev || String.is_prefix prev ~prefix:"#"
            || String.is_suffix prev ~suffix:";" || String.is_suffix prev ~suffix:"}"
            || String.is_suffix prev ~suffix:"*/"
          then k
          else start (k - 1)
      in
      let sig_toks, _ = tokenize (numbered (List.range (start opening) (opening + 1))) in
      match List.drop_while sig_toks ~f:(function _, Punct '(' -> false | _ -> true) with
      | _ :: after_paren -> (
          match fst (split_paren after_paren) with
          | [] | [ (_, Ident "void") ] -> ()
          | params ->
              List.iter (split_commas params) ~f:(fun param ->
                  scan ~depth:0 ~params:[] ~refuse
                    ~decl:(fun ~site:_ ~typedef:_ d -> record ~site:Parameter ~typedef:false d)
                    (param @ [ (line_of param, Punct ';') ])))
      | [] -> refuse sig_toks);
  List.iter bodies ~f:(fun (opening, marker) -> read_body (List.range (opening + 1) marker));
  read_body (List.map (statement_lines source) ~f:fst);
  { declared = List.rev !declared; unrecognized = List.rev !unrecognized }
