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

(** The source extension the run's backend emits: [.metal], [.cu], [.hip], or [.c] for the C
    backends. *)
let extension_of_backend backend_name =
  let has s = String.is_substring (String.lowercase backend_name) ~substring:s in
  if has "metal" then ".metal" else if has "cuda" then ".cu" else if has "hip" then ".hip" else ".c"

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

    [~backend_name] selects the source extension; tests already derive it as
    [String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")]. It is passed in
    rather than read here to keep the dependency at [arrayjit.utils]: this module resolves paths, it
    does not choose backends. *)
let is_symlink p =
  match Unix.lstat p with
  | { Unix.st_kind = Unix.S_LNK; _ } -> true
  | _ -> false
  | exception Unix.Unix_error _ -> false

let init ~backend_name =
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

(** [node_ident label] is the identifier a tensor node whose code name is [label] takes in this
    run's emitted kernels: [label] behind the backend dialect's node-identifier prefix
    ({!Ir.C_syntax.node_ident_prefix_for_backend}, gh-ocannl-1338). A claim pinning a node's
    identifier in emitted text spells it through here, so it states the same identifier on every
    backend. *)
let node_ident label =
  match !backend with
  | Some b -> Ir.C_syntax.node_ident_prefix_for_backend b ^ label
  | None ->
      uninitialized "Generated.node_ident";
      label

(* An identifier spelled with a number a compile mints afresh: a loop symbol ([i<digits>],
   [Indexing.symbol_ident]), a node identifier spelling its id ([n<digits>], optionally followed by
   [_<label>]: [Tnode.styled_ident] for an unlabeled node or one whose label repeats in the kernel),
   or a scope-local scalar spelling its scope id ([v<digits>_<ident>]). A type or builtin whose name
   ends in a digit ([uint3], [float4], [__half2]) is none of these shapes and stays as written. *)
let is_minted_ident ident =
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
  || Option.is_some (numbered ~prefix:"n")
  (* A node identifier behind a GPU dialect's prefix ([t_n12_x], gh-ocannl-1338). *)
  || (match !backend with
    | Some b ->
        Option.is_some (numbered ~prefix:(Ir.C_syntax.node_ident_prefix_for_backend b ^ "n"))
    | None -> false)
  || match numbered ~prefix:"v" with Some rest -> String.mem rest '_' | None -> false

(** [normalize_minted text] is a generated source with every identifier a compile spells with a
    freshly minted number -- loop symbols [i<digits>], node identifiers [n<digits>[_<label>]]
    (behind the backend's {!node_ident} prefix where it has one), scope-local scalars
    [v<digits>_<ident>] -- renamed [@0], [@1], ... by first appearance, and every run of whitespace
    collapsed to one space (a longer identifier moves the pretty-printer's line breaks). Two
    compiles of one program differ in nothing else -- not even across fresh builds of it, which mint
    new node ids (gh-ocannl-1473) -- so their normalized sources are equal, and a difference left is
    one in the code. Number literals are copied whole, so a hex float's letters are not read as an
    identifier. *)
let normalize_minted text =
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
        if is_minted_ident ident then
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
let main_logic_lines source =
  let lines = Array.of_list (String.split_lines source) in
  let n = Array.length lines in
  let intervals = brace_intervals lines in
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
                    List.range first (last + 1)
                | None -> [ k ]))
  in
  Hashtbl.iter_keys parts ~f:(fun f ->
      if not (Hash_set.mem reached f) then
        Verdict.fail
          (Printf.sprintf
             "Generated.main_logic_lines: the outlined part %s is called from no kernel's main \
              logic, so a reading of the kernels' statements would miss its body"
             f));
  List.map numbered ~f:(fun k -> k + 1)

(** [main_logic source] is the text of {!main_logic_lines}: the kernels' statements, an outlined
    kernel's parts spliced in at their calls, one line per line. *)
let main_logic source =
  let lines = Array.of_list (String.split_lines source) in
  String.concat ~sep:"\n" (List.map (main_logic_lines source) ~f:(fun k -> lines.(k - 1)))
