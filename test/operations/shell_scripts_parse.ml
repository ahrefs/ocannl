(* Every shell script in the repository parses, checked by the shell its shebang names.

   Nothing else in `dune build @check` or `dune runtest` looks at shell at all, and the repository
   runs a fair amount of it in places where a syntax error is expensive and late-discovered:
   `scripts/setup-ocaml-env.sh` is the SessionStart hook, so a broken one greets every session with
   a failing hook rather than a failing test; `tools/test-run.sh` fronts every suite run, and its
   Windows-only branches are exercised so rarely that CI added a step of its own to keep them from
   rotting. During PR #438 `bash -n` was run by hand about a dozen times and caught real breakage
   twice -- an edit that produced `D_IGN_PARENT="93.$$"# comment` (a `#` that starts no comment
   there, so the line is not the assignment it reads as), and a scratch harness whose sed-driven
   mutations kept emitting invalid shell until it grew a `bash -n` guard of its own. Both were found
   because someone thought to run the parser; nothing made them run it.

   `-n` parses and executes nothing, so this costs one short-lived process per script and cannot run
   anything the scripts do. That property is load-bearing rather than incidental, and two rules
   below exist to keep it: only shells whose `-n` means "parse only" are ever invoked, and only
   shebang options that cannot redirect what is read are carried through. The one thing this file
   does execute is its own literal fixtures: {!Errexit_execution_controls} runs them under bash to
   measure the errexit scans against the shell. No repository script is ever run.

   {1 How the scripts are found}

   By the rule's `(glob_files_rec ../../*.sh)` dependency, not by `git ls-files`. The glob is what
   makes a new script covered the day it lands, and -- the part git cannot do from inside a dune
   action -- it is also what makes dune RERUN this check when a script changes. A list recovered
   from git would leave the rule depending on nothing that moves when a script is edited, so the
   first run's pass would be served from cache forever after, which is the failure mode this
   repository keeps rediscovering. (`(universe)` would force the reruns, at the price of running
   every check on every dune invocation; the glob gets both properties for free.)

   The glob is over dune's view of the source tree, which also reaches scripts git does not track --
   deliberately: a scratch harness that keeps generating invalid shell is exactly the case PR #438
   hit, and one that lives in `tools/` for an afternoon is worth the same parse.

   Dune's scan skips dot-directories, which used to leave `.github/scripts/*.sh` uncovered, and the
   floor below did NOT catch that -- the twelve visible scripts keep it satisfied however many
   hidden ones are broken (Codex review round 3). The root `dune` names `.github` into the scan
   instead, and says there why it names that one and not `.claude/`. The floor is still worth having
   for what it does cover: a directory that drops out of the scan entirely.

   {1 The shebang is a command line, not a name}

   The first version of this check read the shebang for a word and threw the rest away, which is
   wrong in three ways that all have the same shape (Codex review round 2 on PR #454, three P2s):
   `#!/usr/bin/env -S -u FOO bash` names `FOO` as the interpreter if you take the first token
   without a leading dash, since `-u` has an operand; `#!/usr/bin/env -S bash -O extglob` parses
   only with `-O extglob`, so dropping it turns a valid script into a reported syntax error; and
   `#!/bin/dash` collapsed to `sh` is checked by whatever the host calls `sh`, which on macOS is
   bash -- so `function f() { :; }`, which dash rejects, would be reported as parsing.

   So {!Shebang.parse} reads the line as the command line it is -- but as the KERNEL builds it,
   which is the correction of round 3: everything after the interpreter path is ONE argument, not a
   word list. Only `env`'s `-S`/`--split-string` splits it, which is what that option exists for.
   The difference is not academic in either direction. `#!/usr/bin/env -u FOO bash` hands env the
   single argument `-u FOO bash`, so env unsets a variable named " FOO bash" and execs the script
   itself -- measured here, it does not reach bash, it HANGS (the kernel re-enters the same shebang
   until it gives up). And `#!/bin/bash -O extglob` hands bash the single argument `-O extglob`,
   which bash rejects with "invalid option". Reading either as a word list made this check pin a
   broken script as valid; both are now refused, naming the kernel semantics and pointing at `-S`.

   Beyond that, what the parser will not do is guess. An interpreter outside {!parse_only_shells} is
   refused rather than run, because `-n` means "parse only" for shells and something else entirely
   for a `python3` that a `.sh` file might name -- `perl -n` wraps the program in a read loop and
   RUNS it. An option in {!Shebang.refused_short}/{!Shebang.refused_long} is refused too, because
   `-c` and `-s` make the shell read its program from somewhere other than the file and `--version`
   exits before reading it at all. Every other option is carried through to the shell, which is the
   authority on its own: a universal "harmless flags" list is a guess, and it was wrong for
   `#!/bin/dash -B`. An `env` shebang that assigns a variable or asks for `-0` output is refused for
   the same reason -- the first changes the environment the lookup happens in, and the second is an
   `env` invocation that exits 125 rather than running anything. Refused means a failing verdict
   naming the token, not a silent pass.

   {!shebang_cases} pins that parser on synthetic lines -- the three above among them -- since the
   repository's own scripts exercise two shapes of the grammar and would not notice the rest
   regressing.

   {1 What the golden holds}

   The parser table, then one line per script, so that a script dropping out of the scan is visible
   in the diff, plus two claims that are NOT promotable: a floor on how many scripts were reached,
   and the presence of the two scripts named above. The diff alone would not do -- a scan that finds
   nothing produces an empty golden, and `dune promote` would record it.

   Which shell checked which script goes to stderr, which a `(test)` diff does not read, because it
   is machine-dependent: a host without `dash` checks a dash script with another POSIX shell. *)

open Base
open Stdio

let base_dir = Test_utils.Dune_stanza_scan.base_dir
let repo_relative = Test_utils.Dune_stanza_scan.repo_relative

(* The scripts tracked when this check was written, and the floor it holds the scan to. A floor
   rather than a count: adding a script must not make anyone promote a number, while a scan that
   stops reaching a directory has to fail rather than shrink the golden. *)
let script_floor = 12

(* Named because their breakage is not discovered by a test run: the first is the SessionStart hook,
   the second is what runs the suite. If the scan can no longer see these two it is not seeing the
   repository, whatever else it found. *)
let must_be_scanned = [ "scripts/setup-ocaml-env.sh"; "tools/test-run.sh" ]

module Shebang = struct
  (** What a script's first line says about how to parse it.

      {1 Why this grammar accepts no arguments at all}

      Seven review rounds on PR #454 went into reconstructing what the kernel, `env` and the shell
      would do with a shebang, and the option surface was wrong in both directions to the end:

      - round 6's P1 -- `#!/usr/bin/env -S bash helper.sh` built `bash helper.sh -n target`, and a
        shell stops processing options at its first operand, so `-n` became an argument and bash
        EXECUTED helper.sh (measured, with a marker file);
      - round 7's P1 -- `#!/usr/bin/env -S zsh --exec` builds `zsh -n --exec path`, and zsh's named
        options make `--exec` turn execution back ON after `-n` turned it off;
      - and round 7 again, in the other direction: `bash -n --posix path` exits 2, "invalid option",
        because bash wants its GNU long options BEFORE the short ones -- so the round-6 guard of
        putting `-n` first turned a shebang the table explicitly accepts into a reported syntax
        error (measured, both orderings).

      Placing `-n` is thus not safe in either position, and no ordering rule fixes an option that
      re-enables execution. Each of those was a fix for the round before it. So the grammar accepts
      a shell and NOTHING ELSE: the invocation is always exactly [<shell> -n <path>], with nothing
      from the shebang between the program and the file, and there is no argument surface left to be
      wrong about. What is accepted:

      - no shebang: the file is sourced, checked under both `sh` and `bash`;
      - [#!<path>], where the basename is in {!parse_only_shells};
      - [#!/usr/bin/env <shell>], and the same through [-S <shell>] / [-S<shell>] /
        [--split-string=<shell>].

      Everything else -- any argument to the shell, any `env` option, any `NAME=VALUE` assignment,
      an interpreter outside the whitelist -- is REFUSED: a failing verdict naming what could not be
      vouched for, never a silent pass. Nothing in this repository uses any of it, and a script that
      later wants to gets a message saying exactly why this check will not speak for it.

      The whitelist itself is load-bearing: `-n` means "parse only" for shells and something else
      entirely elsewhere -- `python3 -n` is an error and `perl -n` wraps the program in a read loop
      and RUNS it -- and a `.sh` file is free to name one of those. *)
  type t =
    | Sourced
        (** No shebang: the file is sourced rather than run, so it has no interpreter of its own and
            must parse under whichever shell sources it. *)
    | Interp of launch  (** The interpreter the kernel, or `env`, would exec. *)

  (** How the interpreter is found, which is not the same question for the two shebang forms and
      cannot be flattened to a name (round 4).

      A direct `#!/bin/bash` names a FILE, and the kernel execs that file; resolving `bash` on PATH
      instead can run a different build entirely -- on macOS `/bin/bash` is 3.2 while a Homebrew
      `bash` 5 sits earlier on PATH, and 5 accepts `declare -A` and [${v^^}] that 3.2 rejects. That
      skew is on this repository's own macOS CI leg. A shebang going through `env` is a PATH lookup
      by definition, so a name is the faithful reading there. *)
  and launch =
    | Path of string  (** A direct shebang: exec this exact file, as the kernel would. *)
    | Via_env of { env_path : string; command : string }
        (** Selected through `env`: the kernel execs [env_path], which then resolves [command] on
            PATH. BOTH have to exist for the script to launch, and the env binary was the half this
            check did not verify (round 9) -- it accepted `/bin/env` from the table and probed only
            the command, so on a host with just `/usr/bin/env` an unlaunchable script passed. Same
            rule as [Path] now, applied to the binary the kernel actually execs. *)

  let parse_only_shells = [ "sh"; "bash"; "dash"; "ash"; "ksh"; "mksh"; "zsh" ]

  (** POSIX shells that can stand in for one another. Consulted ONLY for the no-shebang case, where
      `sh` and `bash` are this check's own choice of checkers rather than anything the file asked
      for; a shell a shebang actually names is never substituted (round 7). bash is not a member in
      either direction: standing in for dash it accepts what dash rejects, and standing in for bash
      it rejects every bashism. *)
  let posix_family = [ "sh"; "dash"; "ash" ]

  let basename p = List.last_exn (String.split_on_chars p ~on:[ '/'; '\\' ])

  (** The interpreter paths that get `env` semantics. A basename test is not enough (round 8):
      `#!/opt/custom/env bash` was handed env's meaning and resolved `bash` on PATH, while the
      kernel would have tried to exec `/opt/custom/env` -- a path that may not exist, and if it does
      is not necessarily GNU env. These two are what every `env` shebang in the wild names; anything
      else basenamed `env` falls through to the direct-interpreter path, where it is refused for not
      being a shell. *)
  let env_paths = [ "/usr/bin/env"; "/bin/env" ]

  let words line =
    List.filter (String.split_on_chars line ~on:[ ' '; '\t' ]) ~f:(Fn.non String.is_empty)

  let no_arguments who args =
    Printf.sprintf
      "%s is given %s, and this check runs `<shell> -n <file>` and nothing else -- no placement of \
       `-n` is safe among shell options"
      who (String.concat ~sep:" " args)

  (** What `env` would exec, from the single argument the kernel hands it.

      That argument must be a bare command name. Nothing else is accepted, and `-S`/`--split-string`
      is the notable removal (round 10): whether an `env` implementation supports it is a property
      of the BINARY, not of its path -- BSD env and coreutils before 8.30 have no `-S` -- so
      accepting it meant either probing the feature or passing a shebang that fails before the shell
      starts. Neither was necessary, because since round 7 removed shell arguments the payload can
      only ever be a bare command name, which plain `env` resolves without `-S`. The option had
      become pure surface, so it is gone and `#!/usr/bin/env -S bash` is refused pointing at the
      plain spelling.

      `env` options and `NAME=VALUE` assignments are refused for the older reason: they build an
      environment that both the lookup and the shell's own startup depend on, which this check
      cannot reproduce -- `env -S PATH=/definitely/missing bash` exits 127, and a broken `BASH_ENV`
      makes `bash -n` fail a valid file (both measured). *)
  let env_command env_path argument =
    if String.is_prefix argument ~prefix:"-S" || String.is_prefix argument ~prefix:"--split-string"
    then
      Error
        "`env -S` support is a property of the env binary, not its path, and buys nothing here \
         since no shell arguments are accepted -- write `#!/usr/bin/env <shell>`"
    else if String.exists argument ~f:Char.is_whitespace then
      Error
        (Printf.sprintf
           "the kernel passes `%s` to `env` as ONE argument -- there is no command by that name"
           argument)
    else if String.is_prefix argument ~prefix:"-" then
      Error
        (Printf.sprintf
           "`env` option this check cannot vouch for: %s -- it builds an environment the command's \
            parsing depends on"
           argument)
    else if String.contains argument '=' then
      Error
        (Printf.sprintf
           "`env` assignment `%s`: the command is looked up, and parses, in an environment this \
            check cannot reproduce"
           argument)
    else Ok (Via_env { env_path; command = argument })

  let parse first_line =
    (* `#!` must be the file's first two BYTES: the kernel does not look past anything, so `
       #!/bin/bash` is not a shebang -- exec fails ENOEXEC and the caller's shell runs the file with
       `sh` (measured: such a file executes, under sh, not bash). *)
    if not (String.is_prefix first_line ~prefix:"#!") then Ok Sourced
    else
      let body = String.drop_prefix first_line 2 in
      (* A CR belongs to the interpreter PATH as far as the kernel is concerned: a CRLF
         `#!/bin/bash` file fails to exec with "/bin/bash^M: bad interpreter" (126) while `bash -n`
         on it exits 0 -- so normalising the byte away here would report a script that cannot run as
         parsing (round 7, measured). `.gitattributes` pins `*.sh` to LF, which is what makes this a
         guard rather than a routine path. *)
      if String.exists body ~f:(fun c -> Char.equal c '\r') then
        Error
          "the shebang line ends CRLF, and the kernel keeps the CR in the interpreter path (`bad \
           interpreter`); this file needs LF endings"
      else
        match words body with
        | [] -> Ok Sourced
        | first :: _ ->
            (* The kernel splits a shebang in exactly one place: the interpreter path, then the
               whole remainder as a single argument. *)
            let body = String.strip body in
            let argument = String.strip (String.drop_prefix body (String.length first)) in
            let resolved =
              if List.mem env_paths first ~equal:String.equal then
                if String.is_empty argument then Error "`env` with no command"
                else env_command first argument
              else if
                (* Whitelist first on this branch, so that a path this check gives no special
                   meaning to -- `/opt/custom/env`, a `python3` -- is refused for WHAT IT IS rather
                   than for the arguments it happens to carry. *)
                not (List.mem parse_only_shells (basename first) ~equal:String.equal)
              then
                Error
                  (Printf.sprintf "interpreter whose `-n` this check cannot vouch for: %s"
                     (basename first))
              else if String.is_empty argument then Ok (Path first)
              else Error (no_arguments (Printf.sprintf "`%s`" (basename first)) [ argument ])
            in
            Result.bind resolved ~f:(fun launch ->
                let shell =
                  basename (match launch with Path p -> p | Via_env { command; _ } -> command)
                in
                if List.mem parse_only_shells shell ~equal:String.equal then Ok (Interp launch)
                else
                  Error
                    (Printf.sprintf "interpreter whose `-n` this check cannot vouch for: %s" shell))

  (** Whether a first line looks like a shell shebang at all, decided WITHOUT regard to whether
      {!parse} accepts its arguments.

      This is what puts an unsuffixed file into the scan, and it has to be the looser question
      (round 7): `tools/run-tests` carrying `#!/bin/bash -c` is a file this check must report on,
      and keying scope off a successful parse silently dropped exactly those -- the twelve suffixed
      scripts kept the floor satisfied, so a broken executable could vanish from both the golden and
      the check. Mentioning a shell anywhere in the line is deliberately generous: over-including
      costs a refusal that names the file, while under-including costs silence. *)
  let mentions_a_shell first_line =
    (* A word can carry the shell attached to `env`'s split-string option, and both spellings are
       forms {!parse} accepts -- so a predicate that only basenamed the raw word filtered
       `#!/usr/bin/env -Sbash` out of the scan entirely, before anything could report on it (round
       8). Stripping those prefixes here keeps the two in step. *)
    let shell_of word =
      let word = String.strip word in
      let word =
        match String.chop_prefix word ~prefix:"--split-string=" with
        | Some rest -> rest
        | None -> (
            match String.chop_prefix word ~prefix:"-S" with Some rest -> rest | None -> word)
      in
      basename word
    in
    String.is_prefix first_line ~prefix:"#!"
    && List.exists
         (words (String.drop_prefix first_line 2))
         ~f:(fun word -> List.mem parse_only_shells (shell_of word) ~equal:String.equal)

  (** Lines whose SCOPE must hold whatever {!parse} makes of them, pinned separately because the two
      questions come apart: round 7 found scope keyed off a successful parse, which dropped the
      broken files this check exists for, and round 8 found the looser predicate blind to two
      spellings `parse` accepts. Neither bug is visible in the parse table. *)
  let scope_cases =
    [
      ("#!/bin/bash", true);
      ("#!/usr/bin/env bash", true);
      ("#!/usr/bin/env -Sbash", true);
      ("#!/usr/bin/env -S bash", true);
      ("#!/usr/bin/env --split-string=bash", true);
      (* In scope precisely BECAUSE parse refuses them: these are the files that must be
         reported. *)
      ("#!/bin/bash -c", true);
      ("#!/usr/bin/env -S bash helper.sh", true);
      (* Not shell scripts, and not this check's business. *)
      ("#!/usr/bin/env python3", false);
      ("#!/usr/bin/perl", false);
      ("", false);
      (" #!/bin/bash", false);
    ]

  (** How a parse reads in a verdict label, so that the golden is a table of what the parser does
      rather than a column of booleans. *)
  let render = function
    | Ok Sourced -> "sourced"
    | Ok (Interp (Path p)) -> p
    | Ok (Interp (Via_env { env_path; command })) -> command ^ " via " ^ env_path
    | Error reason -> "refused (" ^ reason ^ ")"

  (** The grammar, as a table of lines and what {!parse} must make of them. The repository's own
      scripts exercise two shapes of it; every other entry comes from a review round that found the
      parser wrong about that line, and is kept so the case cannot regress quietly. *)
  let shebang_cases =
    [
      (* What this repository's scripts use. A direct shebang renders as the PATH it names and an
         `env` one as a bare name (round 4): the kernel execs the file `#!/bin/bash` names, while
         `env` performs a PATH lookup. Flattening both to "bash" let a macOS `/bin/bash` 3.2 script
         be accepted by a Homebrew bash 5. *)
      ("#!/bin/bash", "/bin/bash");
      ("#!/usr/bin/env bash", "bash via /usr/bin/env");
      ("#!/bin/sh", "/bin/sh");
      ("#!/bin/dash", "/bin/dash");
      ("", "sourced");
      (* `env -S`, all three spellings (the attached one is round 6's). *)
      ( "#!/usr/bin/env -S bash",
        "refused (`env -S` support is a property of the env binary, not its path, and buys nothing \
         here since no shell arguments are accepted -- write `#!/usr/bin/env <shell>`)" );
      ( "#!/usr/bin/env -Sbash",
        "refused (`env -S` support is a property of the env binary, not its path, and buys nothing \
         here since no shell arguments are accepted -- write `#!/usr/bin/env <shell>`)" );
      ( "#!/usr/bin/env --split-string=bash",
        "refused (`env -S` support is a property of the env binary, not its path, and buys nothing \
         here since no shell arguments are accepted -- write `#!/usr/bin/env <shell>`)" );
      (* Not a shebang: `#!` must be the first two bytes, so the file is run by the caller's shell
         and gets the no-shebang treatment (round 5). *)
      (" #!/bin/bash", "sourced");
      (* CRLF: the kernel keeps the CR in the interpreter path, so the file cannot exec (126) even
         though `bash -n` on it exits 0 (round 7). *)
      ( "#!/bin/bash\r",
        "refused (the shebang line ends CRLF, and the kernel keeps the CR in the interpreter path \
         (`bad interpreter`); this file needs LF endings)" );
      (* No arguments, in either direction and for every reason rounds 5 to 7 found: an operand is
         EXECUTED (round 6's P1, measured with a marker), `--exec` turns execution back on after
         `-n` turned it off (round 7's P1), `-t` makes `bash -n` exit 0 on a file whose second line
         is a syntax error, `--` makes `-n` the filename (127), and `bash -n --posix` is itself
         rejected (2) because bash wants long options first. No placement of `-n` is safe among
         them, so none of them is accepted. *)
      ( "#!/bin/bash helper.sh",
        "refused (`bash` is given helper.sh, and this check runs `<shell> -n <file>` and nothing \
         else -- no placement of `-n` is safe among shell options)" );
      ( "#!/bin/zsh --exec",
        "refused (`zsh` is given --exec, and this check runs `<shell> -n <file>` and nothing else \
         -- no placement of `-n` is safe among shell options)" );
      ( "#!/bin/bash --posix",
        "refused (`bash` is given --posix, and this check runs `<shell> -n <file>` and nothing \
         else -- no placement of `-n` is safe among shell options)" );
      ( "#!/bin/bash -eu",
        "refused (`bash` is given -eu, and this check runs `<shell> -n <file>` and nothing else -- \
         no placement of `-n` is safe among shell options)" );
      ( "#!/bin/bash -O extglob",
        "refused (`bash` is given -O extglob, and this check runs `<shell> -n <file>` and nothing \
         else -- no placement of `-n` is safe among shell options)" );
      (* Kernel semantics (round 3): everything after the interpreter path is ONE argument, so an
         `env` shebang without `-S` names no command. This one measurably HANGS -- env execs the
         script, which re-enters the same shebang. *)
      ( "#!/usr/bin/env -u FOO bash",
        "refused (the kernel passes `-u FOO bash` to `env` as ONE argument -- there is no command \
         by that name)" );
      (* `env` builds an environment, and both the lookup and the shell's startup depend on it
         (rounds 5 and 6). Refused rather than emulated. *)
      ( "#!/usr/bin/env PATH=/missing",
        "refused (`env` assignment `PATH=/missing`: the command is looked up, and parses, in an \
         environment this check cannot reproduce)" );
      ( "#!/usr/bin/env -i",
        "refused (`env` option this check cannot vouch for: -i -- it builds an environment the \
         command's parsing depends on)" );
      (* `env` semantics belong to the canonical paths, not to every basename `env` (round 8): the
         kernel would try to exec `/opt/custom/env`, which may not exist and need not be GNU env,
         while this check was resolving `bash` on PATH and passing the file. *)
      ("#!/opt/custom/env bash", "refused (interpreter whose `-n` this check cannot vouch for: env)");
      (* Both accepted env paths are named in the rendering, because BOTH have to exist for the
         script to launch and they are not equally present -- most distributions ship only
         /usr/bin/env, and round 9 caught this check accepting /bin/env without asking. *)
      ("#!/bin/env bash", "bash via /bin/env");
      (* An interpreter whose `-n` is not a parse at all. *)
      ( "#!/usr/bin/env python3",
        "refused (interpreter whose `-n` this check cannot vouch for: python3)" );
    ]
end

(** Variables that make a shell parse something other than -- or in addition to -- the file it is
    handed, cleared for the child. `BASH_ENV` is the sharp one: bash expands and PARSES it before a
    non-interactive script, so an exported `BASH_ENV` pointing at a file with a syntax error makes
    `bash -n` fail every valid script in the repository (measured). `ENV` is its POSIX-shell
    counterpart, and `SHELLOPTS`/`BASHOPTS` set shell options at startup, which can reach the
    grammar. Clearing them is better than declaring them as dune dependencies: the check should not
    depend on the ambient environment at all, and a variable that cannot influence the run needs no
    dependency edge (Codex review round 6). *)
let cleared_variables = [ "BASH_ENV"; "ENV"; "SHELLOPTS"; "BASHOPTS" ]

let isolated_environment =
  lazy
    (Array.filter (Unix.environment ()) ~f:(fun binding ->
         let name = List.hd_exn (String.split binding ~on:'=') in
         not (List.mem cleared_variables name ~equal:String.equal)))

(** Run [prog -n args… path] with stdin closed, an isolated environment, and both output streams
    captured; return the exit status together with what the shell said.

    The argument vector is fixed -- program, `-n`, file -- and nothing from the shebang goes into
    it. That is what makes the promise to execute nothing structural rather than a property of the
    parser: round 6's P1 (`bash helper.sh -n target` executed helper.sh) and round 7's (`zsh -n
    --exec path` turns execution back on) both needed a shebang token to reach this vector, and none
    can. It also removes the ordering question round 7 raised in the other direction, where `bash -n
    --posix path` exits 2 because bash wants long options first.

    A missing [prog] arrives as exit 127 on Unix (the forked child cannot exec and exits with it)
    and as [Unix_error] on Windows; both mean the same thing here, so both become [None]. *)
let parse_check prog path =
  let tmp = Stdlib.Filename.temp_file "ocannl_shell_parse" ".log" in
  let devnull = if Stdlib.Sys.win32 then "NUL" else "/dev/null" in
  Exn.protect
    ~finally:(fun () -> try Stdlib.Sys.remove tmp with _ -> ())
    ~f:(fun () ->
      let out = Unix.openfile tmp [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
      let inp = Unix.openfile devnull [ Unix.O_RDONLY ] 0o400 in
      let argv = [| prog; "-n"; path |] in
      let status =
        Exn.protect
          ~finally:(fun () ->
            Unix.close out;
            Unix.close inp)
          ~f:(fun () ->
            match Unix.create_process_env prog argv (force isolated_environment) inp out out with
            | pid -> Some (snd (Unix.waitpid [] pid))
            | exception Unix.Unix_error _ -> None)
      in
      match status with
      | None | Some (Unix.WEXITED 127) -> None
      | Some status -> Some (status, String.strip (In_channel.read_all tmp)))

(** Whether [prog] exists here and can be executed, probed once per name by handing it a file that
    is a valid script in every shell there is.

    "Exists" is the question, deliberately, and it is not the same as "passed the probe". An earlier
    version answered false whenever the probe exited nonzero, which quietly conflated a missing
    binary with a present one that rejected `:` -- and since a false answer sends [resolve] to a
    substitute, a broken or non-shell interpreter would have been swapped out and the script judged
    by a DIFFERENT shell than its shebang names, silently. That is the same class of defect as the
    stand-in the round-2 review caught. So only a failure to exec at all (see [parse_check]) counts
    as absent; anything that ran keeps its verdict, and a `bash` on PATH that cannot parse `:` is a
    reported failure rather than a reason to consult another shell. *)
let available =
  let cache = Hashtbl.create (module String) in
  fun prog ->
    Hashtbl.find_or_add cache prog ~default:(fun () ->
        let tmp = Stdlib.Filename.temp_file "ocannl_shell_probe" ".sh" in
        Exn.protect
          ~finally:(fun () -> try Stdlib.Sys.remove tmp with _ -> ())
          ~f:(fun () ->
            Out_channel.write_all tmp ~data:":\n";
            Option.is_some (parse_check prog tmp)))

(** The shells that may stand in for [shell] when it is not installed, in preference order.

    Only within the POSIX family, and bash is not a member of it in either direction. Checking a
    dash script with bash is what the review objected to (bash accepts what dash rejects, so the
    check passes vacuously); checking a BASH script with dash is worse in the other direction --
    dash rejects arrays, [\[\[], and process substitution, so every bashism becomes a reported
    syntax error in a script that is perfectly valid. bash's absence is therefore a failure, not a
    substitution, and so is ksh's and zsh's. The narrow remaining case -- `sh` where the host has
    only `dash`, or the reverse -- is a genuine equivalence, and it is still announced. *)
let stand_ins shell =
  if List.mem Shebang.posix_family shell ~equal:String.equal then
    List.filter Shebang.posix_family ~f:(Fn.non (String.equal shell))
  else []

(** Whether a path names something this host can execute. Existence is not enough -- a present but
    non-executable interpreter fails exec just as surely -- so this asks the kernel's question. *)
let executable path =
  try
    Unix.access path [ Unix.X_OK ];
    true
  with Unix.Unix_error _ -> false

(** Where Git for Windows keeps the shells it ships, most preferred first; empty off Windows.

    `C:\Windows\System32\bash.exe` is the WSL launcher stub, and on a stock Windows runner image
    System32 precedes Git's `bin` on PATH. The stub is not a shell and parses nothing: it prints
    "Windows Subsystem for Linux has no installed distributions." and exits nonzero without ever
    opening the file. It is also not something [available] may call absent, and must not become one:
    it EXECS, so "exists" is the true answer to the question that function asks, and answering false
    there would send a present-but-broken shell to a stand-in and judge a script by a grammar its
    shebang never named -- the defect its comment above records. The shadowing is a defect of the
    LOOKUP, so it is fixed in the lookup: a shell Git ships is named by its own absolute path, which
    nothing on PATH can precede.

    The environment variables come first because they are what a non-default install has;
    `C:\Progra~1\Git` is the last resort and the same 8.3 spelling `ci.yml` pins for its Git Bash
    smoke step -- space-free, because a path with a space is a path something downstream will
    eventually split. *)
let windows_git_roots =
  lazy
    (if not Stdlib.Sys.win32 then []
     else
       let under name suffix =
         match Stdlib.Sys.getenv_opt name with
         | Some dir when not (String.is_empty dir) ->
             [ List.fold suffix ~init:dir ~f:Stdlib.Filename.concat ]
         | _ -> []
       in
       under "ProgramFiles" [ "Git" ] @ under "ProgramW6432" [ "Git" ]
       @ under "ProgramFiles(x86)" [ "Git" ]
       @ under "LOCALAPPDATA" [ "Programs"; "Git" ]
       @ [ {|C:\Progra~1\Git|} ])

(** The absolute path of [command] as Git for Windows ships it, if it ships it and it runs.

    `bin\<shell>.exe` before `usr\bin\<shell>.exe`: the former is Git's LAUNCHER, which prepends its
    own `/mingw64/bin:/usr/bin` to whatever PATH it inherits, and it is the file `ci.yml` vouches
    for on this image; the raw binary under `usr\bin` provisions nothing. `-n` needs no
    provisioning, but preferring the file CI already exercises keeps one answer to "which bash". Git
    ships `bash`, `sh` and `dash` and no `ksh` or `zsh`, which simply find no candidate here and
    fall back to the name on PATH. *)
let git_for_windows command =
  List.find_map (force windows_git_roots) ~f:(fun root ->
      List.find
        [
          Stdlib.Filename.concat (Stdlib.Filename.concat root "bin") (command ^ ".exe");
          Stdlib.Filename.concat
            (Stdlib.Filename.concat (Stdlib.Filename.concat root "usr") "bin")
            (command ^ ".exe");
        ]
        ~f:(fun path -> Stdlib.Sys.file_exists path && available path))

(** [command] as a program this host can run: what Git for Windows ships under that name, else the
    name itself for PATH to resolve. Off Windows this is exactly [available]. *)
let resolvable command =
  match git_for_windows command with
  | Some shipped -> Some shipped
  | None -> if available command then Some command else None

(** Resolve one wanted interpreter to a program that exists here, or say why it does not.

    Two rules, and round 9 is what made them one rule rather than two. Every path the KERNEL would
    exec must exist: the interpreter of a direct shebang, and the `env` binary of an `env` one. An
    absent one means the script cannot launch at all on this host, so reporting success for it would
    be reporting on something unrunnable. Windows is the exception throughout, and the only one: the
    shebang is honoured there by the shell rather than the kernel, so `/bin/sh` never resolves as a
    literal path and the basename on PATH is the faithful reading rather than a papering-over.

    A shell that a shebang NAMES is never substituted (round 7). Stand-ins apply only when [ours] is
    set -- the no-shebang case, where `sh` and `bash` are this check's own choice of checkers rather
    than anything the file asked for. *)
let rec resolve ~rel ?(ours = false) launch =
  let kernel_path_missing path =
    Error
      (Printf.sprintf "the kernel cannot exec `%s` on this host, so this script cannot run at all"
         path)
  in
  match launch with
  | Shebang.Path path ->
      if available path then Ok path
      else if Stdlib.Sys.win32 then (
        let name = Shebang.basename path in
        (* Announced AFTER resolving, so the name in the message is the program that will actually
           be handed the file rather than the basename the lookup started from. *)
        let resolved = resolve ~rel ~ours (Shebang.Via_env { env_path = ""; command = name }) in
        (match resolved with
        | Ok prog ->
            eprintf "%s: no `%s` on this host (Windows resolves the shebang itself), using `%s`\n"
              rel path prog
        | Error _ -> ());
        resolved)
      else kernel_path_missing path
  | Shebang.Via_env { env_path; command } -> (
      if
        (* [env_path] is empty only for the Windows fallback above and the no-shebang checkers,
           where there is no env binary in the picture. *)
        (not (String.is_empty env_path)) && (not Stdlib.Sys.win32) && not (executable env_path)
      then kernel_path_missing env_path
      else
        match resolvable command with
        | Some prog -> Ok prog
        | None -> (
            if not ours then
              Error
                (Printf.sprintf
                   "`%s` is not installed here, and a shell a shebang names is not substituted"
                   command)
            else
              (* The stand-in is announced by NAME -- which grammar checked the file is the fact a
                 reader needs -- while the program run is whatever that name resolved to. *)
              match
                List.find_map (stand_ins command) ~f:(fun name ->
                    Option.map (resolvable name) ~f:(fun prog -> (name, prog)))
              with
              | Some (name, prog) ->
                  eprintf "%s: no `%s` on this host, parsing with `%s` instead\n" rel command name;
                  Ok prog
              | None ->
                  Error (Printf.sprintf "neither `%s` nor a stand-in is installed here" command)))

(** The first line of a file, or [None] if it cannot be read as text at all. Binary files reach here
    through the directory globs, so a failure to read one is "not a script", not an error.

    [~fix_win_eol:false] is load-bearing, not tidiness: Stdio strips a trailing CR by DEFAULT, which
    silently repaired exactly the CRLF shebang round 7 is about -- the parser refused
    ["#!/bin/bash\r"] in its own table while the file it was handed arrived already normalised, so
    the check passed a script the kernel cannot exec. Read the bytes the kernel would read. *)
let first_line_of path =
  try In_channel.with_file path ~f:(In_channel.input_line ~fix_win_eol:false) with _ -> None

(** The shared lexical reader of the two errexit checks below: quote, substitution and comment
    skipping, shell words, the [set]/[shopt] reading of an errexit option, and
    {!numbered_spliced_lines}, which carries lexical context across physical lines. Both arms read a
    script through it and nothing else. *)
module Shell_lexer = struct
  let starts_at text ~pos token =
    let token_length = String.length token in
    pos + token_length <= String.length text
    && String.equal (String.sub text ~pos ~len:token_length) token

  let literal_shell_word word =
    let decoded = Buffer.create (String.length word) in
    let digit_value character =
      if Char.between character ~low:'0' ~high:'9' then
        Some (Char.to_int character - Char.to_int '0')
      else if Char.between character ~low:'a' ~high:'f' then
        Some (Char.to_int character - Char.to_int 'a' + 10)
      else if Char.between character ~low:'A' ~high:'F' then
        Some (Char.to_int character - Char.to_int 'A' + 10)
      else None
    in
    let add_ansi_escape index =
      let length = String.length word in
      if index + 1 >= length then (
        Buffer.add_char decoded '\\';
        index + 1)
      else
        let escaped = word.[index + 1] in
        let add character =
          Buffer.add_char decoded character;
          index + 2
        in
        match escaped with
        | 'a' -> add '\007'
        | 'b' -> add '\b'
        | 'e' | 'E' -> add '\027'
        | 'f' -> add '\012'
        | 'n' -> add '\n'
        | 'r' -> add '\r'
        | 't' -> add '\t'
        | 'v' -> add '\011'
        | '\\' | '\'' | '"' | '?' -> add escaped
        | 'x' ->
            let rec hexadecimal position count value =
              if position >= length || count = 2 then (position, value)
              else
                match digit_value word.[position] with
                | Some digit -> hexadecimal (position + 1) (count + 1) ((value * 16) + digit)
                | None -> (position, value)
            in
            let finish, value = hexadecimal (index + 2) 0 0 in
            if finish = index + 2 then add 'x'
            else (
              Buffer.add_char decoded (Char.of_int_exn value);
              finish)
        | ('u' | 'U') as kind ->
            let maximum = if Char.equal kind 'u' then 4 else 8 in
            let rec unicode position count value =
              if position >= length || count = maximum then (position, value)
              else
                match digit_value word.[position] with
                | Some digit -> unicode (position + 1) (count + 1) ((value * 16) + digit)
                | None -> (position, value)
            in
            let finish, value = unicode (index + 2) 0 0 in
            if finish = index + 2 then add kind
            else (
              if value <= 0x7f then Buffer.add_char decoded (Char.of_int_exn value)
              else if value <= 0x7ff then (
                Buffer.add_char decoded (Char.of_int_exn (0xc0 lor (value lsr 6)));
                Buffer.add_char decoded (Char.of_int_exn (0x80 lor (value land 0x3f))))
              else if value <= 0xffff && not (value >= 0xd800 && value <= 0xdfff) then (
                Buffer.add_char decoded (Char.of_int_exn (0xe0 lor (value lsr 12)));
                Buffer.add_char decoded (Char.of_int_exn (0x80 lor ((value lsr 6) land 0x3f)));
                Buffer.add_char decoded (Char.of_int_exn (0x80 lor (value land 0x3f))))
              else if value <= 0x10ffff then (
                Buffer.add_char decoded (Char.of_int_exn (0xf0 lor (value lsr 18)));
                Buffer.add_char decoded (Char.of_int_exn (0x80 lor ((value lsr 12) land 0x3f)));
                Buffer.add_char decoded (Char.of_int_exn (0x80 lor ((value lsr 6) land 0x3f)));
                Buffer.add_char decoded (Char.of_int_exn (0x80 lor (value land 0x3f))));
              finish)
        | '0' .. '7' ->
            let rec octal position count value =
              if position >= length || count = 3 then (position, value)
              else
                match digit_value word.[position] with
                | Some digit when digit < 8 -> octal (position + 1) (count + 1) ((value * 8) + digit)
                | _ -> (position, value)
            in
            let finish, value = octal (index + 1) 0 0 in
            Buffer.add_char decoded (Char.of_int_exn (value land 0xff));
            finish
        | _ -> add escaped
    in
    let rec loop index quote escaped =
      if index >= String.length word then Buffer.contents decoded
      else
        let character = word.[index] in
        match quote with
        | `Single ->
            if Char.equal character '\'' then loop (index + 1) `None false
            else (
              Buffer.add_char decoded character;
              loop (index + 1) `Single false)
        | `Double ->
            if escaped then (
              (* Inside double quotes a backslash escapes only these; before anything else it is
                 itself a character. *)
              if not (List.mem [ '$'; '`'; '"'; '\\'; '\n' ] character ~equal:Char.equal) then
                Buffer.add_char decoded '\\';
              Buffer.add_char decoded character;
              loop (index + 1) `Double false)
            else if Char.equal character '\\' then loop (index + 1) `Double true
            else if Char.equal character '"' then loop (index + 1) `None false
            else (
              Buffer.add_char decoded character;
              loop (index + 1) `Double false)
        | `Ansi_c ->
            if Char.equal character '\\' then loop (add_ansi_escape index) `Ansi_c false
            else if Char.equal character '\'' then loop (index + 1) `None false
            else (
              Buffer.add_char decoded character;
              loop (index + 1) `Ansi_c false)
        | `None ->
            if escaped then (
              Buffer.add_char decoded character;
              loop (index + 1) `None false)
            else if Char.equal character '\\' then loop (index + 1) `None true
            else if starts_at word ~pos:index "$'" then loop (index + 2) `Ansi_c false
            else if Char.equal character '\'' then loop (index + 1) `Single false
            else if Char.equal character '"' then loop (index + 1) `Double false
            else if List.mem [ ';'; '&'; '|' ] character ~equal:Char.equal then
              Buffer.contents decoded
            else (
              Buffer.add_char decoded character;
              loop (index + 1) `None false)
    in
    loop 0 `None false

  let is_named_errexit word = String.equal (literal_shell_word word) "errexit"

  (** The errexit effect of a [set] option list: [Some true] when it turns errexit on, [Some false]
      when it turns it off, [None] when it leaves it alone. Options apply left to right, so the last
      mention wins ([set -e +e] ends off), up to [--], a lone [-] or the first positional. *)
  let options_errexit options =
    let rec go last = function
      | [] | "--" :: _ -> last
      | option :: rest -> (
          let option = literal_shell_word option in
          match (option, rest) with
          | "-", _ -> last
          | ("-o" | "+o"), name :: rest ->
              go (if is_named_errexit name then Some (String.equal option "-o") else last) rest
          | ("-o" | "+o"), [] -> last
          | _ ->
              if String.is_prefix option ~prefix:"-" && not (String.is_prefix option ~prefix:"--")
              then go (if String.contains option 'e' then Some true else last) rest
              else if
                String.is_prefix option ~prefix:"+" && not (String.is_prefix option ~prefix:"++")
              then go (if String.contains option 'e' then Some false else last) rest
              else last)
    in
    go None options

  (* Skip one [$()] body while preserving its recursive quote scopes. The caller only needs the byte
     after the matching close: every list operator inside the substitution is nested by definition
     and cannot consume the outer negated pipeline. *)
  let rec skip_command_substitution line index =
    let rec loop index quote escaped depth =
      if index >= String.length line then index
      else
        let character = line.[index] in
        match quote with
        | `Single ->
            if Char.equal character '\'' then loop (index + 1) `None false depth
            else loop (index + 1) `Single false depth
        | `Ansi_c ->
            if escaped then loop (index + 1) `Ansi_c false depth
            else if Char.equal character '\\' then loop (index + 1) `Ansi_c true depth
            else if Char.equal character '\'' then loop (index + 1) `None false depth
            else loop (index + 1) `Ansi_c false depth
        | `Double ->
            if escaped then loop (index + 1) `Double false depth
            else if Char.equal character '\\' then loop (index + 1) `Double true depth
            else if starts_at line ~pos:index "$(" then
              loop (skip_command_substitution line (index + 2)) `Double false depth
            else if starts_at line ~pos:index "${" then
              loop (skip_parameter_expansion line (index + 2)) `Double false depth
            else if Char.equal character '`' then loop (index + 1) `Backtick_double false depth
            else if Char.equal character '"' then loop (index + 1) `None false depth
            else loop (index + 1) `Double false depth
        | (`Backtick_none | `Backtick_double) as backtick ->
            if escaped then loop (index + 1) backtick false depth
            else if Char.equal character '\\' then loop (index + 1) backtick true depth
            else if Char.equal character '`' then
              loop (index + 1)
                (match backtick with `Backtick_double -> `Double | `Backtick_none -> `None)
                false depth
            else loop (index + 1) backtick false depth
        | `None ->
            if escaped then loop (index + 1) `None false depth
            else if Char.equal character '\\' then loop (index + 1) `None true depth
            else if starts_at line ~pos:index "$'" then loop (index + 2) `Ansi_c false depth
            else if starts_at line ~pos:index "${" then
              loop (skip_parameter_expansion line (index + 2)) `None false depth
            else if Char.equal character '\'' then loop (index + 1) `Single false depth
            else if Char.equal character '"' then loop (index + 1) `Double false depth
            else if Char.equal character '`' then loop (index + 1) `Backtick_none false depth
            else if Char.equal character '(' then loop (index + 1) `None false (depth + 1)
            else if Char.equal character ')' then
              if depth = 1 then index + 1 else loop (index + 1) `None false (depth - 1)
            else loop (index + 1) `None false depth
    in
    loop index `None false 1

  and skip_parameter_expansion line index =
    let rec loop index quote escaped depth =
      if index >= String.length line then index
      else
        let character = line.[index] in
        match quote with
        | `Single ->
            if Char.equal character '\'' then loop (index + 1) `None false depth
            else loop (index + 1) `Single false depth
        | `Ansi_c ->
            if escaped then loop (index + 1) `Ansi_c false depth
            else if Char.equal character '\\' then loop (index + 1) `Ansi_c true depth
            else if Char.equal character '\'' then loop (index + 1) `None false depth
            else loop (index + 1) `Ansi_c false depth
        | `Double ->
            if escaped then loop (index + 1) `Double false depth
            else if Char.equal character '\\' then loop (index + 1) `Double true depth
            else if starts_at line ~pos:index "$(" then
              loop (skip_command_substitution line (index + 2)) `Double false depth
            else if starts_at line ~pos:index "${" then
              loop (skip_parameter_expansion line (index + 2)) `Double false depth
            else if Char.equal character '`' then loop (index + 1) `Backtick_double false depth
            else if Char.equal character '"' then loop (index + 1) `None false depth
            else loop (index + 1) `Double false depth
        | (`Backtick_none | `Backtick_double) as backtick ->
            if escaped then loop (index + 1) backtick false depth
            else if Char.equal character '\\' then loop (index + 1) backtick true depth
            else if Char.equal character '`' then
              loop (index + 1)
                (match backtick with `Backtick_double -> `Double | `Backtick_none -> `None)
                false depth
            else loop (index + 1) backtick false depth
        | `None ->
            if escaped then loop (index + 1) `None false depth
            else if Char.equal character '\\' then loop (index + 1) `None true depth
            else if starts_at line ~pos:index "$(" then
              loop (skip_command_substitution line (index + 2)) `None false depth
            else if starts_at line ~pos:index "${" then
              loop (skip_parameter_expansion line (index + 2)) `None false depth
            else if starts_at line ~pos:index "$'" then loop (index + 2) `Ansi_c false depth
            else if Char.equal character '\'' then loop (index + 1) `Single false depth
            else if Char.equal character '"' then loop (index + 1) `Double false depth
            else if Char.equal character '`' then loop (index + 1) `Backtick_none false depth
            else if Char.equal character '{' then loop (index + 1) `None false (depth + 1)
            else if Char.equal character '}' then
              if depth = 1 then index + 1 else loop (index + 1) `None false (depth - 1)
            else loop (index + 1) `None false depth
    in
    loop index `None false 1

  (** Whether an unquoted [#] at [index] starts a comment: only at the start of a word, so [foo#bar]
      is one word and [foo\ #bar] too (the space is escaped when an odd run of backslashes precedes
      it). *)
  let comment_starts line index =
    let rec backslashes_before index count =
      if index > 0 && Char.equal line.[index - 1] '\\' then
        backslashes_before (index - 1) (count + 1)
      else count
    in
    index = 0
    || (Char.is_whitespace line.[index - 1]
       || List.mem [ ';'; '&'; '|'; '('; ')' ] line.[index - 1] ~equal:Char.equal)
       && backslashes_before (index - 1) 0 % 2 = 0

  (** [line] cut at top-level [;], [&&], [||], [|], [&] and a comment: the pieces
      {!numbered_spliced_lines} searches for the [then]/[do] that closes a condition header. *)
  let command_fragments line =
    let length = String.length line in
    let add_fragment fragments start finish =
      String.sub line ~pos:start ~len:(finish - start) :: fragments
    in
    let rec loop index start quote escaped nesting fragments =
      if index >= length then List.rev (add_fragment fragments start length)
      else
        let character = line.[index] in
        match quote with
        | `Single ->
            if Char.equal character '\'' then loop (index + 1) start `None false nesting fragments
            else loop (index + 1) start `Single false nesting fragments
        | `Ansi_c ->
            if escaped then loop (index + 1) start `Ansi_c false nesting fragments
            else if Char.equal character '\\' then
              loop (index + 1) start `Ansi_c true nesting fragments
            else if Char.equal character '\'' then
              loop (index + 1) start `None false nesting fragments
            else loop (index + 1) start `Ansi_c false nesting fragments
        | `Double ->
            if escaped then loop (index + 1) start `Double false nesting fragments
            else if Char.equal character '\\' then
              loop (index + 1) start `Double true nesting fragments
            else if starts_at line ~pos:index "$(" then
              loop
                (skip_command_substitution line (index + 2))
                start `Double false nesting fragments
            else if starts_at line ~pos:index "${" then
              loop (skip_parameter_expansion line (index + 2)) start `Double false nesting fragments
            else if Char.equal character '`' then
              loop (index + 1) start `Backtick_double false nesting fragments
            else if Char.equal character '"' then
              loop (index + 1) start `None false nesting fragments
            else loop (index + 1) start `Double false nesting fragments
        | (`Backtick_none | `Backtick_double) as backtick ->
            if escaped then loop (index + 1) start backtick false nesting fragments
            else if Char.equal character '\\' then
              loop (index + 1) start backtick true nesting fragments
            else if Char.equal character '`' then
              loop (index + 1) start
                (match backtick with `Backtick_double -> `Double | `Backtick_none -> `None)
                false nesting fragments
            else loop (index + 1) start backtick false nesting fragments
        | `None ->
            if escaped then loop (index + 1) start `None false nesting fragments
            else if Char.equal character '\\' then
              loop (index + 1) start `None true nesting fragments
            else if Char.equal character '#' && comment_starts line index then
              List.rev (add_fragment fragments start index)
            else if starts_at line ~pos:index "$(" then
              loop (skip_command_substitution line (index + 2)) start `None false nesting fragments
            else if starts_at line ~pos:index "${" then
              loop (skip_parameter_expansion line (index + 2)) start `None false nesting fragments
            else if starts_at line ~pos:index "$'" then
              loop (index + 2) start `Ansi_c false nesting fragments
            else if Char.equal character '\'' then
              loop (index + 1) start `Single false nesting fragments
            else if Char.equal character '"' then
              loop (index + 1) start `Double false nesting fragments
            else if Char.equal character '`' then
              loop (index + 1) start `Backtick_none false nesting fragments
              (* Parentheses only: a single bracket is a word ([\[ \[ = x \]] is a valid test), so
                 balancing brackets let a literal one hide every separator after it. *)
            else if Char.equal character '(' then
              loop (index + 1) start `None false (nesting + 1) fragments
            else if Char.equal character ')' then
              loop (index + 1) start `None false (Int.max 0 (nesting - 1)) fragments
            else if nesting = 0 && Char.equal character ';' then
              loop (index + 1) (index + 1) `None false nesting (add_fragment fragments start index)
            else if
              nesting = 0
              && index + 1 < length
              && ((Char.equal character '&' && Char.equal line.[index + 1] '&')
                 || (Char.equal character '|' && Char.equal line.[index + 1] '|'))
            then
              loop (index + 2) (index + 2) `None false nesting (add_fragment fragments start index)
            else if
              nesting = 0 && Char.equal character '|'
              && (index = 0 || not (Char.equal line.[index - 1] '>'))
            then
              let next =
                if index + 1 < length && Char.equal line.[index + 1] '&' then index + 2
                else index + 1
              in
              loop next next `None false nesting (add_fragment fragments start index)
            else if
              nesting = 0 && Char.equal character '&'
              && (index = 0 || not (List.mem [ '>'; '<'; '|' ] line.[index - 1] ~equal:Char.equal))
              && (index + 1 >= length || not (Char.equal line.[index + 1] '>'))
            then
              loop (index + 1) (index + 1) `None false nesting (add_fragment fragments start index)
            else loop (index + 1) start `None false nesting fragments
    in
    loop 0 0 `None false 0 []

  let shell_words command =
    let current = Buffer.create (String.length command) in
    let finish words =
      if Buffer.length current = 0 then words
      else
        let word = Buffer.contents current in
        Buffer.clear current;
        word :: words
    in
    let rec loop index quote escaped words =
      if index >= String.length command then List.rev (finish words)
      else
        let character = command.[index] in
        match quote with
        | `Single ->
            Buffer.add_char current character;
            if Char.equal character '\'' then loop (index + 1) `None false words
            else loop (index + 1) `Single false words
        | `Ansi_c ->
            Buffer.add_char current character;
            if escaped then loop (index + 1) `Ansi_c false words
            else if Char.equal character '\\' then loop (index + 1) `Ansi_c true words
            else if Char.equal character '\'' then loop (index + 1) `None false words
            else loop (index + 1) `Ansi_c false words
        | `Double ->
            Buffer.add_char current character;
            if escaped then loop (index + 1) `Double false words
            else if Char.equal character '\\' then loop (index + 1) `Double true words
            else if starts_at command ~pos:index "$(" then (
              let finish = skip_command_substitution command (index + 2) in
              Buffer.add_substring current command ~pos:(index + 1) ~len:(finish - index - 1);
              loop finish `Double false words)
            else if starts_at command ~pos:index "${" then (
              let finish = skip_parameter_expansion command (index + 2) in
              Buffer.add_substring current command ~pos:(index + 1) ~len:(finish - index - 1);
              loop finish `Double false words)
            else if Char.equal character '`' then loop (index + 1) `Backtick_double false words
            else if Char.equal character '"' then loop (index + 1) `None false words
            else loop (index + 1) `Double false words
        | (`Backtick_none | `Backtick_double) as backtick ->
            Buffer.add_char current character;
            if escaped then loop (index + 1) backtick false words
            else if Char.equal character '\\' then loop (index + 1) backtick true words
            else if Char.equal character '`' then
              loop (index + 1)
                (match backtick with `Backtick_double -> `Double | `Backtick_none -> `None)
                false words
            else loop (index + 1) backtick false words
        | `None ->
            if escaped then (
              Buffer.add_char current character;
              loop (index + 1) `None false words)
            else if Char.equal character '\\' then (
              Buffer.add_char current character;
              loop (index + 1) `None true words)
            else if Char.is_whitespace character then loop (index + 1) `None false (finish words)
            else if starts_at command ~pos:index "$(" then (
              let finish = skip_command_substitution command (index + 2) in
              Buffer.add_substring current command ~pos:index ~len:(finish - index);
              loop finish `None false words)
            else if starts_at command ~pos:index "${" then (
              let finish = skip_parameter_expansion command (index + 2) in
              Buffer.add_substring current command ~pos:index ~len:(finish - index);
              loop finish `None false words)
            else if starts_at command ~pos:index "$'" then (
              Buffer.add_string current "$'";
              loop (index + 2) `Ansi_c false words)
            else if Char.equal character '\'' then (
              Buffer.add_char current character;
              loop (index + 1) `Single false words)
            else if Char.equal character '"' then (
              Buffer.add_char current character;
              loop (index + 1) `Double false words)
            else if Char.equal character '`' then (
              Buffer.add_char current character;
              loop (index + 1) `Backtick_none false words)
            else (
              Buffer.add_char current character;
              loop (index + 1) `None false words)
    in
    loop 0 `None false []

  let assignment_prefix word =
    let word = literal_shell_word word in
    match String.lsplit2 word ~on:'=' with
    | Some (name, _) when not (String.is_empty name) ->
        let name = if String.is_suffix name ~suffix:"+" then String.drop_suffix name 1 else name in
        (not (String.is_empty name))
        && (Char.is_alpha name.[0] || Char.equal name.[0] '_')
        && String.for_all (String.drop_prefix name 1) ~f:(fun character ->
            Char.is_alphanum character || Char.equal character '_')
    | _ -> false

  (** [Some attached] when [word] is a redirection, [attached] telling whether its target is in the
      same word. Descriptors are a digit run or bash's variable form [{fd}]; [<<-] is an operator
      whose target follows (the generic reading would take its [-] for an attached target). Both
      arms and the errexit gate read redirections through this one function. *)
  let redirection_prefix word =
    let word =
      match String.chop_prefix word ~prefix:"{" with
      | Some rest -> (
          match String.lsplit2 rest ~on:'}' with
          | Some (name, operator)
            when assignment_prefix (name ^ "=")
                 && List.exists [ "<"; ">" ] ~f:(fun prefix -> String.is_prefix operator ~prefix) ->
              operator
          | _ -> word)
      | None -> word
    in
    if String.equal word "<<-" then Some false
    else
      let rec skip_descriptor index =
        if index < String.length word && Char.is_digit word.[index] then skip_descriptor (index + 1)
        else index
      in
      let operator = if String.is_prefix word ~prefix:"&>" then 0 else skip_descriptor 0 in
      if
        operator >= String.length word
        || not
             (List.mem [ '<'; '>' ] word.[operator] ~equal:Char.equal
             || Char.equal word.[operator] '&'
                && operator + 1 < String.length word
                && Char.equal word.[operator + 1] '>')
      then None
      else
        let rec skip_operator index =
          if
            index < String.length word
            && List.mem [ '<'; '>'; '&'; '|' ] word.[index] ~equal:Char.equal
          then skip_operator (index + 1)
          else index
        in
        Some (skip_operator operator < String.length word)

  (** The errexit effect of one simple command (see {!options_errexit}), read through what can stand
      in front of the builtin: redirections, assignments, [!], [time] ([-p], [--]), and the
      builtin-runners [builtin] ([--]) and [command] ([-p], [--]). [shopt -s -o errexit] is
      [set -o errexit] by another name and [shopt -u -o errexit] is [set +o errexit]. Nothing else
      is followed: a [set] behind [eval], [source], [env] or [bash -c] is not read.

      Turning errexit OFF is reported only for the plainest spelling, where the builtin certainly
      runs in this shell and the words are what they look like: a bare, unquoted [set] with nothing
      in front of it -- no assignment, redirection (a failed one skips the command), [!], [time] (an
      external command under a POSIX shell), [builtin] or [command] (either may itself be a
      function) -- that the script does not shadow with a function of that name ([~shadowed]), and
      whose option words, up to the first positional, are literal [-]/[+] letter bundles or
      [-o]/[+o] with a literal name. Every other spelling ([shopt -u -o errexit], a Bash-only
      builtin a POSIX shell lacks; [builtin set +e]; [set +"$e"]) may turn it off at run time but is
      not trusted to -- loud. Turning it on is reported for every spelling, wherever it may run. *)
  let command_errexit ?(shadowed = []) command =
    let rec strip_redirections = function
      | [] -> []
      | word :: rest -> (
          match redirection_prefix word with
          | Some true -> strip_redirections rest
          | Some false -> strip_redirections (List.drop rest 1)
          | None -> word :: strip_redirections rest)
    in
    let is_time word = String.equal (literal_shell_word word) "time" in
    let rec drop_command_prefixes = function
      | word :: rest when assignment_prefix word -> drop_command_prefixes rest
      | time :: option :: dashdash :: rest
        when is_time time
             && String.equal (literal_shell_word option) "-p"
             && String.equal (literal_shell_word dashdash) "--" ->
          drop_command_prefixes rest
      | time :: option :: rest
        when is_time time && List.mem [ "-p"; "--" ] (literal_shell_word option) ~equal:String.equal
        ->
          drop_command_prefixes rest
      | word :: rest when List.mem [ "!"; "time" ] (literal_shell_word word) ~equal:String.equal ->
          drop_command_prefixes rest
      | words -> words
    in
    let all_words = shell_words (String.strip command) in
    let unredirected = strip_redirections all_words in
    let bare = drop_command_prefixes unredirected in
    let words = List.map bare ~f:literal_shell_word in
    let rec unwrap = function
      | "builtin" :: rest -> unwrap_options rest
      | "command" :: rest -> unwrap_command rest
      | words -> words
    and unwrap_options = function "--" :: rest -> unwrap rest | words -> unwrap words
    and unwrap_command = function
      | ("-p" | "--") :: rest -> unwrap_command rest
      | words -> unwrap words
    in
    let unwrapped = unwrap words in
    let plain word =
      (not (String.is_empty word))
      && String.for_all word ~f:(fun c -> Char.is_alphanum c || Char.equal c '_')
    in
    let rec plain_options = function
      | [] | ("--" | "-") :: _ -> true
      | ("-o" | "+o") :: name :: rest -> plain name && plain_options rest
      | option :: rest when String.length option > 1 && Char.(option.[0] = '-' || option.[0] = '+')
        ->
          plain (String.drop_prefix option 1) && plain_options rest
      | _ -> (* the first positional ends the options *) true
    in
    let certain =
      List.length unredirected = List.length all_words
      &&
      match all_words with
      | "set" :: options ->
          (not (List.mem shadowed "set" ~equal:String.equal)) && plain_options options
      | _ -> false
    in
    Option.filter ~f:(fun on -> on || certain)
    @@
    match unwrapped with
    | "set" :: options -> options_errexit options
    | "shopt" :: arguments ->
        let flags, names =
          List.split_while arguments ~f:(fun word ->
              String.is_prefix word ~prefix:"-" && not (String.equal word "--"))
        in
        let has flag = List.exists flags ~f:(fun word -> String.contains word flag) in
        if
          has 'o'
          && List.mem
               (List.filter names ~f:(Fn.non (String.equal "--")))
               "errexit" ~equal:String.equal
        then
          match (has 's', has 'u') with
          | true, false -> Some true
          | false, true -> Some false
          | _ -> (* Both: bash refuses the command. *) None
        else None
    | _ -> None

  (** Shared cross-line lexical context for the two errexit checks. This is a logical-line reader,
      not an execution model: quotes and opaque substitutions stay in one line, and outer heredoc
      bodies never become code. Multiple literal delimiters, quoted/escaped delimiters
      (single/double quotes and backslashes) and [<<-] are supported; [<<<] is a here-string.
      Dollar-quoted delimiters, multiline delimiters, continued unquoted bodies and unterminated
      bodies are refused explicitly: none is guessed at or allowed to hide the rest of the file. The
      condition grammar is bounded to headers beginning with raw [if]/[elif]/[while]/[until],
      through a raw [then]/[do] in command position. Compound structure and option transitions are
      {!Shell_context}'s to read; heredocs inside substitutions remain outside both.

      Preserve physical start numbers and newline bytes inside quotes. A closing quote followed by
      code must remain visible to the statement scanner, not disappear with its data. *)
  let numbered_spliced_lines ?(refuse = fun _line _reason -> ()) text =
    let length = String.length text in
    let current = Buffer.create length in
    let lines = ref [] and first = ref 1 and number = ref 1 in
    let heredocs = ref [] in
    let condition = ref false in
    let starts_comment () =
      (* Judge word boundaries after backslash-newline removal, not against the original preceding
         newline: [foo\\] / [#bar] is the single word [foo#bar]. *)
      let prefix = Buffer.contents current in
      comment_starts (prefix ^ "#") (String.length prefix)
    in
    let condition_closed line =
      List.exists (command_fragments line) ~f:(fun fragment ->
          match shell_words (String.strip fragment) with ("then" | "do") :: _ -> true | _ -> false)
    in
    let emit () =
      let line = Buffer.contents current in
      let begins_condition =
        match shell_words (String.strip line) with
        | ("if" | "elif" | "while" | "until") :: _ -> true
        | _ -> false
      in
      if (!condition || begins_condition) && not (condition_closed line) then (
        condition := true;
        (* A newline after the header keyword or a list operator continues its operand; otherwise it
           separates condition commands, just like a semicolon. *)
        let words = shell_words (String.strip line) in
        let last = Option.value (List.last words) ~default:"" in
        Buffer.add_string current
          (if List.mem [ "if"; "elif"; "while"; "until"; "&&"; "||"; "|" ] last ~equal:String.equal
           then " "
           else "; "))
      else (
        lines := (!first, line) :: !lines;
        Buffer.clear current;
        condition := false;
        first := !number + 1)
    in
    let add_range start finish =
      Buffer.add_substring current text ~pos:start ~len:(finish - start);
      for index = start to finish - 1 do
        if Char.equal text.[index] '\n' then Int.incr number
      done
    in
    (* Delimiter words undergo quote removal, never expansion. Read that word separately from code
       so a delimiter's quote does not open a multiline value in the surrounding command. *)
    let delimiter start =
      let rec skip index =
        if index < length && List.mem [ ' '; '\t' ] text.[index] ~equal:Char.equal then
          skip (index + 1)
        else index
      in
      let start = skip start in
      let decoded = Buffer.create 32 in
      let quoted = ref false in
      let rec word index quote =
        if index >= length then index
        else
          let c = text.[index] in
          let add () =
            Buffer.add_char decoded c;
            word (index + 1) quote
          in
          match quote with
          | `Single -> if Char.equal c '\'' then word (index + 1) `None else add ()
          | `Double ->
              if Char.equal c '"' then word (index + 1) `None
              else if
                Char.equal c '\\'
                && index + 1 < length
                && List.mem [ '$'; '`'; '"'; '\\'; '\n' ] text.[index + 1] ~equal:Char.equal
              then (
                if not (Char.equal text.[index + 1] '\n') then
                  Buffer.add_char decoded text.[index + 1];
                word (index + 2) `Double)
              else add ()
          | `None ->
              if starts_at text ~pos:index "$'" || starts_at text ~pos:index "$\"" then (
                quoted := true;
                refuse !number "dollar-quoted heredoc delimiter";
                Buffer.add_char decoded '$';
                word (index + 2) (if Char.equal text.[index + 1] '\'' then `Single else `Double))
              else if Char.equal c '\\' && index + 1 < length then (
                if not (Char.equal text.[index + 1] '\n') then (
                  quoted := true;
                  Buffer.add_char decoded text.[index + 1]);
                word (index + 2) `None)
              else if Char.equal c '\'' then (
                quoted := true;
                word (index + 1) `Single)
              else if Char.equal c '"' then (
                quoted := true;
                word (index + 1) `Double)
              else if
                Char.is_whitespace c
                || List.mem [ ';'; '&'; '|'; '<'; '>'; '('; ')' ] c ~equal:Char.equal
              then index
              else add ()
      in
      let finish = word start `None in
      (finish, Buffer.contents decoded, !quoted)
    in
    let rec skip_bodies index = function
      | [] -> index
      | (delimiter, strip_tabs, quoted) :: rest ->
          let rec body index =
            if index >= length then (
              refuse !number "unterminated outer heredoc";
              index)
            else
              let finish =
                match String.index_from text index '\n' with
                | Some finish -> finish
                | None -> length
              in
              let line = String.sub text ~pos:index ~len:(finish - index) in
              let line = if strip_tabs then String.lstrip ~drop:(Char.equal '\t') line else line in
              (* Unquoted bodies remove escaped newlines before delimiter recognition. Refuse that
                 form rather than guess which later physical lines remain shell code. *)
              let rec backslashes index count =
                if index >= 0 && Char.equal line.[index] '\\' then
                  backslashes (index - 1) (count + 1)
                else count
              in
              if
                (not quoted) && finish < length
                && Int.rem (backslashes (String.length line - 1) 0) 2 = 1
              then refuse !number "continued unquoted heredoc body";
              let next = if finish < length then finish + 1 else finish in
              if finish < length then Int.incr number;
              if String.equal line delimiter then skip_bodies next rest else body next
          in
          body index
    in
    let rec loop index quote =
      if index >= length then (
        ignore (skip_bodies index (List.rev !heredocs));
        if Buffer.length current > 0 then lines := (!first, Buffer.contents current) :: !lines;
        List.rev !lines)
      else
        let c = text.[index] in
        let take finish next_quote =
          add_range index finish;
          loop finish next_quote
        in
        match quote with
        | `Single -> take (index + 1) (if Char.equal c '\'' then `None else `Single)
        | `Ansi_c ->
            if Char.equal c '\\' then take (Int.min length (index + 2)) `Ansi_c
            else take (index + 1) (if Char.equal c '\'' then `None else `Ansi_c)
        | `Double ->
            if starts_at text ~pos:index "$(" then
              take (skip_command_substitution text (index + 2)) `Double
            else if starts_at text ~pos:index "${" then
              take (skip_parameter_expansion text (index + 2)) `Double
            else if Char.equal c '`' then take (index + 1) `Backtick_double
            else if Char.equal c '\\' && index + 1 < length && Char.equal text.[index + 1] '\n' then (
              Int.incr number;
              loop (index + 2) `Double)
            else if Char.equal c '\\' then take (Int.min length (index + 2)) `Double
            else take (index + 1) (if Char.equal c '"' then `None else `Double)
        | (`Backtick_none | `Backtick_double) as backtick ->
            if Char.equal c '\\' then take (Int.min length (index + 2)) backtick
            else
              take (index + 1)
                (if Char.equal c '`' then
                   match backtick with `Backtick_double -> `Double | `Backtick_none -> `None
                 else backtick)
        | `None ->
            if Char.equal c '\n' then (
              emit ();
              Int.incr number;
              let next = skip_bodies (index + 1) (List.rev !heredocs) in
              heredocs := [];
              if Buffer.length current = 0 then first := !number;
              loop next `None)
            else if Char.equal c '\\' && index + 1 < length && Char.equal text.[index + 1] '\n' then (
              Int.incr number;
              loop (index + 2) `None)
            else if Char.equal c '\\' then take (Int.min length (index + 2)) `None
            else if Char.equal c '#' && starts_comment () then
              let finish = Option.value (String.index_from text index '\n') ~default:length in
              loop finish `None
            else if starts_at text ~pos:index "$(" then
              take (skip_command_substitution text (index + 2)) `None
            else if starts_at text ~pos:index "${" then
              take (skip_parameter_expansion text (index + 2)) `None
            else if starts_at text ~pos:index "$'" then take (index + 2) `Ansi_c
            else if Char.equal c '\'' then take (index + 1) `Single
            else if Char.equal c '"' then take (index + 1) `Double
            else if Char.equal c '`' then take (index + 1) `Backtick_none
            else if starts_at text ~pos:index "((" then
              take (skip_command_substitution text (index + 1)) `None
            else if starts_at text ~pos:index "<<<" then take (index + 3) `None
            else if starts_at text ~pos:index "<<" then (
              let strip_tabs = index + 2 < length && Char.equal text.[index + 2] '-' in
              let finish, word, quoted = delimiter (index + if strip_tabs then 3 else 2) in
              if String.contains word '\n' then refuse !number "multiline heredoc delimiter";
              heredocs := (word, strip_tabs, quoted) :: !heredocs;
              take finish `None)
            else take (index + 1) `None
    in
    loop 0 `None
end

(** The execution context the two errexit checks judge a statement in (gh-ocannl-1220).

    [! cmd] and [[ A ] && [ B ]] are inert assertions only where errexit is on and nothing consumes
    the statement's status, and neither question is answered by the statement's own line: errexit is
    whatever the [set] commands before it left, and a statement's status is consumed by more than
    the operators around it -- a function returns its last command's status. This module reads that
    much of a script and no more. It asserts that assertions are live; it does not emulate bash.
    Every rule below is pinned by a row of {!Errexit_execution_controls}, which runs it under the
    host's bash -- 3.2 on macOS, 5 on Linux -- against a plain [false] in the same place.

    {1 What it reads}

    The compound structure, over {!Shell_lexer.numbered_spliced_lines}' logical lines in source
    order: brace groups and subshells in command position; function bodies ([f() {], [f() (],
    [function f {], [function f() (], also with the body opening on the next line);
    [if]/[elif]/[else]/[fi]; [while]/[until]/[for]/[select] through [do]/[done]; and [case] through
    [in], its patterns (an optional leading [(], [|] alternatives, extglob parentheses) and its
    arms, each ended by [;;], [;&] or [;;&]. Reserved words are matched raw and only in command
    position: first in a statement, after [!]/[time] for the openers, or right after a compound's
    closer. A statement ends at [;], a lone [&], an arm terminator, or a newline that no
    [&&]/[||]/[|] carries over; its operands are its [&&]/[||]/[|] elements, and a compound is one
    operand of the statement it sits in. Quotes, substitutions, [[[ ... ]]] (across lines too) and
    parentheses that open no command (arrays, [(( ))], process substitution) are text.

    Errexit, as a single may-be-on bit that flows through that tree in source order from off. A
    simple command turns it on or off through {!Shell_lexer.command_errexit}. Wherever bash could
    have errexit on, the reading has it on, so its imprecision can only refuse a live assertion
    (loud), never pass an inert one (silent):
    - turning errexit off counts only where it certainly runs in this shell: as the first operand of
      its statement, outside any pipeline or background job, in a brace group or branch the flow is
      in, and through {!Shell_lexer.command_errexit}'s certainty (no redirection, no [time], no
      [set]/[shopt] function shadowing the builtin). Turning it on counts wherever it may run: after
      [&&]/[||], and as a pipeline's last element ([shopt -s lastpipe] runs it here); in any other
      pipeline element, a subshell or a background job it changes nothing outside;
    - [if]: each condition starts where the previous one ended and each body where its condition
      ended; after [fi] errexit may be on if any body, or (without [else]) the last condition, may
      leave it on. [case]: each arm starts from the state before [case] ([;&]/[;;&] also from the
      arm falling into it); afterwards it may be on if any arm, or no arm, leaves it on. A loop body
      is read again from a head that may be on when one pass may leave it on, so a [set -e] late in
      a body reaches its earlier statements, and a loop whose body holds a [set -e] anywhere may
      leave errexit on ([break] can leave right after it);
    - inside a compound whose status a condition, a [!] or a following [&&]/[||] reads, bash ignores
      errexit for every command, so nothing there is judged under it;
    - a function body runs where it is called, which a text scan cannot see: it is entered with
      errexit on when the file may turn it on anywhere outside that body (a subshell included, where
      a function defined beside the [set -e] may run). And since the scan does not follow calls, a
      body that may hand errexit back ON to a caller that called it with errexit off -- any [set -e]
      in it, since [return] can leave right after one; so [enable() { set -e; }], and the
      [set +e ... set -e] save-and-restore that assumes its caller had it on -- makes every off
      state in the file untrustworthy: such a script is read with errexit on from its first line,
      and no [set +e] in it turns it off.

    Status consumers. A statement's status is consumed -- it is not an assertion errexit has to stop
    on -- when the statement is a condition ([if]/[elif]/[while]/[until]), when its own list
    consumes it (each arm's rule), or when it is the LAST statement of a context that hands its
    status on:
    - a function body: its status is the return value the call site weighs;
    - a subshell, whose nonzero exit is a failure of its own, unless a pipeline discards it or it
      runs in the background;
    - a brace group, [if] branch or [case] arm whose compound hands its status on in turn: as a
      non-last operand of an [&&]/[||] list, or as the last operand of the last statement of a
      context that hands its status on.

    The genre this guards is a status overwritten before it reaches its consumer, so two contexts
    hand nothing on: a loop body, whose last statement a later iteration overwrites, and a [case]
    arm ending in [;&]/[;;&], whose status the arm it falls into overwrites.

    {1 What it refuses}

    In a script that may turn errexit on somewhere -- elsewhere nothing is judged, so nothing rests
    on the structure -- loudly, as unsupported: a closer or continuation word that does not fit the
    innermost open construct, a [)] or [}] that closes nothing, a construct still open at the end of
    the file (a script the parse check above accepts is valid shell, so a mismatch means the reading
    lost the structure, and every judgement after it would be a guess).

    {1 What it deliberately does not read}

    Loud (a live assertion flagged): an errexit transition inside a branch is read as possibly not
    taken; a [set +e] behind [time] or a redirection is read as possibly not run, and a [set -e]
    ending a pipeline as possibly run here; a loop that ends by turning errexit off in its condition
    still reads as possibly leaving it on; a function body is judged under errexit even where every
    call runs with it off; a function that may turn errexit on discards every off state in the file,
    called or not; a loop body's last statement is not consumed even where the loop runs once, nor a
    fall-through arm's where no arm follows it at run time; the script's own last statement is not a
    consumer, since an EXIT trap can replace the exit status -- [|| exit 1] is the explicit
    spelling; a [rc=$?] or a bare [return] on the next statement is not one either -- [|| rc=$?] and
    [|| return 1] are; and a statement whose status a pipeline discards is flagged although a plain
    failure there is lost too.

    Silent (an inert assertion not flagged): errexit turned on by [eval], a sourced file, an
    invocation flag ([bash -e script]; a shebang flag is refused by {!Shebang} already), or a caller
    of a sourced library; a function's last statement where every call discards the function's
    status ([f | cat], [f &]) -- the scan does not follow calls, and takes the return value as read;
    commands inside substitutions; and what each arm declares of its own. *)
module Shell_context = struct
  module L = Shell_lexer

  type connector = And | Or | Pipe
  type kind = Group of [ `Paren | `Brace ] | Function of [ `Paren | `Brace ] | If | Loop | Case

  type statement = {
    id : int;
    line : int;  (** The physical line of the logical line it starts on. *)
    operands : operand list;
    connectors : connector list;  (** Between consecutive operands. *)
    async : bool;  (** Ended by a lone [&]. *)
  }

  and operand = {
    text : string;
        (** A simple command's text, or what precedes a compound's opener ([!], [time], a function
            head). *)
    compound : compound option;
  }

  and compound = { kind : kind; branches : branch list }

  and branch = {
    condition : bool;
    fallthrough : bool;  (** A [case] arm the previous arm may fall into ([;&], [;;&]). *)
    statements : statement list;
  }

  let describe = function
    | Group `Paren -> "a subshell"
    | Group `Brace -> "a brace group"
    | Function _ -> "a function body"
    | If -> "an `if`"
    | Loop -> "a loop"
    | Case -> "a `case`"

  (** What a [(] or [{] opens, judged by the operand text before it: a group when nothing precedes
      it but [!] or [time] ([-p], [--]); a function body after a function head in any spelling
      ([f()], [f ()], [function f], [function f()]); otherwise nothing ([arr=(], [<(]). *)
  let opener text =
    let rec drop = function
      | "!" :: rest -> drop rest
      | "time" :: "-p" :: "--" :: rest | "time" :: "--" :: rest -> drop rest
      | "time" :: "-p" :: rest -> drop rest
      | "time" :: rest -> drop rest
      | words -> words
    in
    (* A function name is any word that is not an assignment and needs no quoting. *)
    let name word =
      (not (String.is_empty word))
      && not (String.exists word ~f:(List.mem [ '='; '$'; '\''; '"'; '`'; '\\' ] ~equal:Char.equal))
    in
    match drop (L.shell_words text) with
    | [] -> `Group
    | [ head ] when Option.exists (String.chop_suffix head ~suffix:"()") ~f:name -> `Function_body
    | [ head; "()" ] when name head -> `Function_body
    | [ "function"; head; "()" ] when name head -> `Function_body
    | [ "function"; head ]
      when name head || Option.exists (String.chop_suffix head ~suffix:"()") ~f:name ->
        `Function_body
    | _ -> `No

  type phase =
    | Commands
    | Header of { mutable words : int; mutable ended : bool }
        (** [for]/[select] words before [do]; a [case] subject before [in]. *)
    | Pattern of { mutable started : bool }  (** A [case] pattern before its [)]. *)

  type frame = {
    kind : kind option;  (** [None]: the script's top level. *)
    opened_at : int;
    mutable phase : phase;
    mutable open_branch : (bool * bool) option;  (** Its [condition] and [fallthrough]. *)
    mutable statements : statement list;  (** Of the open branch, latest first. *)
    mutable branches : branch list;  (** Closed, latest first. *)
    mutable falls_through : bool;  (** The next arm is entered by fall-through. *)
    outer : operand list * connector list * int option * string;
        (** The enclosing statement so far, and the text before the opener. *)
  }

  (** The compound tree of a script's logical lines (see the module header), with a refusal for each
      structural mismatch. *)
  let structure ~refuse lines =
    let buffer = Buffer.create 4096 in
    let starts =
      Array.of_list_map lines ~f:(fun (number, line) ->
          let start = Buffer.length buffer in
          Buffer.add_string buffer line;
          Buffer.add_char buffer '\n';
          (start, number))
    in
    let text = Buffer.contents buffer in
    let length = String.length text in
    let line_of offset =
      let rec search low high =
        if high - low <= 1 then snd starts.(low)
        else
          let middle = (low + high) / 2 in
          if fst starts.(middle) <= offset then search middle high else search low middle
      in
      if Array.is_empty starts then 1 else search 0 (Array.length starts)
    in
    let sub start finish = String.sub text ~pos:start ~len:(finish - start) in
    let ids = ref 0 in
    let operands = ref [] and connectors = ref [] and operand_start = ref 0 in
    let statement_start = ref None and closed = ref None in
    let new_frame kind ~at ~phase ~open_branch ~prefix =
      {
        kind;
        opened_at = at;
        phase;
        open_branch;
        statements = [];
        branches = [];
        falls_through = false;
        outer = (!operands, !connectors, Some (Option.value !statement_start ~default:at), prefix);
      }
    in
    let frames =
      ref [ new_frame None ~at:0 ~phase:Commands ~open_branch:(Some (false, false)) ~prefix:"" ]
    in
    let top () = List.hd_exn !frames in
    let close_operand finish =
      let operand =
        match !closed with
        | Some (compound, prefix) -> { text = prefix; compound = Some compound }
        | None -> { text = String.strip (sub !operand_start finish); compound = None }
      in
      closed := None;
      operands := operand :: !operands
    in
    let end_statement ?(async = false) finish next =
      close_operand finish;
      (match (!connectors, !operands) with
      | [], [ { text = ""; compound = None } ] -> ()
      | _ ->
          Int.incr ids;
          let frame = top () in
          frame.statements <-
            {
              id = !ids;
              line = line_of (Option.value !statement_start ~default:finish);
              operands = List.rev !operands;
              connectors = List.rev !connectors;
              async;
            }
            :: frame.statements);
      operands := [];
      connectors := [];
      statement_start := None;
      operand_start := next
    in
    let connect finish connector next =
      close_operand finish;
      connectors := connector :: !connectors;
      operand_start := next
    in
    let close_branch frame =
      Option.iter frame.open_branch ~f:(fun (condition, fallthrough) ->
          frame.branches <-
            { condition; fallthrough; statements = List.rev frame.statements } :: frame.branches);
      frame.statements <- [];
      frame.open_branch <- None
    in
    let open_branch frame ~condition =
      close_branch frame;
      frame.open_branch <- Some (condition, frame.falls_through);
      frame.falls_through <- false
    in
    let push kind ~at ~next ~phase ~open_branch =
      let prefix = String.strip (sub !operand_start at) in
      frames := new_frame (Some kind) ~at ~phase ~open_branch ~prefix :: !frames;
      operands := [];
      connectors := [];
      statement_start := None;
      closed := None;
      operand_start := next
    in
    let pop at next =
      end_statement at next;
      let frame = top () in
      close_branch frame;
      frames := List.tl_exn !frames;
      let outer_operands, outer_connectors, outer_start, prefix = frame.outer in
      operands := outer_operands;
      connectors := outer_connectors;
      statement_start := outer_start;
      closed :=
        Some ({ kind = Option.value_exn frame.kind; branches = List.rev frame.branches }, prefix);
      operand_start := next
    in
    let mismatch at word =
      refuse (line_of at)
        (Printf.sprintf "`%s` does not close or continue the innermost open construct (%s)" word
           (Option.value_map (top ()).kind ~default:"the top level" ~f:describe))
    in
    let is_meta c =
      Char.is_whitespace c || List.mem [ ';'; '&'; '|'; '('; ')'; '<'; '>' ] c ~equal:Char.equal
    in
    let word_ends_at index = index >= length || is_meta text.[index] in
    let word_at index =
      let rec finish index =
        if index < length && not (is_meta text.[index]) then finish (index + 1) else index
      in
      sub index (finish index)
    in
    let starts_at index token = L.starts_at text ~pos:index token in
    (* Blank since the operand began: where a closer or continuation word is one. *)
    let blank index = String.is_empty (String.strip (sub !operand_start index)) in
    (* The last index that a word continues past -- an escaped character, a substitution's closer --
       so the character after it starts no word: [foo\ #bar] and [$(x)#y] hold no comment. *)
    let glued_at = ref (-1) in
    let rec loop index quote escaped parens dbracket =
      if index < length then
        let c = text.[index] in
        let continue ?(quote = quote) ?(escaped = false) ?(parens = parens) ?(dbracket = dbracket)
            next =
          loop next quote escaped parens dbracket
        in
        match quote with
        | `Single -> continue ?quote:(Option.some_if (Char.equal c '\'') `None) (index + 1)
        | `Ansi_c ->
            if escaped then continue (index + 1)
            else if Char.equal c '\\' then continue ~escaped:true (index + 1)
            else continue ?quote:(Option.some_if (Char.equal c '\'') `None) (index + 1)
        | `Double ->
            if escaped then continue (index + 1)
            else if Char.equal c '\\' then continue ~escaped:true (index + 1)
            else if starts_at index "$(" then
              continue (L.skip_command_substitution text (index + 2))
            else if starts_at index "${" then continue (L.skip_parameter_expansion text (index + 2))
            else if Char.equal c '`' then continue ~quote:`Backtick_double (index + 1)
            else continue ?quote:(Option.some_if (Char.equal c '"') `None) (index + 1)
        | (`Backtick_none | `Backtick_double) as backtick ->
            if escaped then continue (index + 1)
            else if Char.equal c '\\' then continue ~escaped:true (index + 1)
            else if Char.equal c '`' then
              continue
                ~quote:(match backtick with `Backtick_double -> `Double | `Backtick_none -> `None)
                (index + 1)
            else continue (index + 1)
        | `None -> (
            let frame = top () in
            let word_start =
              index = 0
              || !glued_at <> index - 1
                 && (Char.is_whitespace text.[index - 1]
                    || List.mem [ ';'; '&'; '|'; '('; ')' ] text.[index - 1] ~equal:Char.equal)
            in
            let mark () =
              if not (Char.is_whitespace c) then
                match frame.phase with
                | Commands -> if Option.is_none !statement_start then statement_start := Some index
                | Pattern pattern -> pattern.started <- true
                | Header header -> if word_start then header.words <- header.words + 1
            in
            (* What a [(] or [{] here would open; nothing right after a compound's closer. *)
            let opening () =
              if Option.is_some !closed then `No else opener (sub !operand_start index)
            in
            (* Words, quotes and substitutions in every phase; command structure in [Commands]. *)
            let rec text_step () =
              mark ();
              let glued finish =
                glued_at := finish - 1;
                continue finish
              in
              if starts_at index "$'" then continue ~quote:`Ansi_c (index + 2)
              else if starts_at index "$(" then glued (L.skip_command_substitution text (index + 2))
              else if starts_at index "${" then glued (L.skip_parameter_expansion text (index + 2))
              else if Char.equal c '\'' then continue ~quote:`Single (index + 1)
              else if Char.equal c '"' then continue ~quote:`Double (index + 1)
              else if Char.equal c '`' then continue ~quote:`Backtick_none (index + 1)
              else if dbracket then
                if word_start && starts_at index "]]" && word_ends_at (index + 2) then
                  continue ~dbracket:false (index + 2)
                else continue (index + 1)
              else if parens > 0 then
                if Char.equal c '(' then continue ~parens:(parens + 1) (index + 1)
                else if Char.equal c ')' then continue ~parens:(parens - 1) (index + 1)
                else continue (index + 1)
              else
                match frame.phase with
                | Header _ | Pattern _ ->
                    if Char.equal c '(' then continue ~parens:1 (index + 1) else continue (index + 1)
                | Commands -> command_step ()
            and command_step () =
              if word_start && starts_at index "[[" && word_ends_at (index + 2) then
                continue ~dbracket:true (index + 2)
              else if Char.equal c '(' then
                let rec skip_blank index =
                  if index < length && List.mem [ ' '; '\t' ] text.[index] ~equal:Char.equal then
                    skip_blank (index + 1)
                  else index
                in
                let after = skip_blank (index + 1) in
                if after < length && Char.equal text.[after] ')' then
                  (* A function head's [()]: the body is what opens next. *)
                  continue (after + 1)
                else
                  match opening () with
                  | `Group when not (starts_at index "((") ->
                      push (Group `Paren) ~at:index ~next:(index + 1) ~phase:Commands
                        ~open_branch:(Some (false, false));
                      continue (index + 1)
                  | `Function_body ->
                      push (Function `Paren) ~at:index ~next:(index + 1) ~phase:Commands
                        ~open_branch:(Some (false, false));
                      continue (index + 1)
                  | `Group | `No -> continue ~parens:1 (index + 1)
              else if Char.equal c ')' then (
                (match frame.kind with
                | Some (Group `Paren | Function `Paren) -> pop index (index + 1)
                | _ ->
                    refuse (line_of index) "a `)` that closes no subshell or case pattern";
                    end_statement index (index + 1));
                continue (index + 1))
              else
                match if word_start then keyword (word_at index) else None with
                | Some next -> continue next
                | None ->
                    if Char.equal c ';' then (
                      match frame.kind with
                      | Some Case when starts_at index ";;" || starts_at index ";&" ->
                          let width = if starts_at index ";;&" then 3 else 2 in
                          end_statement index (index + width);
                          close_branch frame;
                          frame.phase <- Pattern { started = false };
                          frame.falls_through <- not (width = 2 && starts_at index ";;");
                          continue (index + width)
                      | _ ->
                          end_statement index (index + 1);
                          continue (index + 1))
                    else if starts_at index "&&" then (
                      connect index And (index + 2);
                      continue (index + 2))
                    else if starts_at index "||" then (
                      connect index Or (index + 2);
                      continue (index + 2))
                    else if starts_at index "|&" then (
                      connect index Pipe (index + 2);
                      continue (index + 2))
                    else if Char.equal c '|' && not (index > 0 && Char.equal text.[index - 1] '>')
                    then (
                      connect index Pipe (index + 1);
                      continue (index + 1))
                    else if
                      Char.equal c '&'
                      && (not
                            (index > 0 && List.mem [ '>'; '<' ] text.[index - 1] ~equal:Char.equal))
                      && not (starts_at index "&>")
                    then (
                      end_statement ~async:true index (index + 1);
                      continue (index + 1))
                    else if Char.equal c '\n' then
                      if
                        ((not (List.is_empty !connectors)) && blank index && Option.is_none !closed)
                        || Poly.equal (opening ()) `Function_body
                      then
                        (* An operand still to come, or a function body on the next line. *)
                        continue (index + 1)
                      else (
                        end_statement index (index + 1);
                        continue (index + 1))
                    else continue (index + 1)
            (* A reserved word in command position: where the scan resumes after it, if it was
               one. *)
            and keyword word =
              let next = index + String.length word in
              let fits = blank index in
              let opens () = Poly.equal (opening ()) `Group in
              let continue_after () = Some next in
              let enter kind ~phase ~open_branch =
                push kind ~at:index ~next ~phase ~open_branch;
                continue_after ()
              in
              let switch ~condition =
                end_statement index next;
                open_branch frame ~condition;
                continue_after ()
              in
              let misfit () =
                mismatch index word;
                continue_after ()
              in
              match (word, frame.kind, frame.open_branch) with
              | "{", _, _ when opens () ->
                  enter (Group `Brace) ~phase:Commands ~open_branch:(Some (false, false))
              | "{", _, _ when Poly.equal (opening ()) `Function_body ->
                  enter (Function `Brace) ~phase:Commands ~open_branch:(Some (false, false))
              | "if", _, _ when opens () ->
                  enter If ~phase:Commands ~open_branch:(Some (true, false))
              | ("while" | "until"), _, _ when opens () ->
                  enter Loop ~phase:Commands ~open_branch:(Some (true, false))
              | ("for" | "select"), _, _ when opens () ->
                  enter Loop ~phase:(Header { words = 0; ended = false }) ~open_branch:None
              | "case", _, _ when opens () ->
                  enter Case ~phase:(Header { words = 0; ended = false }) ~open_branch:None
              | "}", Some (Group `Brace | Function `Brace), _ when fits ->
                  pop index next;
                  continue_after ()
              | "then", Some If, Some (true, _) when fits -> switch ~condition:false
              | "elif", Some If, Some (false, _) when fits -> switch ~condition:true
              | "else", Some If, Some (false, _) when fits -> switch ~condition:false
              | "do", Some Loop, Some (true, _) when fits -> switch ~condition:false
              | ("fi", Some If, Some (false, _) | "done", Some Loop, Some (false, _)) when fits ->
                  pop index next;
                  continue_after ()
              | "esac", Some Case, _ when fits ->
                  pop index next;
                  continue_after ()
              | ("}" | "then" | "elif" | "else" | "do" | "fi" | "done" | "esac"), _, _ when fits ->
                  misfit ()
              | _ -> None
            in
            if escaped then (
              glued_at := index;
              continue (index + 1))
            else if Char.equal c '\\' then (
              mark ();
              continue ~escaped:true (index + 1))
            else if Char.equal c '#' && word_start then
              continue (Option.value (String.index_from text index '\n') ~default:length)
            else if parens > 0 || dbracket then text_step ()
            else
              match frame.phase with
              | Commands -> text_step ()
              | Pattern pattern ->
                  (* An unquoted [;] cannot be part of a pattern; it is the separator
                     {!Shell_lexer.numbered_spliced_lines} adds when it joins a condition header
                     that holds a [case]. *)
                  if Char.is_whitespace c || List.mem [ '|'; ';' ] c ~equal:Char.equal then
                    continue (index + 1)
                  else if Char.equal c '(' && not pattern.started then (
                    pattern.started <- true;
                    continue (index + 1))
                  else if Char.equal c ')' then (
                    frame.phase <- Commands;
                    open_branch frame ~condition:false;
                    operand_start := index + 1;
                    continue (index + 1))
                  else if word_start && (not pattern.started) && String.equal (word_at index) "esac"
                  then (
                    pop index (index + 4);
                    continue (index + 4))
                  else text_step ()
              | Header header ->
                  let word = if word_start then word_at index else "" in
                  let loop_header = match frame.kind with Some Loop -> true | _ -> false in
                  if Char.equal c '\n' || (loop_header && Char.equal c ';') then (
                    header.ended <- true;
                    continue (index + 1))
                  else if Char.is_whitespace c then continue (index + 1)
                  else if loop_header && String.equal word "do" && (header.ended || header.words = 1)
                  then (
                    frame.phase <- Commands;
                    open_branch frame ~condition:false;
                    operand_start := index + 2;
                    continue (index + 2))
                  else if (not loop_header) && String.equal word "in" && header.words = 1 then (
                    frame.phase <- Pattern { started = false };
                    continue (index + 2))
                  else text_step ())
    in
    loop 0 `None false 0 false;
    end_statement length length;
    while List.length !frames > 1 do
      let frame = top () in
      refuse (line_of frame.opened_at)
        (Printf.sprintf "%s still open at the end of the file"
           (Option.value_map frame.kind ~default:"" ~f:describe));
      pop length length
    done;
    let top = top () in
    close_branch top;
    match top.branches with
    | [ branch ] -> branch
    | _ -> { condition = false; fallthrough = false; statements = [] }

  let is_pipe = function Pipe -> true | And | Or -> false

  (** Whether operand [index] of [statement] runs in a shell of its own: a pipeline element, or part
      of a background job. *)
  let runs_apart statement index =
    statement.async
    || (index > 0 && is_pipe (List.nth_exn statement.connectors (index - 1)))
    || index < List.length statement.connectors
       && is_pipe (List.nth_exn statement.connectors index)

  (** Every command that may turn errexit on anywhere -- in a subshell or pipeline too, where a
      function defined beside it may run -- with the function bodies it sits in. *)
  let rec enabling_sites ~functions sites (branch : branch) =
    List.fold branch.statements ~init:sites ~f:(fun sites statement ->
        List.fold statement.operands ~init:sites ~f:(fun sites operand ->
            match operand.compound with
            | None -> (
                match L.command_errexit operand.text with
                | Some true -> functions :: sites
                | Some false | None -> sites)
            | Some ({ kind; branches } as compound) ->
                let functions =
                  match kind with Function _ -> compound :: functions | _ -> functions
                in
                List.fold branches ~init:sites ~f:(enabling_sites ~functions)))

  (** The names of the functions [branch] defines anywhere. *)
  let rec defined_functions (branch : branch) =
    List.concat_map branch.statements ~f:(fun statement ->
        List.concat_map statement.operands ~f:(fun operand ->
            match operand.compound with
            | None -> []
            | Some compound ->
                let own =
                  match (compound.kind, L.shell_words operand.text) with
                  | ( Function _,
                      ([ head ] | [ "function"; head ] | [ head; "()" ] | [ "function"; head; "()" ])
                    ) ->
                      [ Option.value (String.chop_suffix head ~suffix:"()") ~default:head ]
                  | _ -> []
                in
                own @ List.concat_map compound.branches ~f:defined_functions))

  type judgement = {
    statement : statement;
    errexit : bool;  (** Errexit may be on where the statement runs. *)
    consumed : bool;  (** Its status reaches a condition or a context that hands it on. *)
  }

  (** Every statement of [top] with the errexit state it may run under and whether its status is
      consumed (see the module header), in source order; and whether the script may turn errexit on
      at all. *)
  let judge top =
    let sites = enabling_sites ~functions:[] [] top in
    let shadowed =
      List.filter (defined_functions top) ~f:(List.mem [ "set"; "shopt" ] ~equal:String.equal)
    in
    let errexit_of text = L.command_errexit ~shadowed text in
    (* A loop or function body holding any [set -e] may be left -- by [break], [continue] or
       [return] -- right after it, whatever its end state says. *)
    let may_enable branches =
      List.exists branches ~f:(fun b -> not (List.is_empty (enabling_sites ~functions:[] [] b)))
    in
    (* One reading of the tree. [leaky]: some function body may turn errexit on for a caller that
       had it off, so no off state can be trusted. Returns the judgements and whether a body was
       found to be leaky. *)
    let reading ~leaky =
      let judgements = Hashtbl.create (module Int) in
      let leak = ref false in
      (* Off while a function body's transfer is computed from an entry it is not judged under. *)
      let recording = ref true in
      let record statement ~errexit ~consumed =
        if !recording then
          Hashtbl.update judgements statement.id ~f:(function
            | None -> { statement; errexit; consumed }
            | Some previous -> { previous with errexit = previous.errexit || errexit })
      in
      (* [ignored]: inside a compound whose status a condition, a [!] or a following [&&]/[||]
         reads, where bash ignores errexit for every command (the module header). *)
      let rec branch (b : branch) ~entry ~hands_on ~ignored =
        let last = List.length b.statements - 1 in
        let ignored = ignored || b.condition in
        List.foldi b.statements ~init:entry ~f:(fun index errexit statement ->
            let consumed = b.condition || (hands_on && index = last) in
            record statement ~errexit:(errexit && not ignored) ~consumed;
            run statement ~errexit ~consumed ~ignored)
      and run statement ~errexit ~consumed ~ignored =
        List.foldi statement.operands ~init:errexit ~f:(fun index errexit operand ->
            let apart = runs_apart statement index in
            let after = List.nth statement.connectors index in
            (* A pipeline's last element may run in this shell ([shopt -s lastpipe]). *)
            let may_run_here =
              (not statement.async) && not (Option.value_map after ~default:false ~f:is_pipe)
            in
            match operand.compound with
            | None -> (
                match errexit_of operand.text with
                | Some true when may_run_here -> true
                | Some false when (not leaky) && (not apart) && index = 0 -> false
                | _ -> errexit)
            | Some compound -> (
                (* The compound's own status: a following [&&]/[||] reads it, a following [|]
                   discards it, and as the last operand it is the statement's. *)
                let status_consumed =
                  match after with
                  | Some (And | Or) -> true
                  | Some Pipe -> false
                  | None -> consumed && not statement.async
                in
                let discarded =
                  statement.async || Option.value_map after ~default:false ~f:is_pipe
                in
                let ignored =
                  ignored
                  || (match after with Some (And | Or) -> true | Some Pipe | None -> false)
                  || String.is_prefix operand.text ~prefix:"!"
                in
                let exit = enter compound ~entry:errexit ~status_consumed ~discarded ~ignored in
                match compound.kind with
                | (Group `Brace | If | Loop | Case) when apart && may_run_here -> errexit || exit
                | _ when apart -> errexit
                | Group `Paren | Function _ -> errexit
                | Group `Brace | If | Loop | Case -> if index = 0 then exit else errexit || exit))
      and enter compound ~entry ~status_consumed ~discarded ~ignored =
        let branch ?(ignored = ignored) b ~entry ~hands_on = branch b ~entry ~hands_on ~ignored in
        match compound.kind with
        | Group `Brace ->
            List.fold compound.branches ~init:entry ~f:(fun entry body ->
                branch body ~entry ~hands_on:status_consumed)
        | Group `Paren ->
            List.iter compound.branches ~f:(fun body ->
                ignore (branch body ~entry ~hands_on:(not discarded) : bool));
            entry
        | Function shape ->
            let entry' =
              List.exists sites ~f:(fun functions ->
                  not (List.mem functions compound ~equal:phys_equal))
            in
            List.iter compound.branches ~f:(fun body ->
                (* Where the body runs is unknown: neither the definition's context nor its ignoring
                   applies. *)
                ignore (branch body ~entry:(leaky || entry') ~hands_on:true ~ignored:false : bool);
                (* A call from a caller with errexit off: may the body hand it back on? *)
                match shape with
                | `Brace ->
                    let saved = !recording in
                    recording := false;
                    if branch body ~entry:false ~hands_on:true ~ignored:false || may_enable [ body ]
                    then leak := true;
                    recording := saved
                | `Paren -> ());
            entry
        | If ->
            let last_condition, exits, bodies, conditions =
              List.fold compound.branches ~init:(entry, [], 0, 0)
                ~f:(fun (state, exits, bodies, conditions) b ->
                  if b.condition then
                    (branch b ~entry:state ~hands_on:false, exits, bodies, conditions + 1)
                  else
                    ( state,
                      branch b ~entry:state ~hands_on:status_consumed :: exits,
                      bodies + 1,
                      conditions ))
            in
            List.exists exits ~f:Fn.id || (bodies <= conditions && last_condition)
        | Loop ->
            (* A body hands nothing on: a later iteration overwrites its last statement's status. *)
            let pass head =
              List.fold compound.branches ~init:(head, head) ~f:(fun (state, joined) b ->
                  let exit = branch b ~entry:state ~hands_on:false in
                  (exit, joined || exit))
            in
            let exit, joined = pass entry in
            let head = entry || exit in
            (if Bool.equal head entry then joined else snd (pass head))
            || may_enable compound.branches
        | Case ->
            (* An arm ending in [;&]/[;;&] hands nothing on: the arm it falls into overwrites its
               status. *)
            let rec arms previous joined = function
              | [] -> joined
              | arm :: rest ->
                  let start =
                    match previous with
                    | Some exit when arm.fallthrough -> entry || exit
                    | Some _ | None -> entry
                  in
                  let falls_on = match rest with next :: _ -> next.fallthrough | [] -> false in
                  let exit = branch arm ~entry:start ~hands_on:(status_consumed && not falls_on) in
                  arms (Some exit) (joined || exit) rest
            in
            arms None entry compound.branches
      in
      ignore (branch top ~entry:leaky ~hands_on:false ~ignored:false : bool);
      ( Hashtbl.data judgements
        |> List.sort ~compare:(fun a b -> Int.compare a.statement.id b.statement.id),
        !leak )
    in
    let _, leaky = reading ~leaky:false in
    (fst (reading ~leaky), not (List.is_empty sites))

  type t = {
    judgements : judgement list;
    lexical : (int * string) list;  (** {!Shell_lexer.numbered_spliced_lines}' refusals. *)
    refusals : (int * string) list;
        (** This module's, for a script that may turn errexit on: elsewhere nothing is judged, so
            nothing rests on the structure. *)
  }

  let read text =
    let lexical = ref [] and refusals = ref [] in
    let lines =
      L.numbered_spliced_lines text ~refuse:(fun line reason ->
          lexical := (line, reason) :: !lexical)
    in
    let refuse line reason = refusals := (line, reason) :: !refusals in
    let judgements, enables_errexit = judge (structure ~refuse lines) in
    let refusals =
      if enables_errexit then
        List.dedup_and_sort !refusals ~compare:(fun (line, reason) (line', reason') ->
            match Int.compare line line' with 0 -> String.compare reason reason' | order -> order)
      else []
    in
    { judgements; lexical = List.rev !lexical; refusals }

  (** The lines of the statements [shape] picks that run under errexit with their status consumed by
      nothing. *)
  let flagged t ~shape =
    List.filter_map t.judgements ~f:(fun { statement; errexit; consumed } ->
        Option.some_if (errexit && (not consumed) && shape statement) statement.line)
    |> List.dedup_and_sort ~compare:Int.compare

  let report ~fail ~rel t =
    List.iter t.lexical ~f:(fun (line, reason) ->
        fail (Printf.sprintf "%s:%d: shell lexical context is unsupported: %s" rel line reason));
    List.iter t.refusals ~f:(fun (line, reason) ->
        fail (Printf.sprintf "%s:%d: shell execution context is unsupported: %s" rel line reason))

  let controls () =
    List.iter
      [
        ("quoted", "cat <<'END'\nEN\\\nD\nEND\n");
        ("escaped", "cat <<E\\ND\nEN\\\nD\nEND\n");
        ("paired backslashes", "cat <<END\nEN\\\\\nD\nEND\n");
      ]
      ~f:(fun (name, text) ->
        let messages = ref [] in
        report ~fail:(fun message -> messages := message :: !messages) ~rel:"fixture.sh" (read text);
        Verdict.p_empty
          (Printf.sprintf "heredoc fixture %s preserves its supported body" name)
          ~over:[ text ] !messages);
    (* Whether [text] reaches the [expected] refusal, observed under its [format]. *)
    let refuses text ~expected ~format =
      let messages = ref [] in
      report ~fail:(fun message -> messages := message :: !messages) ~rel:"fixture.sh" (read text);
      let refused = List.mem !messages expected ~equal:String.equal in
      if refused then
        Test_utils.Refusal_control_manifest.observe_failure
          ~source:"test/operations/shell_scripts_parse.ml" ~format;
      refused
    in
    List.iter
      [
        ("cat <<END\ndata\n", 3, "unterminated outer heredoc");
        ("cat <<END", 1, "unterminated outer heredoc");
        ("cat <<$'END'\ndata\nEND\n", 1, "dollar-quoted heredoc delimiter");
        ("cat <<$\"END\"\ndata\nEND\n", 1, "dollar-quoted heredoc delimiter");
        ("cat <<'two\nlines'\ndata\n", 1, "multiline heredoc delimiter");
        ("set -e\ncat <<END\nEN\\\nD\n! probe\nEND\n", 3, "continued unquoted heredoc body");
        ("set -e\ncat <<EN\\\nD\nEN\\\nD\n! probe\nEND\n", 4, "continued unquoted heredoc body");
      ]
      ~f:(fun (text, line, reason) ->
        Verdict.pf "unsupported shell lexical fixture %s reaches its refusal" reason
          (refuses text
             ~expected:
               (Printf.sprintf "fixture.sh:%d: shell lexical context is unsupported: %s" line reason)
             ~format:"%s:%d: shell lexical context is unsupported: %s"));
    List.iter
      [
        ( "set -e\nfi\n",
          2,
          "`fi` does not close or continue the innermost open construct (the top level)" );
        ( "set -e\nif true; then\n  :\ndone\nfi\n",
          4,
          "`done` does not close or continue the innermost open construct (an `if`)" );
        ("set -e\necho x )\n", 2, "a `)` that closes no subshell or case pattern");
        ( "set -e\n}\n",
          2,
          "`}` does not close or continue the innermost open construct (the top level)" );
        ("set -e\nwhile true; do\n  :\n", 2, "a loop still open at the end of the file");
        ("set -e\ncase x in\n  x) : ;;\n", 2, "a `case` still open at the end of the file");
      ]
      ~f:(fun (text, line, reason) ->
        Verdict.pf "unsupported shell execution context %S reaches its refusal" text
          (refuses text
             ~expected:
               (Printf.sprintf "fixture.sh:%d: shell execution context is unsupported: %s" line
                  reason)
             ~format:"%s:%d: shell execution context is unsupported: %s"));
    (* The opposing control: without errexit there is nothing to judge, so nothing is refused. *)
    let messages = ref [] in
    report ~fail:(fun message -> messages := message :: !messages) ~rel:"fixture.sh" (read "fi\n");
    Verdict.p_empty "a script that never turns errexit on is not refused for its structure"
      ~over:[ "fi\n" ] !messages
end

(** A deliberately textual check for statement-position command negation in scripts that enable
    errexit (gh-ocannl-895).

    Bash exempts [! command] from errexit because the command's status is being inverted. With
    nothing consuming that inverted status, the spelling looks like a negative assertion but cannot
    stop a [set -e] harness. A statement is flagged when the pipeline whose status its list hands on
    -- the last one, after any [&&]/[||], since a connector reads each earlier one -- begins with a
    [!] word (also with a redirection glued to it, [!>file]), and {!Shell_context} judges that it
    may run under errexit with its status consumed by nothing else: not a condition, not the last
    statement of a function body or a subshell. A [!] inside a command substitution or a test
    bracket is not a statement's first word, so those shapes stay valid. Beyond {!Shell_context}'s
    own boundary nothing is excluded: the arm is the shape above. *)
module Errexit_negation = struct
  module C = Shell_context

  type finding = { line : int }

  (* The pipeline whose status the list hands on is its last one, after the last [&&]/[||]: an
     earlier [! cmd] is read by the connector after it. *)
  let negated (statement : C.statement) =
    let start =
      List.foldi statement.connectors ~init:0 ~f:(fun index start -> function
        | C.And | C.Or -> index + 1
        | C.Pipe -> start)
    in
    match List.drop statement.operands start with
    | { text; _ } :: _ ->
        String.is_prefix text ~prefix:"!"
        && (String.length text = 1
           || Char.is_whitespace text.[1]
           || List.mem [ '<'; '>' ] text.[1] ~equal:Char.equal)
    | [] -> false

  let findings text = List.map (C.flagged (C.read text) ~shape:negated) ~f:(fun line -> { line })

  let report ~fail ~rel context =
    List.iter (C.flagged context ~shape:negated) ~f:(fun line ->
        fail
          (Printf.sprintf
             "%s:%d: statement-position `! command` is inert under errexit; route the assertion \
              through an `absent()`-style helper whose body uses `if`"
             rel line))

  let cases =
    [
      ("multiline single quote", "set -e\nx='data\n! probe\n'\n", []);
      ("multiline double quote", "set -e\nx=\"data\n! probe\n\"\n", []);
      ("quoted heredoc", "set -e\ncat <<'END'\n! probe\nEND\n", []);
      ("tab-stripped heredoc", "set -e\ncat <<-END\n\t! probe\n\tEND\n", []);
      ("continued if condition", "set -e\nif\n! probe\nthen :; fi\n", []);
      ("continued while condition", "set -e\nwhile\n! probe\ndo :; done\n", []);
      ("continued until condition", "set -e\nuntil\n! probe\ndo :; done\n", []);
      ("negation after heredoc", "set -e\ncat <<END\n! data\nEND\n! probe\n", [ 5 ]);
      ("negation after quote", "set -e\nx='data\n! data\n'\n! probe\n", [ 5 ]);
      ("negation in if body", "set -e\nif\n! probe\nthen\n! assertion\nfi\n", [ 5 ]);
      ("multiple heredocs", "set -e\ncat <<A <<'B'\n! data\nA\n! data\nB\n! probe\n", [ 7 ]);
      ("escaped delimiter", "set -e\ncat <<E\\ND\n! data\nEND\n! probe\n", [ 5 ]);
      ("quoted body preserves continuation", "set -e\ncat <<'END'\nEN\\\nD\nEND\n! probe\n", [ 6 ]);
      ("escaped body preserves continuation", "set -e\ncat <<E\\ND\nEN\\\nD\nEND\n! probe\n", [ 6 ]);
      ( "unquoted body with paired backslashes",
        "set -e\ncat <<END\nEN\\\\\nD\nEND\n! probe\n",
        [ 6 ] );
      ( "double quote preserves ordinary backslash",
        "set -e\ncat <<\"E\\ND\"\n! data\nE\\ND\n! probe\n",
        [ 5 ] );
      ( "quoted delimiter has whitespace",
        "set -e\ncat <<'END HERE'\n! data\nEND HERE\n! probe\n",
        [ 5 ] );
      ( "continued condition with heredoc",
        "set -e\nif cat <<END\n! data\nthen\nEND\n! probe\nthen :; fi\n! assertion\n",
        [ 8 ] );
      ("heredoc set text is data", "cat <<END\nset -e\nEND\n! probe\n", []);
      ("quoted set text is data", "x='data\nset -e\n'\n! probe\n", []);
      ("here-string resumes code", "set -e\ncat <<<word\n! probe\n", [ 3 ]);
      ("arithmetic shift resumes code", "set -e\n((x<<1))\n! probe\n", [ 3 ]);
      ("real comment hides consumer", "set -e\n! probe # || recover\n", [ 2 ]);
      ("continued consumer", "set -e\n! probe \\\n || recover\n", []);
      ("hash after spliced word", "set -e\n! probe foo\\\n#bar || recover\n", []);
      ("comment after spliced whitespace", "set -e\n! probe \\\n# || recover\n", [ 2 ]);
      ("statement-position ! grep", "set -e\n! grep -q missing output\n", [ 2 ]);
      ("combined errexit option", "set -euo pipefail\n! grep -q missing output\n", [ 2 ]);
      ("named errexit option", "set -o errexit\n! grep -q missing output\n", [ 2 ]);
      ("punctuated named errexit option", "set -o errexit;\n! grep -q missing output\n", [ 2 ]);
      ("quoted named errexit option", "set -o 'errexit'\n! grep -q missing output\n", [ 2 ]);
      ("quoted short errexit option", "set \"-e\"\n! grep -q missing output\n", [ 2 ]);
      ("concatenated quoted errexit name", "set -o erre'x'it\n! grep -q missing output\n", [ 2 ]);
      ("concatenated quoted short option", "set \"-\"e\n! grep -q missing output\n", [ 2 ]);
      ("builtin set", "builtin set -e\n! grep -q missing output\n", [ 2 ]);
      ("command set", "command set -e\n! grep -q missing output\n", [ 2 ]);
      ("command -- set", "command -- set -e\n! grep -q missing output\n", [ 2 ]);
      ("command -p set", "command -p set -e\n! grep -q missing output\n", [ 2 ]);
      ("command -p -- set", "command -p -- set -e\n! grep -q missing output\n", [ 2 ]);
      ("builtin -- set", "builtin -- set -e\n! grep -q missing output\n", [ 2 ]);
      ("plus bundle before errexit", "set +u -e\n! grep -q missing output\n", [ 2 ]);
      ("named plus option before errexit", "set +o nounset -e\n! grep -q missing output\n", [ 2 ]);
      ("assignment-prefixed set", "X=y set -e\n! grep -q missing output\n", [ 2 ]);
      ("append-assignment-prefixed set", "X+=y set -e\n! grep -q missing output\n", [ 2 ]);
      ("quoted assignment-prefixed set", "X='a b' set -e\n! grep -q missing output\n", [ 2 ]);
      ("redirection-prefixed set", ">/dev/null set -e\n! grep -q missing output\n", [ 2 ]);
      ("separate redirection-prefixed set", "> /dev/null set -e\n! grep -q missing output\n", [ 2 ]);
      ("ampersand-redirection-prefixed set", "&>/dev/null set -e\n! grep -q missing output\n", [ 2 ]);
      ( "separate ampersand-redirection-prefixed set",
        "&> /dev/null set -e\n! grep -q missing output\n",
        [ 2 ] );
      ("quoted set command", "s'et' -e\n! grep -q missing output\n", [ 2 ]);
      ("ANSI-C octal errexit option", "set -$'\\145'\n! grep -q missing output\n", [ 2 ]);
      ("ANSI-C hexadecimal errexit option", "set $'-\\x65'\n! grep -q missing output\n", [ 2 ]);
      ("ANSI-C short Unicode errexit option", "set -$'\\u0065'\n! grep -q missing output\n", [ 2 ]);
      ( "ANSI-C long Unicode errexit option",
        "set -$'\\U00000065'\n! grep -q missing output\n",
        [ 2 ] );
      ( "command-substitution assignment prefix",
        "X=$(printf value) set -e\n! grep -q missing output\n",
        [ 2 ] );
      ( "double-quoted command-substitution assignment prefix",
        "X=\"$(printf \"%s\" \"some value\")\" set -e\n! grep -q missing output\n",
        [ 2 ] );
      ( "double-quoted parameter-expansion assignment prefix",
        "X=\"${x:-\"some value\"}\" set -e\n! grep -q missing output\n",
        [ 2 ] );
      ( "parameter-expansion assignment prefix",
        "X=${x:-some value} set -e\n! grep -q missing output\n",
        [ 2 ] );
      ("noclobber redirection-prefixed set", ">| output set -e\n! grep -q missing output\n", [ 2 ]);
      ( "attached noclobber redirection-prefixed set",
        ">|output set -e\n! grep -q missing output\n",
        [ 2 ] );
      ("later set after set", "set -u; set -e\n! grep -q missing output\n", [ 2 ]);
      ("later set after command", "prepare; set -e\n! grep -q missing output\n", [ 2 ]);
      ("later set after AND", "prepare && set -e\n! grep -q missing output\n", [ 2 ]);
      ("later set after OR", "prepare || set -e\n! grep -q missing output\n", [ 2 ]);
      ("later set after async command", "prepare & set -e\n! grep -q missing output\n", [ 2 ]);
      ("timed set", "time set -e\n! grep -q missing output\n", [ 2 ]);
      ("portable timed set", "time -p set -e\n! grep -q missing output\n", [ 2 ]);
      ("piped timed set does not affect parent", "time set -e | cat\n! grep -q missing output\n", []);
      ("pipeline set does not affect parent", "set -e | cat\n! grep -q missing output\n", []);
      ("pipe-and set does not affect parent", "set -e |& cat\n! grep -q missing output\n", []);
      ("background set does not affect parent", "set -e &\n! grep -q missing output\n", []);
      ( "set after pipeline affects parent",
        "set -u | cat; set -e\n! grep -q missing output\n",
        [ 2 ] );
      ("set inside brace group", "{ set -e; }\n! grep -q missing output\n", [ 2 ]);
      ("set inside later brace group", "prepare; { set -e; }\n! grep -q missing output\n", [ 2 ]);
      ("set inside parenthesized group", "( set -e; )\n! grep -q missing output\n", []);
      ("set inside piped brace group", "{ set -e; } | cat\n! grep -q missing output\n", []);
      ("set inside pipe-and brace group", "{ set -e; } |& cat\n! grep -q missing output\n", []);
      ("set inside background brace group", "{ set -e; } &\n! grep -q missing output\n", []);
      ( "set inside redirected piped brace group",
        "{ set -e; } >/dev/null | cat\n! grep -q missing output\n",
        [] );
      ( "set inside separately redirected piped brace group",
        "{ set -e; } > /dev/null | cat\n! grep -q missing output\n",
        [] );
      ("set in if condition", "if set -e; then :; fi\n! grep -q missing output\n", [ 2 ]);
      ("set in while condition", "while set -e; do :; done\n! grep -q missing output\n", [ 2 ]);
      ("set in then body", "if true; then set -e; fi\n! grep -q missing output\n", [ 2 ]);
      ("set in case arm", "case x in x) set -e;; esac\n! grep -q missing output\n", [ 2 ]);
      ("quoted set text", "printf '%s' 'set -e;'; set -u\n! grep -q missing output\n", []);
      ( "set text inside unquoted parameter expansion",
        "echo ${x:-foo; set -e}\n! grep -q missing output\n",
        [] );
      ( "set text inside unquoted command substitution",
        "echo $(printf x; set -e)\n! grep -q missing output\n",
        [] );
      ("if ! command", "set -e\nif ! grep -q missing output; then :; fi\n", []);
      ("while ! command", "set -e\nwhile ! ready; do :; done\n", []);
      ("until ! command", "set -e\nuntil ! ready; do :; done\n", []);
      ("! command || fallback", "set -e\n! grep -q missing output || recover\n", []);
      ("! command && continuation", "set -e\n! grep -q missing output && continue_run\n", []);
      ("command substitution", "set -e\nresult=$(! grep -q missing output)\n", []);
      ("single-bracket test", "set -e\n[ ! -f output ]\n", []);
      ("double-bracket test", "set -e\n[[ ! -f output ]]\n", []);
      ("negation without errexit", "set -u\n! grep -q missing output\n", []);
      ("quoted AND/OR text", "set -e\n! printf '%s\\n' 'not || consumed'\n", [ 2 ]);
      ("nested AND/OR group", "set -e\n! (probe || recover)\n", [ 2 ]);
      ("nested AND/OR substitution", "set -e\n! cmd $(probe || recover)\n", [ 2 ]);
      ("nested AND/OR backtick substitution", "set -e\n! cmd `probe || recover`\n", [ 2 ]);
      ( "nested AND/OR double-quoted backtick substitution",
        "set -e\n! true \"`printf \"%s\" \"x || y\"`\"\n",
        [ 2 ] );
      ( "nested quotes in double-quoted substitution",
        "set -e\n! true \"$(printf \"%s\" \"x || y\")\"\n",
        [ 2 ] );
      ("nested quotes in parameter expansion", "set -e\n! true \"${x:-\"a || b\"}\"\n", [ 2 ]);
      ("ANSI-C quoted AND/OR text", "set -e\n! true $'can\\'t || consume'\n", [ 2 ]);
      ("outer AND/OR after nested group", "set -e\n! (probe || recover) || fallback\n", []);
      ("later OR after semicolon", "set -e\n! probe; cleanup || recover\n", [ 2 ]);
      ("later OR after async terminator", "set -e\n! probe & cleanup || recover\n", [ 2 ]);
      ("outer OR after stderr redirect", "set -e\n! probe &>/dev/null || recover\n", []);
      ("outer OR after descriptor redirect", "set -e\n! probe 2>&1 || recover\n", []);
      ("outer OR after pipe-and", "set -e\n! probe |& sink || recover\n", []);
      ("nested AND/OR in if command", "set -e\n! if probe && ready; then :; fi\n", [ 2 ]);
      ("outer OR after if command", "set -e\n! if probe && ready; then :; fi || recover\n", []);
      ( "if keyword used as an argument",
        "set -e\n! if probe; then echo fi; ready && steady; fi\n",
        [ 2 ] );
      ("nested if command", "set -e\n! if if probe || recover; then :; fi; then :; fi\n", [ 2 ]);
      ("ordinary if argument before outer OR", "set -e\n! echo if || recover\n", []);
      ("nested AND/OR in while command", "set -e\n! while probe || recover; do :; done\n", [ 2 ]);
      ("nested AND/OR in case command", "set -e\n! case x in x) probe || recover ;; esac\n", [ 2 ]);
      ("bang-adjacent output redirect", "set -e\n!>/dev/null probe\n", [ 2 ]);
      ("bang-adjacent input redirect", "set -e\n!</dev/null probe\n", [ 2 ]);
      ("bang-prefixed command name", "set -e\n!probe\n", []);
      ( "errexit set after an embedded hash",
        "echo foo#bar; set -e\n! grep -q missing output\n",
        [ 2 ] );
      ("errexit set by a continued command", "set \\\n-e\n! grep -q missing output\n", [ 3 ]);
      ("errexit set through shopt", "shopt -s -o errexit\n! grep -q missing output\n", [ 2 ]);
      ("builtin takes no -p", "builtin -p set -e\n! grep -q missing output\n", []);
      ( "errexit set after a literal-bracket test",
        "[ [ = x ]; set -e\n! grep -q missing output\n",
        [ 2 ] );
      ("errexit set through bundled shopt", "shopt -so errexit\n! grep -q missing output\n", [ 2 ]);
      ("errexit unset through shopt", "shopt -u -o errexit\n! grep -q missing output\n", []);
      ( "errexit set behind a variable-descriptor redirection",
        "{fd}>/dev/null set -e\n! grep -q missing output\n",
        [ 2 ] );
      ("embedded hash before an outer OR", "set -e\n! grep -q x#y output || recover\n", []);
      (* Execution context: any statement position, option transitions, status consumers. *)
      ("negation after a semicolon", "set -e\nprepare; ! probe\n", [ 2 ]);
      ("negation on a then line", "set -e\nif x; then ! probe; fi\n", [ 2 ]);
      ("negation in a brace group", "set -e\n{ ! probe; }\n", [ 2 ]);
      ("negation before a function's last statement", "set -e\nf() {\n  ! probe\n  :\n}\n", [ 3 ]);
      ("function's final negation", "set -e\nabsent() {\n  ! grep -q x f\n}\n", []);
      ("subshell's final negation", "set -e\n( ! probe )\n", []);
      ("negation before set -e", "! probe\nset -e\n", []);
      ("negation after set +e", "set -e\nset +e\n! probe\n", []);
      ("negation after set +e and set -e", "set -e\nset +e\nset -e\n! probe\n", [ 4 ]);
      ("negation after a conditional set +e", "set -e\nif x; then set +e; fi\n! probe\n", [ 3 ]);
      ("negation after a guarded set +e", "set -e\nx && set +e\n! probe\n", [ 3 ]);
      ("negation after a subshell set +e", "set -e\n( set +e )\n! probe\n", [ 3 ]);
      ("negation after set -e +e", "set -e +e\n! probe\n", []);
      ("negation after shopt -u -o errexit", "set -e\nshopt -u -o errexit\n! probe\n", [ 3 ]);
      (* A status overwritten before it reaches its consumer, and a call that turns errexit on. *)
      ( "function's final negation in a loop body",
        "set -e\nf() {\n  for x in y; do\n    ! probe\n  done\n}\n",
        [ 4 ] );
      ( "function's final negation in a fall-through arm",
        "set -e\nf() {\n  case $1 in\n    a) ! probe ;&\n    b) : ;;\n  esac\n}\n",
        [ 4 ] );
      ( "function's final negation in the last arm, ending in ;&",
        "set -e\nf() {\n  case $1 in\n    a) : ;;\n    b) ! probe ;&\n  esac\n}\n",
        [] );
      ( "negation after a call that turns errexit on",
        "set -e\nf() { set -e; }\nset +e\nf\n! probe\n",
        [ 5 ] );
      ( "negation after a builtin shadowed by a function",
        "set -e\nbuiltin() { :; }\nbuiltin set +e\n! probe\n",
        [ 4 ] );
      ("negation after an assignment and a ! word", "set -e\nX=y ! set +e || :\n! probe\n", [ 3 ]);
      ("negation after set + an expansion", "set -e\nset +\"$e\" || :\n! probe\n", [ 3 ]);
      ("negation as the last operand of an AND list", "set -e\nprepare && ! probe\n", [ 2 ]);
      ("negation as the last operand of an OR list", "set -e\nprepare || ! probe\n", [ 2 ]);
      ("negation after set -e - +e", "set -e - +e\n! probe\n", [ 2 ]);
      ("negation after a conflicting shopt", "set -e\nshopt -s -u -o errexit || :\n! probe\n", [ 3 ]);
      ( "negation after a double-quoted backslash name",
        "set -e\n\"s\\et\" +e || :\n! probe\n",
        [ 3 ] );
      ( "negation after a set -e ending a pipeline",
        "shopt -s lastpipe\n: | set -e\n! probe\n",
        [ 3 ] );
      ("negation in a group an || tail reads", "set -e\n{ ! probe; x; } || recover\n", []);
      ("negation after a redirected set +e", "set -e\nset +e 2>/dev/null\n! probe\n", [ 3 ]);
      ( "negation after a builtin set +e beside a set function",
        "set -e\nset() { :; }\nbuiltin set +e\n! probe\n",
        [ 4 ] );
      ( "negation after set +e in a file with a save-and-restore function",
        "set -e\nf() {\n  set +e\n  x\n  set -e\n}\nset +e\n! probe\n",
        [ 8 ] );
    ]

  let controls () =
    List.iter cases ~f:(fun (name, text, expected) ->
        let actual = List.map (findings text) ~f:(fun finding -> finding.line) in
        if not (List.equal Int.equal actual expected) then
          eprintf "errexit-negation case %s: expected lines %s, found %s\n" name
            (String.concat ~sep:"," (List.map expected ~f:Int.to_string))
            (String.concat ~sep:"," (List.map actual ~f:Int.to_string));
        Verdict.pf "errexit-negation fixture %s is classified as specified" name
          (List.equal Int.equal actual expected));
    let messages = ref [] in
    report
      ~fail:(fun message -> messages := message :: !messages)
      ~rel:"fixture.sh"
      (C.read "set -e\n! grep -q missing output\n");
    let refused =
      List.equal String.equal !messages
        [
          "fixture.sh:2: statement-position `! command` is inert under errexit; route the \
           assertion through an `absent()`-style helper whose body uses `if`";
        ]
    in
    if refused then
      Test_utils.Refusal_control_manifest.observe_failure
        ~source:"test/operations/shell_scripts_parse.ml"
        ~format:
          "%s:%d: statement-position `! command` is inert under errexit; route the assertion \
           through an `absent()`-style helper whose body uses `if`";
    Verdict.p "the statement-position ! grep fixture reaches the absent()-style refusal" refused
end

(** The second member of the errexit-exempt family: a statement-position AND list of tests,
    [[ A ] && [ B ]], in a script that enables errexit (gh-ocannl-1023).

    Bash exempts every command of an [&&]/[||] list except the last from errexit, so when [A] fails
    the list is simply false and the script carries on. Only [B] can stop the harness. The pair
    reads as two checks and performs one, and it goes wrong only when the LEFT test fails -- which
    for an assertion is usually the case it was written for: `cancel_sweep`'s readiness check in
    `test/operations/sweep_harness.sh` was silent in exactly the tick-budget case it guarded. Unlike
    [! cmd], which any positive control exposes, the pair passes every control that breaks its right
    operand.

    {1 What it reads}

    The statements {!Shell_context} reads, under its errexit and consumer judgement. A statement is
    flagged when all of these hold:
    - it is an [&&]-only list of at least two operands, and every operand is a test command
      ({!is_test}: [\[], [\[\[] or [test], also by path and through [command]/[builtin], after [!],
      [time], assignments and redirections); a compound operand is not one;
    - {!Shell_context} judges that it may run under errexit with nothing consuming its status: not a
      condition, and not the last statement of a function body or a subshell -- directly, or as the
      last statement of a group, branch or arm that is itself such a last statement.

    A [||] anywhere in the list is its consumer: [[ A ] && [ B ] || die ...] makes the failure
    explicit, so nothing is exempt. That is the spelling the refusal points at, together with one
    predicate per statement. It is also the spelling for capturing the status in an expected-error
    control: [[ A ] && [ B ] || rc=$?]. A [rc=$?] as the NEXT statement is refused like any other --
    whether a later expansion still sees the list's status is a lexer of its own, and each of its
    mistakes would be a silent pass.

    {1 What it deliberately does not read}

    Beyond {!Shell_context}'s own boundary, everything below is outside this arm. The LOUD items
    (valid shell refused) are accepted costs; the SILENT ones are named so a reader knows exactly
    what the scan cannot vouch for.
    - Lists that mix a non-test command in, such as [[ A ] && grep -q x f] or
      [grep -q x f && [ B ]]. [[ A ] && action] is the conditional-execution idiom and is correct;
      telling an assertion from an action for arbitrary commands is not a textual question.
    - Lists whose connectors mix [||] and [&&] (the [||] is read as the consumer), and pipelines.
    - Wrappers beyond the POSIX builtin-runners: a test run through [env], [nice], [exec], [sudo]
      and the like is not recognized as a test. (Silent; {!Errexit_execution_controls} measures the
      [env] case.)
    - Operators glued to the word before them without whitespace, other than a redirection's own
      descriptor ([2>], [{fd}>]): [command>/dev/null test -e a] keeps [command>/dev/null] as one
      word, so the operand is not recognized as a test. (Silent.) *)
module Errexit_and_list = struct
  module L = Shell_lexer
  module C = Shell_context

  type finding = { line : int }

  (** Whether [operand] is a test command: its command word is [\[], [\[\[] or [test] -- or a path
      naming one, [/bin/test] -- once everything that can stand in front of it is dropped: [!],
      [time]/[time -p], assignments, redirections ([{fd}>] too), and the wrappers that run a builtin
      as such ([command], [command -p], [builtin]); [time] and each wrapper with an optional [--].
      [command -v test] is not a test: [-v] is not a dropped option. *)
  let is_test operand =
    let literal = L.literal_shell_word in
    let is word expected = String.equal (literal word) expected in
    let rec drop = function
      | "!" :: rest -> drop rest
      | "time" :: "-p" :: rest | "time" :: rest -> drop (drop_raw_dashdash rest)
      | word :: rest when L.assignment_prefix word -> drop rest
      | word :: rest when Option.is_some (L.redirection_prefix word) -> (
          match L.redirection_prefix word with
          | Some false -> drop (List.drop rest 1)
          | _ -> drop rest)
      | word :: rest when is word "command" || is word "builtin" -> (
          match rest with
          | option :: rest when is option "-p" && is word "command" -> drop (drop_dashdash rest)
          | rest -> drop (drop_dashdash rest))
      | words -> words
    and drop_dashdash = function word :: rest when is word "--" -> rest | words -> words
    and drop_raw_dashdash = function "--" :: rest -> rest | words -> words in
    match drop (L.shell_words operand) with
    | word :: _ ->
        List.mem [ "["; "[["; "test" ] (Shebang.basename (literal word)) ~equal:String.equal
    | [] -> false

  let bare_test_list (statement : C.statement) =
    List.length statement.operands >= 2
    && List.for_all statement.connectors ~f:(function C.And -> true | C.Or | C.Pipe -> false)
    && List.for_all statement.operands ~f:(fun (operand : C.operand) ->
        Option.is_none operand.compound && is_test operand.text)

  let findings text =
    List.map (C.flagged (C.read text) ~shape:bare_test_list) ~f:(fun line -> { line })

  let report ~fail ~rel context =
    List.iter (C.flagged context ~shape:bare_test_list) ~f:(fun line ->
        fail
          (Printf.sprintf
             "%s:%d: statement-position `[ A ] && [ B ]` passes silently when its FIRST test fails \
              under errexit; write one predicate per statement, or end the list with `|| die ...` \
              / `|| return 1`"
             rel line))

  let cases =
    [
      ("pair in multiline quote", "set -e\nx='data\n[ -e a ] && [ -e b ]\n'\n", []);
      ("pair in heredoc", "set -e\ncat <<END\n[ -e a ] && [ -e b ]\nEND\n", []);
      ("pair in continued if", "set -e\nif\n[ -e a ] && [ -e b ]\nthen :; fi\n", []);
      ("pair in continued while", "set -e\nwhile\n[ -e a ] && [ -e b ]\ndo :; done\n", []);
      ("pair in continued until", "set -e\nuntil\n[ -e a ] && [ -e b ]\ndo :; done\n", []);
      ("pair after heredoc", "set -e\ncat <<END\ndata\nEND\n[ -e a ] && [ -e b ]\n", [ 5 ]);
      ("quote closes before pair", "set -e\nx='data\n' ; [ -e a ] && [ -e b ]\n", [ 2 ]);
      (* Flagged: the shape, and each place a body statement can sit. *)
      ("bare pair", "set -e\n[ -e ready ] && [ -e running ]\n", [ 2 ]);
      ( "the cancel_sweep readiness check",
        "set -e\n[ -e \"$p.ready\" ] && [ -e \"$p.ssh\" ]\n",
        [ 2 ] );
      ("double-bracket pair", "set -e\n[[ -e ready ]] && [[ -e running ]]\n", [ 2 ]);
      ("test-command pair", "set -e\ntest -e ready && test -e running\n", [ 2 ]);
      ("three tests", "set -e\n[ -n \"$a\" ] && [ -d \"$a\" ] && [ \"$a\" != / ]\n", [ 2 ]);
      ("negated left test", "set -e\n! [ -e ready ] && [ -e running ]\n", [ 2 ]);
      ("negated right test", "set -e\n[ -e ready ] && ! [ -e running ]\n", [ 2 ]);
      ("pair after semicolon", "set -e\nprepare; [ -e ready ] && [ -e running ]\n", [ 2 ]);
      ("pair before semicolon", "set -e\n[ -e ready ] && [ -e running ]; cleanup\n", [ 2 ]);
      ("pair before comment", "set -e\n[ -e ready ] && [ -e running ] # both up\n", [ 2 ]);
      ("pair in then branch", "set -e\nif probe; then [ -e a ] && [ -e b ]; fi\n", [ 2 ]);
      ("pair in else branch", "set -e\nif probe; then :; else [ -e a ] && [ -e b ]; fi\n", [ 2 ]);
      ("pair in loop body", "set -e\nfor f in x; do [ -e a ] && [ -e b ]; done\n", [ 2 ]);
      ("pair in brace group", "set -e\n{ [ -e a ] && [ -e b ]; }\n", [ 2 ]);
      ("pair opening a subshell", "set -e\n( [ -e a ] && [ -e b ]; cleanup )\n", [ 2 ]);
      ("pair in case arm", "set -e\ncase $x in\n  y) [ -e a ] && [ -e b ] ;;\nesac\n", [ 3 ]);
      ( "pair in multi-pattern case arm",
        "set -e\ncase $x in\n  y|z) [ -e a ] && [ -e b ] ;;\nesac\n",
        [ 3 ] );
      ("indented pair", "set -e\nf() {\n  [ -e a ] && [ -e b ]\n  cleanup\n}\n", [ 3 ]);
      ("pair wrapped after &&", "set -e\n[ -e a ] &&\n  [ -e b ]\ncleanup\n", [ 2 ]);
      ("pair wrapped by backslash", "set -e\n[ -e a ] \\\n  && [ -e b ]\ncleanup\n", [ 2 ]);
      ("pair wrapped after && and comment", "set -e\n[ -e a ] && # a first\n  [ -e b ]\n", [ 2 ]);
      ("literal opening-bracket argument", "set -e\n[ [ = x ] && [ -e b ]\n", [ 2 ]);
      ("literal closing-bracket argument", "set -e\n[ x = ] ] && [ -e b ]\n", [ 2 ]);
      ("timed subshell", "set -e\ntime ( [ -e a ] && [ -e b ]; : )\n", [ 2 ]);
      ("portable timed subshell", "set -e\ntime -p ( [ -e a ] && [ -e b ]; : )\n", [ 2 ]);
      ("negated subshell", "set -e\n! ( [ -e a ] && [ -e b ]; : )\n", []);
      ("group as a later operand", "set -e\nprobe && { [ -e a ] && [ -e b ]; y; }\n", [ 2 ]);
      ("subshell as a later operand", "set -e\nprobe || ( [ -e a ] && [ -e b ]; y )\n", [ 2 ]);
      ("group in a pipeline", "set -e\n{ [ -e a ] && [ -e b ]; y; } | cat\n", [ 2 ]);
      ("group nested in a subshell", "set -e\n( { [ -e a ] && [ -e b ]; y; } )\n", [ 2 ]);
      ( "group opened on the pair's line",
        "set -e\nprobe && {\n  [ -e a ] && [ -e b ]\n  y\n}\n",
        [ 3 ] );
      ( "double-bracket pair with a regex group",
        "set -e\n[[ $x =~ ^(a|b)$ ]] && [[ -e b ]]\n",
        [ 2 ] );
      ("quoted operator in a test", "set -e\n[ \"$x\" = '||' ] && [ -e b ]\n", [ 2 ]);
      ("pair after async command", "set -e\nserver & [ -e a ] && [ -e b ]\n", [ 2 ]);
      (* A function's last command is its return value, which the call site weighs. *)
      ("function's final pair", "set -e\nready() {\n  [ -e a ] && [ -e b ]\n}\n", []);
      ("one-line function body", "set -e\nready() { [ -e a ] && [ -e b ]; cleanup; }\n", [ 2 ]);
      ("spaced one-line function body", "set -e\nready () { [ -e a ] && [ -e b ]; x; }\n", [ 2 ]);
      ("function-keyword body", "set -e\nfunction ready { [ -e a ] && [ -e b ]; x; }\n", [ 2 ]);
      ( "function-keyword body with parens",
        "set -e\nfunction ready() { [ -e a ] && [ -e b ]; x; }\n",
        [ 2 ] );
      ("one-line function's final pair", "set -e\nready() { [ -e a ] && [ -e b ]; }\n", []);
      ( "one-line function's final pair, explicit",
        "set -e\nready() { [ -e a ] && [ -e b ] || return 1; }\n",
        [] );
      ("compact function head", "set -e\nready(){ [ -e a ] && [ -e b ]; cleanup; }\n", [ 2 ]);
      ( "compact function-keyword head",
        "set -e\nfunction ready(){ [ -e a ] && [ -e b ]; x; }\n",
        [ 2 ] );
      ("command-wrapped tests", "set -e\ncommand test -e a && command test -e b\n", [ 2 ]);
      ("builtin-wrapped tests", "set -e\nbuiltin [ -e a ] && builtin -- test -e b\n", [ 2 ]);
      ("command -p wrapped tests", "set -e\ncommand -p test -e a && command -- [ -e b ]\n", [ 2 ]);
      ("tests named by path", "set -e\n/bin/test -e a && /usr/bin/test -e b\n", [ 2 ]);
      ( "parenthesized case pattern on the case line",
        "set -e\ncase x in (x) [ -e a ] && [ -e b ]; echo y;; esac\n",
        [ 2 ] );
      ( "parenthesized case pattern on its own line",
        "set -e\ncase $x in\n  (a|b) [ -e a ] && [ -e b ] ;;\nesac\n",
        [ 3 ] );
      ("redirected subshell", "set -e\n( y; [ -e a ] && [ -e b ]; z ) 2>/dev/null\n", [ 2 ]);
      ("escaped space before a hash", "set -e\n[ \"$x\" = foo\\ #bar ] && [ -e b ]\n", [ 2 ]);
      ( "redirected parenthesized case arm",
        "set -e\ncase x in (x) >/dev/null [ -e a ] && [ -e b ]; echo y;; esac\n",
        [ 2 ] );
      ("separately redirected subshell", "set -e\n( y; [ -e a ] && [ -e b ]; z ) > out\n", [ 2 ]);
      ( "timed subshell with an option terminator",
        "set -e\ntime -- ( [ -e a ] && [ -e b ]; : )\n",
        [ 2 ] );
      ( "portably timed subshell with an option terminator",
        "set -e\ntime -p -- ( [ -e a ] && [ -e b ]; : )\n",
        [ 2 ] );
      ("subshell function body", "set -e\nf() ( [ -e a ] && [ -e b ]; : )\n", [ 2 ]);
      ("spaced subshell function body", "set -e\nf () ( [ -e a ] && [ -e b ]; : )\n", [ 2 ]);
      ("function-keyword subshell body", "set -e\nfunction f() ( [ -e a ] && [ -e b ]; : )\n", [ 2 ]);
      ("errexit set by a continued command", "set \\\n-e\n[ -e a ] && [ -e b ]\n", [ 3 ]);
      ( "errexit set behind a variable-descriptor redirection",
        "{fd}>/dev/null set -e\n[ -e a ] && [ -e b ]\n",
        [ 2 ] );
      ( "function-keyword subshell body without parens",
        "set -e\nfunction f ( [ -e a ] && [ -e b ]; : )\n",
        [ 2 ] );
      ("errexit set by a timed command", "time -- set -e\n[ -e a ] && [ -e b ]\n", [ 2 ]);
      ("errexit set by a portably timed command", "time -p -- set -e\n[ -e a ] && [ -e b ]\n", [ 2 ]);
      ( "pair after a spliced line keeps its line number",
        "set -e\necho a \\\n  b\n[ -e a ] && [ -e b ]\n",
        [ 4 ] );
      ( "double-bracket regex with a character class",
        "set -e\n[[ $x =~ [[:space:]] && -e a ]] && [[ -e b ]]\n",
        [ 2 ] );
      ( "errexit set after a literal-bracket test",
        "[ [ = x ]; set -e\n[ -e a ] && [ -e b ]\n",
        [ 2 ] );
      ( "parenthesized case arm whose body is a subshell",
        "set -e\ncase x in (x) ( [ -e a ] && [ -e b ]; : );; esac\n",
        [ 2 ] );
      ( "pair after a comment ending in a backslash",
        "set -e\n# note \\\n[ -e a ] && [ -e b ]\n",
        [ 3 ] );
      ( "pair after a single-quoted trailing backslash",
        "set -e\necho 'x\\'\n[ -e a ] && [ -e b ]\n",
        [ 3 ] );
      ("builtin takes no -p", "builtin -p set -e\n[ -e a ] && [ -e b ]\n", []);
      ("timed test", "set -e\ntime [ -e a ] && [ -e b ]\n", [ 2 ]);
      ("timed test with an option terminator", "set -e\ntime -- [ -e a ] && [ -e b ]\n", [ 2 ]);
      ( "portably timed test with an option terminator",
        "set -e\ntime -p -- [ -e a ] && time -- [ -e b ]\n",
        [ 2 ] );
      ( "variable-descriptor redirected tests",
        "set -e\n{fd}>/dev/null [ -e a ] && {fd2}> /dev/null [ -e b ]\n",
        [ 2 ] );
      ( "separated here-document redirections",
        "set -e\n<<- EOF [ -e a ] && <<- EOF2 [ -e b ]\n",
        [ 2 ] );
      ("errexit set after an embedded hash", "echo foo#bar; set -e\n[ -e a ] && [ -e b ]\n", [ 2 ]);
      ("redirected test", "set -e\n2>/dev/null [ -e a ] && [ -e b ]\n", [ 2 ]);
      ("assignment-prefixed test", "set -e\nLC_ALL=C [ a \\< b ] && [ -e b ]\n", [ 2 ]);
      ( "function's final pair, explicit",
        "set -e\nready() {\n  [ -e a ] && [ -e b ] || return 1\n}\n",
        [] );
      (* Not flagged: the list's value is consumed. *)
      ("or-die tail", "set -e\n[ -e a ] && [ -e b ] || die 'not ready'\n", []);
      ("or-exit tail", "set -e\n[ -e a ] && [ -e b ] || exit 2\n", []);
      ("or-brace tail", "set -e\n[ -e a ] && [ -e b ] || { echo no >&2; exit 2; }\n", []);
      ("or tail on the next line", "set -e\n[ -e a ] && [ -e b ] ||\n  die 'not ready'\n", []);
      ("if condition", "set -e\nif [ -e a ] && [ -e b ]; then :; fi\n", []);
      ("elif condition", "set -e\nif x; then :; elif [ -e a ] && [ -e b ]; then :; fi\n", []);
      ("while condition", "set -e\nwhile [ -e a ] && [ -e b ]; do :; done\n", []);
      ("until condition", "set -e\nuntil [ -e a ] && [ -e b ]; do :; done\n", []);
      ("wrapped if condition", "set -e\nif [ -e a ] &&\n   [ -e b ]; then :; fi\n", []);
      ( "condition keyword on an earlier line",
        "set -e\nif\n  [ -e a ] && [ -e b ]; then :; fi\n",
        [] );
      (* A status capture is not a consumer: `|| rc=$?` is the spelling that is. *)
      ("status captured on the same line", "set -e\n[ -e a ] && [ -e b ]; rc=$?\n", [ 2 ]);
      ("status captured on the next line", "set -e\n[ -e a ] && [ -e b ]\nrc=$?\n", [ 2 ]);
      ("status captured by an or-tail", "set -e\n[ -e a ] && [ -e b ] || rc=$?\n", []);
      (* Option transitions in source order. *)
      ( "status captured with errexit off",
        "set -e\nset +e\n[ -e a ] && [ -e b ]; rc=$?\nset -e\n",
        [] );
      ("pair before set -e", "[ -e a ] && [ -e b ]\nset -e\n", []);
      ("pair after set +e and set -e", "set -e\nset +e\nset -e\n[ -e a ] && [ -e b ]\n", [ 4 ]);
      ( "pair after set +e in a piped group",
        "set -e\n{ set +e; } | cat\n[ -e a ] && [ -e b ]\n",
        [ 3 ] );
      ( "pair in a loop before a later set -e",
        "for f in x y; do\n  [ -e a ] && [ -e b ]\n  set -e\ndone\n",
        [ 2 ] );
      ( "pair in a function whose file turns errexit off",
        "set -e\nf() {\n  [ -e a ] && [ -e b ]\n  :\n}\nset +e\n",
        [ 3 ] );
      (* More status consumers: the last statement of a subshell, and through compound commands. *)
      ("subshell's final pair", "set -e\n( [ -e a ] && [ -e b ] )\n", []);
      ("piped subshell's final pair", "set -e\n( [ -e a ] && [ -e b ] ) | cat\n", [ 2 ]);
      ( "function's final pair in an if branch",
        "set -e\nf() {\n  if x; then\n    [ -e a ] && [ -e b ]\n  fi\n}\n",
        [] );
      ( "function's final pair in a case arm",
        "set -e\nf() {\n  case $1 in\n    a) [ -e a ] && [ -e b ] ;;\n  esac\n}\n",
        [] );
      ( "function's final pair before break",
        "set -e\nf() {\n  for x in y; do\n    [ -e a ] && [ -e b ]\n    break\n  done\n}\n",
        [ 4 ] );
      ( "function's final group read by an or-tail",
        "set -e\nf() {\n  { [ -e a ] && [ -e b ]; } || return 1\n  :\n}\n",
        [] );
      ("top-level if branch's final pair", "set -e\nif x; then\n  [ -e a ] && [ -e b ]\nfi\n", [ 3 ]);
      ("script's final pair", "set -e\nprepare\n[ -e a ] && [ -e b ]\n", [ 3 ]);
      (* A status overwritten before it reaches its consumer, and a call that turns errexit on. *)
      ( "function's final pair in a loop body",
        "set -e\nf() {\n  for x in y; do\n    [ -e a ] && [ -e b ]\n  done\n}\n",
        [ 4 ] );
      ( "function's final pair in a fall-through arm",
        "set -e\nf() {\n  case $1 in\n    a) [ -e a ] && [ -e b ] ;&\n    b) : ;;\n  esac\n}\n",
        [ 4 ] );
      ( "function's final pair in a ;;& arm",
        "set -e\nf() {\n  case $1 in\n    a) [ -e a ] && [ -e b ] ;;&\n    *) : ;;\n  esac\n}\n",
        [ 4 ] );
      ( "function's final pair in the last arm, ending in ;&",
        "set -e\nf() {\n  case $1 in\n    a) : ;;\n    b) [ -e a ] && [ -e b ] ;&\n  esac\n}\n",
        [] );
      ( "pair after a call that turns errexit on",
        "set -e\nf() { set -e; }\nset +e\nf\n[ -e a ] && [ -e b ]\n",
        [ 5 ] );
      (* The shared reader carries the header to its keyword, and excludes heredoc data. *)
      ("condition before a do line", "set -e\nwhile\n  [ -e a ] && [ -e b ]\ndo :; done\n", []);
      ("condition before a then line", "set -e\nif\n  [ -e a ] && [ -e b ];\nthen :; fi\n", []);
      ( "heredoc data line reading as a keyword",
        "set -e\n[ -e a ] && [ -e b ] <<EOF\nthen\nEOF\n",
        [ 2 ] );
      ("quoted do after the pair", "set -e\n[ -e a ] && [ -e b ]; \"do\"\n", [ 2 ]);
      ("escaped then after the pair", "set -e\n[ -e a ] && [ -e b ]; \\then\n", [ 2 ]);
      (* Not flagged: outside the declared shape. *)
      ("command -v lookups", "set -e\ncommand -v test && command -v jq\n", []);
      ("conditional action", "set -e\n[ -n \"$a\" ] && echo \"$a\"\n", []);
      ("tests then an action", "set -e\n[ -e a ] && [ -e b ] && touch ready\n", []);
      ("single double-bracket test", "set -e\n[[ -e a && -e b ]]\n", []);
      ("double-bracket test with a group", "set -e\n[[ ( -e a || -e b ) && -e c ]]\n", []);
      ("subshell as an if condition", "set -e\nif ( [ -e a ] && [ -e b ] ); then :; fi\n", []);
      ("mixed or-and list", "set -e\n[ -e a ] || [ -e b ] && [ -e c ]\n", []);
      ("pair piped", "set -e\n[ -e a ] && [ -e b ] | cat\n", []);
      ("quoted pair", "set -e\necho '[ -e a ] && [ -e b ]'\n", []);
      ("pair in command substitution", "set -e\nok=$([ -e a ] && [ -e b ] && echo y)\n", []);
      ("assignment then a test", "set -e\nx=1 && [ -e b ]\n", []);
      ("pair without errexit", "set -u\n[ -e a ] && [ -e b ]\n", []);
    ]

  let controls () =
    List.iter cases ~f:(fun (name, text, expected) ->
        let actual = List.map (findings text) ~f:(fun finding -> finding.line) in
        if not (List.equal Int.equal actual expected) then
          eprintf "errexit-and-list case %s: expected lines %s, found %s\n" name
            (String.concat ~sep:"," (List.map expected ~f:Int.to_string))
            (String.concat ~sep:"," (List.map actual ~f:Int.to_string));
        Verdict.pf "errexit-and-list fixture %s is classified as specified" name
          (List.equal Int.equal actual expected));
    (* The negative control: a genuine bare pair reaches the refusal the repository scan issues. *)
    let messages = ref [] in
    report
      ~fail:(fun message -> messages := message :: !messages)
      ~rel:"fixture.sh"
      (C.read "set -e\n[ -e ready ] && [ -e running ]\n");
    let refused =
      List.equal String.equal !messages
        [
          "fixture.sh:2: statement-position `[ A ] && [ B ]` passes silently when its FIRST test \
           fails under errexit; write one predicate per statement, or end the list with `|| die \
           ...` / `|| return 1`";
        ]
    in
    if refused then
      Test_utils.Refusal_control_manifest.observe_failure
        ~source:"test/operations/shell_scripts_parse.ml"
        ~format:
          "%s:%d: statement-position `[ A ] && [ B ]` passes silently when its FIRST test fails \
           under errexit; write one predicate per statement, or end the list with `|| die ...` / \
           `|| return 1`";
    Verdict.p "the bare [ A ] && [ B ] fixture reaches the one-predicate-per-statement refusal"
      refused
end

(** {!Shell_context}'s contract, measured: each row places one assertion in a context, runs the
    script under the host's bash, and runs it again with a plain [false] in the assertion's place.
    The assertion is inert exactly when bash's outcome -- exit status and output -- differs from
    [false]'s: a failure that does not stop where [false] stops is one errexit lets past. A row
    declares what the scan does with it, and its claim holds only when bash agrees:
    - inert: flagged, and bash runs past the failure where [false] stops;
    - live: not flagged, and bash treats the failure exactly as it treats [false];
    - loud: flagged although bash treats it as [false] -- a declared over-approximation;
    - silent: not flagged although bash runs past it -- a declared blind spot;
    - refused: the script is refused as unsupported, and bash runs past it.

    These are the only scripts this file executes rather than parses, and they are this file's own
    literals: no repository script is ever run. Each row runs both arms' assertion ([! true] and
    [[ -n "" ] && [ -n x ]]) unless it names its own. The host decides the bash -- 3.2 on macOS, 5
    on Linux, Git for Windows' on Windows -- so a disagreement between versions surfaces as a failed
    claim on that platform rather than as a golden difference. *)
module Errexit_execution_controls = struct
  type arm = Negation | Pair
  type boundary = Inert | Live | Loud | Silent

  let default = function Negation -> "! true" | Pair -> "[ -n \"\" ] && [ -n x ]"

  let describe = function
    | Inert -> "flagged; bash runs past its failure where `false` stops"
    | Live -> "not flagged; bash treats its failure as it treats `false`"
    | Loud -> "flagged although bash treats its failure as `false` (declared loud)"
    | Silent -> "not flagged although bash runs past its failure (declared silent)"

  (** [;&] and [;;&], which bash 3.2 does not parse: a row that uses them states it, and is measured
      only where the host's bash runs this probe. *)
  let fallthrough = ("`;&` and `;;&`", "case x in x) : ;& y) : ;;& esac\n")

  (** [shopt -s lastpipe], which bash 3.2 lacks. *)
  let lastpipe = ("`lastpipe`", "shopt -s lastpipe\n")

  let both ?requires boundary name template =
    ( name,
      template,
      [ (Negation, default Negation, boundary); (Pair, default Pair, boundary) ],
      requires )

  let rows =
    [
      (* Option transitions, in source order. *)
      both Inert "after set -e" "set -e\n@@\necho SURVIVED\n";
      both Live "before set -e" "@@\nset -e\necho SURVIVED\n";
      both Live "after set +e" "set -e\nset +e\n@@\nset -e\necho SURVIVED\n";
      both Inert "after set +e and set -e again" "set -e\nset +e\nset -e\n@@\necho SURVIVED\n";
      both Live "after set +o errexit" "set -o errexit\nset +o errexit\n@@\necho SURVIVED\n";
      both Inert "after set +e in a subshell" "set -e\n( set +e )\n@@\necho SURVIVED\n";
      both Inert "after set +e in a pipeline" "set -e\nset +e | :\n@@\necho SURVIVED\n";
      both Inert "after set +e in a background job" "set -e\nset +e &\nwait\n@@\necho SURVIVED\n";
      both Live "after set +e in a brace group" "set -e\n{ set +e; }\n@@\necho SURVIVED\n";
      both Inert "after set +e on an untaken branch"
        "set -e\nif [ -n \"\" ]; then set +e; fi\n@@\necho SURVIVED\n";
      both Loud "after set +e on a taken branch"
        "set -e\nif [ -n x ]; then set +e; fi\n@@\necho SURVIVED\n";
      both Inert "after set +e behind a failed &&"
        "set -e\n[ -n \"\" ] && set +e\n@@\necho SURVIVED\n";
      both Loud "after set +e behind a passed &&" "set -e\n[ -n x ] && set +e\n@@\necho SURVIVED\n";
      both Inert "after set -e in a case arm" "case x in x) set -e ;; esac\n@@\necho SURVIVED\n";
      both Inert "in a loop body before a later set -e"
        "for i in 1 2; do\n\
        \  if [ \"$i\" = 2 ]; then\n\
        \    @@\n\
        \    echo REACHED\n\
        \  fi\n\
        \  set -e\n\
         done\n\
         echo SURVIVED\n";
      both Inert "after a function that turns errexit on" "f() { set -e; }\nf\n@@\necho SURVIVED\n";
      both Inert "after a function that turns errexit on, called after set +e"
        "set -e\nf() { set -e; }\nset +e\nf\n@@\necho SURVIVED\n";
      both Loud "after set +e, beside an uncalled function that turns errexit on"
        "set -e\nf() { set -e; }\nset +e\n@@\necho SURVIVED\n";
      both Silent "after eval set -e" "eval 'set -e'\n@@\necho SURVIVED\n";
      both Inert "after set -e - +e" "set -e - +e\n@@\necho SURVIVED\n";
      both Inert "after set -e with a positional before +e"
        "set -e positional +e\n@@\necho SURVIVED\n";
      both Inert "after set +e behind a failed redirection"
        "set -e\nset +e >/nonexistent-ocannl-dir/out || :\n@@\necho SURVIVED\n";
      both Inert "after set +e, shadowed by a function named set"
        "set -e\nset() { :; }\nset +e\n@@\necho SURVIVED\n";
      both Loud "after time set +e" "set -e\ntime set +e\n@@\necho SURVIVED\n";
      both Loud "after builtin set +e" "set -e\nbuiltin set +e\n@@\necho SURVIVED\n";
      both Loud "after shopt -u -o errexit" "set -e\nshopt -u -o errexit\n@@\necho SURVIVED\n";
      both Inert "after set + an expansion that is not e"
        "set -e\ne=x\nset +\"$e\" || :\n@@\necho SURVIVED\n";
      both ~requires:lastpipe Inert "after set -e ending a lastpipe pipeline"
        "shopt -s lastpipe\n: | set -e\n@@\necho SURVIVED\n";
      both Inert "after a loop left by break right after set -e"
        "for i in 1; do\n  set -e\n  break\n  set +e\ndone\n@@\necho SURVIVED\n";
      both Inert "in a function defined and called in one subshell"
        "(\n  set -e\n  f() {\n    @@\n    :\n  }\n  f\n  echo REACHED\n)\necho SURVIVED\n";
      (* Status consumers. *)
      both Live "as a function's last statement" "set -e\nf() {\n  @@\n}\nf\necho SURVIVED\n";
      both Inert "before a function's last statement"
        "set -e\nf() {\n  @@\n  :\n}\nf\necho SURVIVED\n";
      both Live "as a function's last if branch"
        "set -e\nf() {\n  if [ -n x ]; then\n    @@\n  fi\n}\nf\necho SURVIVED\n";
      both Loud "as a function's last loop body, run once"
        "set -e\nf() {\n  for i in 1; do\n    @@\n  done\n}\nf\necho SURVIVED\n";
      both Inert "in a function's last loop body, overwritten by a later iteration"
        "set -e\n\
         f() {\n\
        \  for i in 1 2; do\n\
        \    if [ \"$i\" = 1 ]; then\n\
        \      @@\n\
        \    else\n\
        \      :\n\
        \    fi\n\
        \  done\n\
         }\n\
         f\n\
         echo SURVIVED\n";
      both Inert "before a break in a function's last loop"
        "set -e\nf() {\n  for i in 1; do\n    @@\n    break\n  done\n}\nf\necho SURVIVED\n";
      both Live "as a function's last case arm"
        "set -e\nf() {\n  case x in\n    x) @@ ;;\n  esac\n}\nf\necho SURVIVED\n";
      both ~requires:fallthrough Inert "as a function's case arm falling through with ;&"
        "set -e\nf() {\n  case x in\n    x) @@ ;&\n    y) : ;;\n  esac\n}\nf\necho SURVIVED\n";
      both ~requires:fallthrough Inert "as a function's case arm continuing with ;;&"
        "set -e\nf() {\n  case x in\n    x) @@ ;;&\n    x) : ;;\n  esac\n}\nf\necho SURVIVED\n";
      both ~requires:fallthrough Live "as a function's last case arm, ending in ;&"
        "set -e\nf() {\n  case x in\n    y) : ;;\n    x) @@ ;&\n  esac\n}\nf\necho SURVIVED\n";
      both Live "as a function's last brace group"
        "set -e\nf() {\n  {\n    @@\n  }\n}\nf\necho SURVIVED\n";
      both Live "as a function's last && operand group"
        "set -e\nf() {\n  [ -n x ] && {\n    @@\n  }\n}\nf\necho SURVIVED\n";
      both Live "as a subshell's last statement" "set -e\n(\n  @@\n)\necho SURVIVED\n";
      both Live "as a subshell's last statement in a branch"
        "set -e\nif [ -n x ]; then\n  ( @@ )\nfi\necho SURVIVED\n";
      both Loud "as the last statement of a piped subshell" "set -e\n( @@ ) | cat\necho SURVIVED\n";
      both Inert "as a top-level if branch's last statement"
        "set -e\nif [ -n x ]; then\n  @@\nfi\necho SURVIVED\n";
      both Inert "as a top-level brace group's last statement" "set -e\n{\n  @@\n}\necho SURVIVED\n";
      both Loud "as the script's last statement" "set -e\n@@\n";
      both Loud "before a statement reading its status"
        "set -e\n@@\nrc=$?\n[ \"$rc\" -eq 0 ]\necho SURVIVED\n";
      both Live "as an if condition" "set -e\nif @@; then :; fi\necho SURVIVED\n";
      both Live "before an || tail" "set -e\n@@ || echo caught\necho SURVIVED\n";
      ( "as the last operand of an && list",
        "set -e\n[ -n x ] && @@\necho SURVIVED\n",
        [ (Negation, default Negation, Inert) ],
        None );
      both Live "in a group an || tail reads"
        "set -e\n{\n  @@\n  echo AFTER\n} || echo caught\necho SURVIVED\n";
      (* Pair only: the [!] in front is itself a negation statement, which the other arm flags. *)
      ( "in a negated subshell",
        "set -e\n! (\n  @@\n  echo AFTER\n)\necho SURVIVED\n",
        [ (Pair, default Pair, Live) ],
        None );
      both Loud "in a function only called with errexit off"
        "set -e\nf() {\n  @@\n  :\n}\nset +e\nf\necho SURVIVED\n";
      (* Wrappers. *)
      ( "through builtin-runners",
        "set -e\n@@\necho SURVIVED\n",
        [ (Pair, "command [ -n \"\" ] && builtin test -n x", Inert) ],
        None );
      ( "through time",
        "set -e\n@@\necho SURVIVED\n",
        [ (Pair, "time [ -n \"\" ] && [ -n x ]", Inert) ],
        None );
      ( "through env",
        "set -e\n@@\necho SURVIVED\n",
        [ (Pair, "env test -n \"\" && env test -n x", Silent) ],
        None );
    ]

  (** The exit status and standard output of [bash] running [text], with stdin and stderr closed to
      [/dev/null] and the startup variables {!cleared_variables} names cleared. *)
  let run bash text =
    let script = Stdlib.Filename.temp_file "ocannl_errexit_control" ".sh" in
    let output = Stdlib.Filename.temp_file "ocannl_errexit_control" ".out" in
    Exn.protect
      ~finally:(fun () ->
        List.iter [ script; output ] ~f:(fun path -> try Stdlib.Sys.remove path with _ -> ()))
      ~f:(fun () ->
        Out_channel.write_all script ~data:text;
        let devnull = if Stdlib.Sys.win32 then "NUL" else "/dev/null" in
        let out = Unix.openfile output [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
        let inp = Unix.openfile devnull [ Unix.O_RDONLY ] 0o400 in
        let err = Unix.openfile devnull [ Unix.O_WRONLY ] 0o200 in
        let status =
          Exn.protect
            ~finally:(fun () -> List.iter [ out; inp; err ] ~f:Unix.close)
            ~f:(fun () ->
              let pid =
                Unix.create_process_env bash [| bash; script |] (force isolated_environment) inp out
                  err
              in
              snd (Unix.waitpid [] pid))
        in
        (status, In_channel.read_all output))

  let controls () =
    (* Resolved, and refused when absent, exactly as a bash script's checker is. *)
    let rel = "errexit execution controls" in
    match resolve ~rel (Shebang.Via_env { env_path = ""; command = "bash" }) with
    | Error reason -> Verdict.fail (Printf.sprintf "%s: %s" rel reason)
    | Ok bash ->
        let version = snd (run bash "printf '%s' \"$BASH_VERSION\"\n") in
        eprintf "errexit execution controls run under `%s`, bash %s (not part of the golden)\n" bash
          version;
        List.iter rows ~f:(fun (name, template, legs, requires) ->
            let hole =
              1
              + String.count
                  (String.prefix template (String.substr_index_exn template ~pattern:"@@"))
                  ~f:(Char.equal '\n')
            in
            let fill assertion =
              String.substr_replace_all template ~pattern:"@@" ~with_:assertion
            in
            (* Where the host's bash lacks the syntax a row needs, its bash half cannot be measured;
               the arms' fixture tables pin the scan's half of such a row on every host. *)
            let measurable, lacking =
              match requires with
              | None -> (true, "")
              | Some (syntax, probe) -> (Poly.equal (fst (run bash probe)) (Unix.WEXITED 0), syntax)
            in
            let reference = run bash (fill "false") in
            List.iter legs ~f:(fun (arm, assertion, boundary) ->
                let text = fill assertion in
                let observed = run bash text in
                let differs = not (Poly.equal observed reference) in
                let lines =
                  match arm with
                  | Negation -> List.map (Errexit_negation.findings text) ~f:(fun f -> f.line)
                  | Pair -> List.map (Errexit_and_list.findings text) ~f:(fun f -> f.line)
                in
                let reading =
                  match ((Shell_context.read text).refusals, lines) with
                  | _ :: _, _ -> `Refused
                  | [], [] -> `Unflagged
                  | [], [ line ] when line = hole -> `Flagged
                  | [], _ :: _ -> `Flagged_elsewhere
                in
                (* What the scan must read, and whether bash must show the assertion inert. *)
                let expected, inert =
                  match boundary with
                  | Inert -> (`Flagged, true)
                  | Live -> (`Unflagged, false)
                  | Loud -> (`Flagged, false)
                  | Silent -> (`Unflagged, true)
                in
                let holds = Poly.equal reading expected && Bool.equal differs inert in
                if measurable && not holds then
                  eprintf
                    "errexit execution control %s with `%s`: flagged lines %s (hole %d), refused \
                     %b, bash %s `false`\n"
                    name assertion
                    (String.concat ~sep:"," (List.map lines ~f:Int.to_string))
                    hole (Poly.equal reading `Refused)
                    (if differs then "differs from" else "matches");
                Verdict.gated ~aggregation:`Environment ~when_:measurable
                  ~on:(Printf.sprintf "bash %s, which lacks %s" version lacking)
                  (Printf.sprintf "errexit execution control %s, `%s`: %s" name assertion
                     (describe boundary))
                  holds))
end

module Harness_contract = struct
  (* test-run.sh and test-harnesses.sh are production entrypoints, not fixture harnesses. Standalone
     harnesses outside the filename convention declare their lifecycle explicitly; Dune actions use
     the function-only API. *)
  let member path text =
    String.is_suffix path ~suffix:".sh"
    && (not (List.mem [ "tools/test-run.sh"; "tools/test-harnesses.sh" ] path ~equal:String.equal))
    && (String.is_prefix path ~prefix:"tools/test-"
       || String.is_prefix path ~prefix:"scripts/test-"
       || String.is_substring text ~substring:"# ocannl-harness: standalone\n")

  let compliant text =
    let lines = String.split_lines text |> List.map ~f:String.strip in
    let source =
      List.exists lines ~f:(fun line ->
          (String.is_prefix line ~prefix:". " || String.is_prefix line ~prefix:"source ")
          && String.is_substring line ~substring:"harness-support.sh")
    in
    (not (List.is_empty lines))
    && source
    && List.mem lines "harness_args \"$@\"" ~equal:String.equal
    && List.exists lines ~f:(fun line -> String.is_prefix line ~prefix:"harness_scratch ")
    && List.mem lines "finish" ~equal:String.equal
    && not
         (List.exists lines ~f:(fun line ->
              List.exists [ "report()"; "skip()"; "finish()"; "mutant()"; "expect_rejected()" ]
                ~f:(fun name ->
                  let compact = String.filter line ~f:(Fn.non Char.is_whitespace) in
                  String.is_prefix compact ~prefix:name
                  || String.is_prefix compact ~prefix:("function" ^ name)
                  || String.is_prefix compact ~prefix:("function" ^ String.drop_suffix name 2 ^ "{"))))

  let controls () =
    let correct =
      ". \"$HERE/harness-support.sh\"\nharness_args \"$@\"\nharness_scratch example\nfinish\n"
    in
    Verdict.p "shared harness source and lifecycle are accepted" (compliant correct);
    Verdict.p "new hand-run harnesses are discovered"
      (member "scripts/test-new.sh" ""
      && member "test/operations/new.sh" "# ocannl-harness: standalone\n"
      && not (member "test/operations/action.sh" ""));
    Verdict.p "production test-run is outside the harness family"
      ((not (member "tools/test-run.sh" "# ocannl-harness: standalone\n"))
      && (not (member "tools/test-harnesses.sh" ""))
      && member "tools/test-test-harnesses.sh" "");
    List.iter [ "report()"; "skip()"; "finish()"; "mutant()"; "expect_rejected()" ] ~f:(fun name ->
        List.iter
          [
            name ^ " { :; }";
            String.drop_suffix name 2 ^ " () { :; }";
            "function " ^ name ^ " { :; }";
            "function " ^ String.drop_suffix name 2 ^ " { :; }";
          ]
          ~f:(fun definition ->
            Verdict.pf "duplicate harness %s is refused" name
              (not (compliant (correct ^ definition ^ "\n")))));
    Verdict.p "a comment mentioning support cannot satisfy sourcing"
      (not
         (compliant "# harness-support.sh\nharness_args \"$@\"\nharness_scratch example\nfinish\n"))
end

let () =
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <workspace_root> <ocannl_config and shell scripts...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  (* The shebang grammar, on lines built to break it rather than on the two shapes this repository's
     scripts happen to use. Each case reads as what the parser must make of the line, so the golden
     is the specification. *)
  List.iter Shebang.shebang_cases ~f:(fun (line, expected) ->
      let actual = Shebang.render (Shebang.parse line) in
      if not (String.equal actual expected) then eprintf "shebang %S read as `%s`\n" line actual;
      Verdict.pf "shebang %S reads as `%s`" line expected (String.equal actual expected));
  (* The scope predicate, pinned separately from the parse table: a line can be one this check must
     report on precisely BECAUSE its arguments are refused, so the two questions do not answer each
     other. Phrased so that `true` is the passing reading either way. *)
  List.iter Shebang.scope_cases ~f:(fun (line, expected) ->
      let actual = Shebang.mentions_a_shell line in
      Verdict.pf "shebang %S is %s" line
        (if expected then "a shell script this check reports on" else "outside this check's scope")
        (Bool.equal actual expected));
  Shell_context.controls ();
  Errexit_negation.controls ();
  Errexit_and_list.controls ();
  Errexit_execution_controls.controls ();
  Harness_contract.controls ();
  let base = base_dir Stdlib.Sys.argv.(1) in
  (* Reported repository-relative, opened as dune handed them over: the working directory is deep in
     the build tree and the paths arrive relative to it. *)
  (* A `.sh` file is in scope by its name; anything else the globs hand over is in scope only if it
     actually starts with a shell shebang, which is how `tools/run-tests` gets covered without the
     rule depending on every file in the repository (round 6). Reading two bytes off each candidate
     is the price, over the few hundred files in `tools/` and `scripts/`. *)
  let in_scope path =
    String.is_suffix path ~suffix:".sh"
    || Shebang.mentions_a_shell (Option.value ~default:"" (first_line_of path))
  in
  let scripts =
    Array.to_list Stdlib.Sys.argv |> Fn.flip List.drop 2
    |> List.filter ~f:(fun path -> (not (Stdlib.Sys.is_directory path)) && in_scope path)
    |> List.map ~f:(fun path -> (repo_relative base path, path))
    (* The `*.sh` glob and the two directory globs overlap, so the same script arrives twice; one
       entry per repository-relative path, which is also what the golden is keyed by. *)
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  List.iter scripts ~f:(fun (rel, path) ->
      let text = In_channel.read_all path in
      if Harness_contract.member rel text then
        Verdict.pf "%s uses the shared harness contract" rel (Harness_contract.compliant text);
      let first_line = Option.value (first_line_of path) ~default:"" in
      let context = Shell_context.read text in
      Shell_context.report ~fail:Verdict.fail ~rel context;
      Errexit_negation.report ~fail:Verdict.fail ~rel context;
      Errexit_and_list.report ~fail:Verdict.fail ~rel context;
      match Shebang.parse first_line with
      | Error reason -> Verdict.fail (Printf.sprintf "%s: %s" rel reason)
      | Ok parsed ->
          let ours = match parsed with Shebang.Sourced -> true | Shebang.Interp _ -> false in
          let wanted =
            match parsed with
            | Shebang.Sourced ->
                [
                  Shebang.Via_env { env_path = ""; command = "sh" };
                  Shebang.Via_env { env_path = ""; command = "bash" };
                ]
            | Shebang.Interp launch -> [ launch ]
          in
          let resolutions = List.map wanted ~f:(resolve ~rel ~ours) in
          let usable = List.filter_map resolutions ~f:Result.ok in
          (* EVERY wanted checker has to resolve, not merely one of them (round 10). A sourced file
             is required to parse under both `sh` and `bash`, and reporting only when the whole list
             failed meant that a host missing `sh` and its stand-ins checked `tools/opam-env.sh`
             under bash alone -- silently, since the surviving checker kept `usable` non-empty and
             the golden line still read `parses: true`. A skipped grammar is not a passing one. *)
          if List.exists resolutions ~f:Result.is_error then
            List.iter resolutions ~f:(function
              | Error reason -> Verdict.fail (Printf.sprintf "%s: %s" rel reason)
              | Ok _ -> ())
          else
            let complaints =
              List.filter_map usable ~f:(fun prog ->
                  match parse_check prog path with
                  | None ->
                      Some (Printf.sprintf "`%s` disappeared between the probe and the check" prog)
                  | Some (Unix.WEXITED 0, _) -> None
                  | Some (_, said) -> Some (Printf.sprintf "`%s -n` said: %s" prog said))
            in
            eprintf "%s: parsed with %s\n" rel (String.concat ~sep:", " usable);
            List.iter complaints ~f:(fun complaint -> eprintf "  %s: %s\n" rel complaint);
            Verdict.p_empty (Printf.sprintf "%s parses" rel) ~over:usable complaints);
  eprintf "shell scripts scanned: %d\n" (List.length scripts);
  Verdict.pf "the scan reached at least the %d shell scripts this repository is known to have"
    script_floor
    (List.length scripts >= script_floor);
  let scanned = Set.of_list (module String) (List.map scripts ~f:fst) in
  let missing = List.filter must_be_scanned ~f:(Fn.non (Set.mem scanned)) in
  List.iter missing ~f:(fun rel -> eprintf "not reached by the scan: %s\n" rel);
  Verdict.p_empty "the scan reached the session hook and the suite runner" ~over:must_be_scanned
    missing;
  Test_utils.Refusal_control_manifest.print "shell_scripts_parse.ml"
