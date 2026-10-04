open Base
open Verdict.Claims

(* Use Git and the built reader only; no Python executable name is assumed. *)
let command program args =
  let output = Stdlib.Filename.temp_file "946-output-" "" in
  let errors = Stdlib.Filename.temp_file "946-errors-" "" in
  let read path =
    let channel = Stdlib.open_in_bin path in
    Exn.protect
      ~f:(fun () -> Stdlib.really_input_string channel (Stdlib.in_channel_length channel))
      ~finally:(fun () -> Stdlib.close_in channel)
  in
  Exn.protect
    ~f:(fun () ->
      let out_fd = Unix.openfile output [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
      let err_fd = Unix.openfile errors [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
      let pid =
        Exn.protect
          ~f:(fun () ->
            Unix.create_process program (Array.of_list (program :: args)) Unix.stdin out_fd err_fd)
          ~finally:(fun () ->
            Unix.close out_fd;
            Unix.close err_fd)
      in
      let _, status = Unix.waitpid [] pid in
      (status, read output ^ read errors))
    ~finally:(fun () ->
      Unix.unlink output;
      Unix.unlink errors)

let rec remove path =
  Unix.chmod path 0o700;
  if Stdlib.Sys.is_directory path then (
    Stdlib.Sys.readdir path |> Array.iter ~f:(fun child -> remove (path ^ "/" ^ child));
    Unix.rmdir path)
  else Unix.unlink path

let () =
  Stdlib.set_binary_mode_out Stdlib.stdout true;
  let reader =
    match Array.to_list Stdlib.Sys.argv with
    | [ _; reader ] ->
        if Stdlib.Filename.is_relative reader then Stdlib.Sys.getcwd () ^ "/" ^ reader else reader
    | _ -> failwith "expected the API-reader executable path"
  in
  let scratch = Stdlib.Filename.temp_file "946-api-drift-" "" in
  Unix.unlink scratch;
  Unix.mkdir scratch 0o700;
  let git_config = Stdlib.Filename.temp_file "946-git-config-" "" in
  let previous = Stdlib.Sys.getcwd () in
  Exn.protect
    ~f:(fun () ->
      Unix.chdir scratch;
      List.iter
        [
          ("GIT_CONFIG_NOSYSTEM", "1");
          ("GIT_CONFIG_GLOBAL", git_config);
          ("GIT_AUTHOR_NAME", "Fixture");
          ("GIT_AUTHOR_EMAIL", "fixture@example.invalid");
          ("GIT_COMMITTER_NAME", "Fixture");
          ("GIT_COMMITTER_EMAIL", "fixture@example.invalid");
        ]
        ~f:(fun (name, value) -> Unix.putenv name value);
      let git args =
        match command "git" args with
        | Unix.WEXITED 0, output -> String.strip output
        | _, output -> failwith ("fixture git failed: " ^ output)
      in
      let write path text =
        let rec mkdir directory =
          if not (Stdlib.Sys.file_exists directory) then (
            mkdir (Stdlib.Filename.dirname directory);
            Unix.mkdir directory 0o700)
        in
        mkdir (Stdlib.Filename.dirname path);
        let channel = Stdlib.open_out_bin path in
        Exn.protect
          ~f:(fun () -> Stdlib.output_string channel text)
          ~finally:(fun () -> Stdlib.close_out channel)
      in
      let commit subject =
        ignore (git [ "add"; "." ] : string);
        ignore (git [ "commit"; "-m"; subject ] : string);
        git [ "rev-parse"; "HEAD" ]
      in
      let read ?(until = "HEAD") ?(success = true) since =
        let status, output = command reader [ since; until ] in
        if Bool.equal (Poly.equal status (Unix.WEXITED 0)) success then output
        else failwith ("unexpected reader status: " ^ output)
      in
      let has report text = String.is_substring report ~substring:text in
      ignore (git [ "init"; "-b"; "master" ] : string);
      List.iter
        [
          ("arrayjit/lib/cap.mli", "type t = { old_scope : bool }\nval run :\n int -> int\n");
          ("lib/implicit.ml", "let public_value = 1\n");
          ("lib/hidden.ml", "let hidden = 1\n");
          ("lib/hidden.mli", "val public : int\n");
          ("bin/private.ml", "let private_value = 1\n");
          ( "arrayjit/lib/dune",
            "(library (name backend) (public_name pkg.backend) (modules impl) (libraries (select \
             impl.ml from (cuda -> impl.cuda.ml) (-> impl.missing.ml))))\n" );
          ("arrayjit/lib/impl.mli", "val hidden : int\n");
          ("arrayjit/lib/impl.cuda.ml", "let hidden = 1\n");
          ("arrayjit/lib/impl.missing.ml", "let hidden = 1\n");
          ( "tensor/dune",
            "(ocamllex lexer private_lexer)\n\
             (menhir (modules parser))\n\
             (library (name parserlib) (public_name pkg.parserlib) (modules lexer parser))\n\
             (executable (name private) (modules private_lexer))\n" );
          ("tensor/parser.mly", "%token OLD\n%%\n");
          ("tensor/lexer.mll", "{let exported = 1}\nrule token = parse | eof { () }\n");
          ("tensor/private_lexer.mll", "{let internal = 1}\nrule token = parse | eof { () }\n");
        ]
        ~f:(fun (path, text) -> write path text);
      let base = commit "base" in
      ignore (git [ "checkout"; "-b"; "feature" ] : string);
      List.iter
        [
          ("arrayjit/lib/cap.mli", "type t = { scopes : int list }\nval run :\n int -> string\n");
          ("lib/implicit.ml", "let public_value = \"new inferred type\"\n");
          ("lib/hidden.ml", "let hidden = \"not an export\"\n");
          ("bin/private.ml", "let private_value = \"also not public\"\n");
          ("arrayjit/lib/impl.cuda.ml", "let hidden = true\n");
          ("arrayjit/lib/impl.missing.ml", "let hidden = true\n");
          ("tensor/parser.mly", "%token NEW\n%%\n");
          ("tensor/lexer.mll", "{let exported = true}\nrule token = parse | eof { () }\n");
          ("tensor/private_lexer.mll", "{let internal = true}\nrule token = parse | eof { () }\n");
        ]
        ~f:(fun (path, text) -> write path text);
      let side = commit "feature implementation" in
      ignore (git [ "checkout"; "master" ] : string);
      ignore
        (git [ "merge"; "--no-ff"; "feature"; "-m"; "Merge pull request #624 from fixture/feature" ]
          : string);
      let merged = git [ "rev-parse"; "HEAD" ] in
      let report = read base in
      p "merge attribution, multiline signatures, implicit exports and exclusions"
        (has report ("commit " ^ merged ^ " Merge pull request #624")
        && (not (has report ("commit " ^ side)))
        && has report "old_scope" && has report "scopes" && has report "int -> string"
        && has report "public_value" && has report "new inferred type"
        && (not (has report "lib/hidden.ml"))
        && not (has report "bin/private.ml"));
      p "select interfaces exclude arm bodies; public generator input changes are visible"
        ((not (has report "impl.cuda.ml"))
        && (not (has report "impl.missing.ml"))
        && has report "tensor/parser.mly" && has report "%token NEW"
        && has report "tensor/lexer.mll" && has report "exported = true"
        && not (has report "private_lexer"));
      write "arrayjit/lib/cap.mli" "type t = { old_scope : bool }\nval run : int -> int\n";
      let reverted = commit "Revert API change (#625)" in
      let report = read base in
      p "reverted declarations retain both first-parent attribution points"
        (has report ("commit " ^ merged) && has report ("commit " ^ reverted));
      write "lib/implicit.mli" "val public_value : string\n";
      let narrowed = commit "Publish an explicit interface (#626)" in
      let report = read reverted in
      p "adding an interface records retirement of the implicit source surface"
        (has report "- let public_value" && has report "+ val public_value"
        && has report ("commit " ^ narrowed));
      ignore (git [ "rm"; "lib/implicit.mli" ] : string);
      let removed = commit "Remove an interface (#627)" in
      let report = read narrowed in
      p "removing an interface exposes the implementation surface"
        (has report "- val public_value" && has report "+ let public_value");
      write "arrayjit/lib/cap.mli"
        "(** prose only *)\ntype t = { old_scope : bool }\nval run :\n int -> int\n";
      let documented = commit "Document API (#628)" in
      let prose_only = has (read ~until:documented removed) "0 declaration changes" in
      write "arrayjit/lib/cap.mli"
        "[@@@ocaml.text \"floating docs\"]\ntype t = { old_scope : bool }\nval run : int -> int\n";
      let floating = commit "Floating documentation only" in
      p "documentation-only edits, empty windows and invalid endpoints"
        (prose_only
        && has (read ~until:floating documented) "0 declaration changes"
        && has (read ~success:false side) "first-parent history"
        && has (read ~success:false "missing-revision") "git failed"
        && has (read ~until:base base) "0 declaration changes across 0");
      ignore (git [ "rm"; "arrayjit/lib/impl.mli" ] : string);
      let exposed = commit "Expose selected implementation" in
      let report = read ~until:exposed floating in
      p "select arms become visible when the compiled target interface is removed"
        (has report "impl.cuda.ml" && has report "impl.missing.ml" && has report "hidden = true");
      write "lib/initializer.ml"
        "let value = 1\n\
         let () = print_endline \"old\";;\n\
         print_endline \"old bare\";;\n\
         let exported = 1 and () = print_endline \"old mixed\"\n";
      let initialized = commit "Initial module with unnamed initialization" in
      write "lib/initializer.ml"
        "let value = 1\n\
         let () = print_endline \"new\";;\n\
         print_endline \"new bare\";;\n\
         let exported = 1 and () = print_endline \"new mixed\"\n";
      let quiet = commit "Change non-exporting initialization" in
      p "non-exporting initializer and evaluation edits do not create API entries"
        (has (read ~until:quiet initialized) "0 declaration changes");
      write "lib/pattern.ml" "let [%publish earlier] = package\n";
      let pattern = commit "Introduce pattern PPX input" in
      write "lib/pattern.ml" "let [%publish later] = package\n";
      let changed_pattern = commit "Change pattern PPX input" in
      p "pattern PPX input changes remain visible"
        (has (read ~until:changed_pattern pattern) "publish later");
      write "arrayjit/lib/cap.mli" "open B\nopen A\nval x : t\n";
      let ordered = commit "Public declaration after opens" in
      write "arrayjit/lib/cap.mli" "open B\nval x : t\nopen A\n";
      let reordered = commit "Move public declaration across open" in
      p "declaration movement preserves ordering context"
        (has (read ~until:reordered ordered) "val x");
      write "arrayjit/lib/cap.mli" "val";
      ignore (commit "Invalid source must refuse" : string);
      p "invalid source refuses the real historical reader"
        (has (read ~success:false reordered) "api-drift:"))
    ~finally:(fun () ->
      Unix.chdir previous;
      Unix.unlink git_config;
      remove scratch)
