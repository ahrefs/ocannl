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
      let read ?(args = []) ?(until = "HEAD") ?(success = true) since =
        let status, output = command reader (args @ [ since; until ]) in
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
      write "lib/attributes.ml" "type t = A [@@deriving sexp]\n";
      let attributed = commit "First deriving input" in
      write "lib/attributes.ml" "type t = A [@@deriving compare]\n";
      let changed_attributes = commit "Change deriving input" in
      p "non-documentation attribute payload changes are visible"
        (has (read ~until:changed_attributes attributed) "deriving compare");
      write "lib/dune"
        "(library (name first) (public_name pkg.first) (modules attributes)) (library (name \
         second) (public_name pkg.second) (modules pattern))\n";
      let owned = commit "Public library owners" in
      write "lib/dune"
        "(library (name first) (public_name pkg.first) (modules pattern)) (library (name second) \
         (public_name pkg.second) (modules attributes))\n";
      let moved = commit "Move unchanged source between public libraries" in
      p "Dune-only public module moves produce publication-input review entries"
        (has (read ~until:moved owned) "public library first"
        && has (read ~until:moved owned) "public library second");
      write "lib/dune"
        "(ocamllex lexer public_lexer) (library (name lib) (public_name pkg.lib) (modules lexer \
         public_lexer) (private_modules lexer))\n";
      write "lib/lexer.mll" "{let value = 1}\nrule token = parse | eof { () }\n";
      write "lib/public_lexer.mll" "{let value = 1}\nrule token = parse | eof { () }\n";
      let generators = commit "Private and public generated modules" in
      write "lib/lexer.mll" "{let value = true}\nrule token = parse | eof { () }\n";
      let private_edit = commit "Change private generator input" in
      p "private generator edits of a public library remain evidence"
        (has (read ~until:private_edit generators) "lib/lexer.mll");
      write "lib/public_lexer.mll" "{let value = true}\nrule token = parse | eof { () }\n";
      let public_edit = commit "Change public generator input" in
      p "public generator peers still produce entries"
        (has (read ~until:public_edit private_edit) "lib/public_lexer.mll");
      write "lib/dune"
        "(ocamllex lexer public_lexer) (library (name lib) (public_name pkg.lib) (modules lexer \
         public_lexer) (private_modules lexer) (empty_module_interface_if_absent))\n";
      let empty_interface = commit "Give public generated module an empty interface" in
      p "Dune-only empty-interface policy changes stay visible"
        (has (read ~until:empty_interface public_edit) "empty_module_interface_if_absent");
      write "lib/dune"
        "(library (name first) (public_name pkg.first) (modules attributes)) (library (name \
         second) (public_name pkg.second) (modules pattern))\n";
      let independent = commit "Independent public owners" in
      write "lib/dune"
        "(library (name second) (public_name pkg.second) (modules pattern)) (library (name first) \
         (public_name pkg.first) (modules attributes))\n";
      let reordered_dune = commit "Reorder independent Dune stanzas" in
      p "Dune stanza reordering stays quiet"
        (has (read ~until:reordered_dune independent) "0 declaration changes");
      write "tensor/dune"
        "(menhir (modules parser) (flags --table)) (library (name parserlib) (public_name \
         pkg.parserlib) (modules parser))\n";
      let parser_config = commit "Generator configuration" in
      write "tensor/dune"
        "(menhir (modules parser) (flags --code)) (library (name parserlib) (public_name \
         pkg.parserlib) (modules parser))\n";
      let new_config = commit "Change generator configuration without changing inputs" in
      p "Dune-only generator configuration edits remain visible"
        (has (read ~until:new_config parser_config) "--code");
      write "tensor/dune"
        "(menhir (flags --code) (modules parser)) (library (name parserlib) (public_name \
         pkg.parserlib) (modules parser))\n";
      let generator_order = commit "Reorder generator fields without changing configuration" in
      p "generator configuration field ordering stays quiet"
        (has (read ~until:generator_order new_config) "0 declaration changes");
      let select_config condition =
        "(library (name backend) (public_name pkg.backend) (modules impl) (libraries (select \
         impl.ml from (" ^ condition ^ " -> impl.cuda.ml) (-> impl.missing.ml))))\n"
      in
      write "arrayjit/lib/dune" (select_config "cuda_alternative");
      let selected_config = commit "Select configuration" in
      write "arrayjit/lib/dune" (select_config "hip");
      let changed_select = commit "Change select condition with the same arm paths" in
      p "Dune-only selected-module configuration edits remain visible"
        (has (read ~until:changed_select selected_config) "hip -> impl.cuda.ml");
      let library_config fields =
        "(library (name backend) (public_name pkg.backend) (modules impl) " ^ fields ^ ")\n"
      in
      write "arrayjit/lib/dune" (library_config "(preprocess (pps ppx_sexp_conv))");
      let required_library = commit "Required public library preprocessing" in
      write "arrayjit/lib/dune" (library_config "(optional) (preprocess (pps ppx_sexp_conv))");
      let optional_library = commit "Make public library optional without changing source" in
      p "Dune-only optional library changes remain literal review evidence"
        (has (read ~until:optional_library required_library) "(optional)");
      write "arrayjit/lib/dune" (library_config "(optional) (preprocess (pps ppx_compare))");
      let preprocessing = commit "Change public library preprocessing without changing source" in
      p "Dune-only preprocessing changes remain literal review evidence"
        (has (read ~until:preprocessing optional_library) "ppx_compare");
      write "arrayjit/lib/dune"
        "(library (preprocess (pps ppx_compare)) (modules impl) (optional) (public_name \
         pkg.backend) (name backend) (libraries dependency) (synopsis \"new prose\"))\n";
      let config_order = commit "Reorder fields and edit dependencies and prose" in
      p "public configuration field ordering dependencies and prose stay quiet"
        (has (read ~until:config_order preprocessing) "0 declaration changes");
      write "lib/dune"
        "(library (name first) (public_name pkg.first) (modules attributes secret api) \
         (private_modules secret)) (library (name second) (public_name pkg.second) (modules \
         pattern))\n";
      write "lib/secret.ml" "let exported = 1\n";
      write "lib/api.ml" "include Secret\n";
      let privatized = commit "Private ordinary module included by a public one" in
      write "lib/secret.ml" "let exported = \"changed public type\"\n";
      let secret_edit = commit "Change a private ordinary module" in
      p "a private ordinary module a public one includes remains evidence"
        (has (read ~until:privatized config_order) "lib/secret.ml"
        && has (read ~until:secret_edit privatized) "changed public type");
      let long_body edited =
        "module Body = struct\n"
        ^ String.concat
            (List.init 40 ~f:(fun i ->
                 Printf.sprintf "  let step%d = %d\n" i (if edited && i = 20 then 2000 else i)))
        ^ "end\n"
      in
      write "lib/long.ml" (long_body false);
      let long = commit "Long implementation body" in
      write "lib/long.ml" (long_body true);
      let edited = commit "Edit one line of a long body (#629)" in
      let compact = read ~args:[ "--context"; "1" ] ~until:edited long in
      let full = read ~until:edited long in
      p "compact rendering keeps attribution and the changed fragment, and counts the rest"
        (has compact ("commit " ^ edited ^ " Edit one line of a long body (#629)")
        && has compact "lib/long.ml"
        && has compact "- module Body[0] (line 1)"
        && has compact "+ module Body[0] (line 1)"
        && has compact "step20 = 20\n" && has compact "step20 = 2000" && has compact "step19 = 19"
        && has compact "step21 = 21" && has compact "\n~ "
        && (not (has compact "step5 = 5"))
        && has compact ("Full declaration text: tools/api-drift.sh " ^ long ^ " " ^ edited));
      p "full rendering stays the default evidence"
        (has full "step5 = 5" && has full "step20 = 2000"
        && (not (has full "\n~ "))
        && not (has full "Compact rendering"));
      p "an invalid context count refuses with usage"
        (has (read ~args:[ "--context"; "-1" ] ~success:false long) "non-negative"
        && has (read ~args:[ "--context" ] ~success:false long) "Usage");
      write "arrayjit/lib/cap.mli" "val";
      ignore (commit "Invalid source must refuse" : string);
      p "invalid source refuses the real historical reader"
        (has (read ~success:false moved) "api-drift:"))
    ~finally:(fun () ->
      Unix.chdir previous;
      Unix.unlink git_config;
      remove scratch)
