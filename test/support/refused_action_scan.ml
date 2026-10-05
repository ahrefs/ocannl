(* Bounded Dune refusal-stream contract (gh-ocannl-1005).

   Membership is direct [(libraries arrayjit.verdict)] on a program declaration; runner identity
   comes from {!Dune_stanza_scan.program_runners}, including dependency origin under [chdir]. A run
   accepting a nonzero status must drop or capture both streams on THAT action branch. This is not a
   Dune evaluator: literal exit predicates (:standard, integers, or/and/not) are supported;
   includes, directory-spanning declarations, computed library membership, opaque launchers,
   pipelines and unknown actions in refusal branches fail explicitly. No shell parsing or transitive
   library closure is attempted. *)

open Base
module Dune = Dune_stanza_scan

type result = { refusals : string list; problems : string list }

let rec accepts predicate code =
  match predicate with
  | Sexp.Atom ":standard" -> code = 0
  | Sexp.Atom atom -> (
      match Int.of_string_opt atom with
      | Some n when n >= 0 && n <= 255 -> code = n
      | _ -> invalid_arg ("unsupported accepted-exit predicate " ^ Sexp.to_string predicate))
  | Sexp.List (Sexp.Atom "or" :: children) ->
      List.map children ~f:(fun p -> accepts p code) |> List.exists ~f:Fn.id
  | Sexp.List (Sexp.Atom "and" :: children) ->
      List.map children ~f:(fun p -> accepts p code) |> List.for_all ~f:Fn.id
  | Sexp.List [ Sexp.Atom "not"; child ] -> not (accepts child code)
  | _ -> invalid_arg ("unsupported accepted-exit predicate " ^ Sexp.to_string predicate)

let accepts_failure predicate =
  List.init 256 ~f:(fun code -> accepts predicate code)
  |> List.existsi ~f:(fun code accepted -> code <> 0 && accepted)

let rec contains_acceptance = function
  | Sexp.List (Sexp.Atom "with-accepted-exit-codes" :: _) -> true
  | Sexp.List children -> List.exists children ~f:contains_acceptance
  | Sexp.Atom _ -> false

let scan dune_files =
  let problems = ref [] and refusals = ref [] in
  let problem where text = problems := (where ^ ": " ^ text) :: !problems in
  let located =
    List.concat_map dune_files ~f:(fun (path, content) ->
        try
          let directory = match Stdlib.Filename.dirname path with "." -> "" | dir -> dir in
          List.iter (Dune.path_rewriting_stanzas content) ~f:(fun _ ->
              problem path "unsupported environment command-resolution override");
          Dune.walk directory (Dune.stanzas content) ~f:(fun dir stanza -> [ (path, dir, stanza) ])
        with exn ->
          problem path ("cannot read Dune input: " ^ Exn.to_string exn);
          [])
  in
  List.iter located ~f:(fun (path, _, stanza) ->
      match Dune.head stanza with
      | Some "include" -> problem path "unsupported include directive; declarations may be hidden"
      | Some "include_subdirs"
        when not (Sexp.equal stanza (Sexp.List [ Sexp.Atom "include_subdirs"; Sexp.Atom "no" ])) ->
          problem path "unsupported include_subdirs mode; executable ownership spans directories"
      | _ -> ());
  let declarations =
    List.filter located ~f:(fun (_, _, stanza) ->
        match Dune.head stanza with
        | Some ("executable" | "executables" | "test" | "tests") -> true
        | _ -> false)
  in
  let membership stanza =
    match Dune.field stanza "libraries" with
    | None -> false
    | Some libraries ->
        if
          List.exists libraries ~f:(function Sexp.List _ -> true | Sexp.Atom _ -> false)
          || List.exists libraries ~f:(function
            | Sexp.Atom a -> String.is_substring a ~substring:"%{" || String.is_prefix a ~prefix:":"
            | _ -> false)
        then invalid_arg "unsupported computed libraries membership";
        List.mem libraries (Sexp.Atom "arrayjit.verdict") ~equal:Sexp.equal
  in
  let owners ~dir runner =
    List.concat_map declarations ~f:(fun (_path, owner_dir, declaration) ->
        Dune.program_runners ~subdir:owner_dir ~runner_stanzas:[ (dir, runner) ] [] declaration
        |> List.filter_map ~f:(fun (name, runners) ->
            if List.is_empty runners then None
            else
              Some
                ( Dune.normalize_path (Dune.in_subdir owner_dir (name ^ ".exe")),
                  membership declaration )))
  in
  List.iter located ~f:(fun (path, dir, stanza) ->
      let outside_action =
        match stanza with
        | Sexp.List (head :: fields) ->
            Sexp.List
              (head
              :: List.filter fields ~f:(fun f ->
                  not (Option.value_map (Dune.head f) ~default:false ~f:(String.equal "action"))))
        | other -> other
      in
      if contains_acceptance outside_action then
        problem path "unsupported accepted-exit action outside the direct action field";
      let deps = Option.value (Dune.field stanza "deps") ~default:[] in
      let test_names =
        match Dune.head stanza with Some ("test" | "tests") -> Dune.names_of stanza | _ -> []
      in
      let runner context action =
        Sexp.List
          [
            Sexp.Atom "rule";
            Sexp.List (Sexp.Atom "deps" :: deps);
            Sexp.List [ Sexp.Atom "action"; context action ];
          ]
      in
      let rec walk ~refused ~stdout ~stderr ~context action =
        let descend ?(refused = refused) ?(stdout = stdout) ?(stderr = stderr) ?(context = context)
            children =
          List.iter children ~f:(walk ~refused ~stdout ~stderr ~context)
        in
        let redirect stream destination children =
          if Dune.is_absolute destination then
            if refused || List.exists children ~f:contains_acceptance then
              problem path
                "unsupported absolute capture destination (may alias an inherited stream)"
            else descend children
          else
            match stream with
            | `Stdout -> descend ~stdout:true children
            | `Stderr -> descend ~stderr:true children
            | `Both -> descend ~stdout:true ~stderr:true children
        in
        match action with
        | Sexp.List [ Sexp.Atom "with-accepted-exit-codes"; predicate; child ] -> (
            try descend ~refused:(accepts_failure predicate) [ child ]
            with exn -> problem path (Exn.to_string exn))
        | Sexp.List (Sexp.Atom "ignore-stdout" :: children) -> descend ~stdout:true children
        | Sexp.List (Sexp.Atom "ignore-stderr" :: children) -> descend ~stderr:true children
        | Sexp.List (Sexp.Atom "ignore-outputs" :: children) ->
            descend ~stdout:true ~stderr:true children
        | Sexp.List (Sexp.Atom "with-stdout-to" :: Sexp.Atom dest :: children) ->
            redirect `Stdout dest children
        | Sexp.List (Sexp.Atom "with-stderr-to" :: Sexp.Atom dest :: children) ->
            redirect `Stderr dest children
        | Sexp.List (Sexp.Atom "with-outputs-to" :: Sexp.Atom dest :: children) ->
            redirect `Both dest children
        | Sexp.List (Sexp.Atom (("chdir" | "setenv") as head) :: args) ->
            let n = if String.equal head "chdir" then 1 else 2 in
            let prefix, children = List.split_n args n in
            descend
              ~context:(fun leaf -> context (Sexp.List ((Sexp.Atom head :: prefix) @ [ leaf ])))
              children
        | Sexp.List (Sexp.Atom (("run" | "dynamic-run") as head) :: Sexp.Atom cmd :: args)
          when refused ->
            let commands =
              if String.equal cmd Dune.test_pform then
                List.map test_names ~f:(fun name -> "%{dep:" ^ name ^ ".exe}")
              else [ cmd ]
            in
            if List.is_empty commands then problem path "unresolved %{test} in refused action";
            List.iter commands ~f:(fun cmd ->
                let leaf = Sexp.List (Sexp.Atom head :: Sexp.Atom cmd :: args) in
                let runner = runner context leaf in
                try
                  let classified = Dune.executables_run runner in
                  let owned = owners ~dir runner in
                  if
                    List.exists classified ~f:(fun (_, command) ->
                        match command with
                        | Dune.Unrecognized _ | Unknown_directory _ | Path_rewritten _ -> true
                        | _ -> false)
                  then problem path ("unsupported refused command " ^ cmd)
                  else if List.is_empty owned && not (List.is_empty classified) then
                    problem path ("no executable declaration for refused command " ^ cmd)
                  else
                    List.iter owned ~f:(fun (program, verdict) ->
                        if verdict then (
                          let site = path ^ ": " ^ program in
                          refusals := site :: !refusals;
                          if not (stdout && stderr) then
                            problem site
                              ("accepted failure inherits "
                              ^
                              match (stdout, stderr) with
                              | false, false -> "stdout and stderr"
                              | false, true -> "stdout"
                              | true, false -> "stderr"
                              | true, true -> assert false)))
                with exn -> problem path (Exn.to_string exn))
        | Sexp.List
            (Sexp.Atom (("bash" | "system" | "pipe-stdout" | "pipe-stderr" | "pipe-outputs") as h)
            :: _)
          when refused || contains_acceptance action ->
            problem path ("unsupported refused action " ^ h)
        | Sexp.List (Sexp.Atom head :: children) ->
            if
              (refused || contains_acceptance action)
              && not (List.mem (Dune.inert_actions @ Dune.program_actions) head ~equal:String.equal)
            then problem path ("unsupported refused action " ^ head)
            else descend children
        | Sexp.Atom _ | Sexp.List [] -> ()
        | Sexp.List _ -> problem path "unsupported non-atomic action head"
      in
      Option.iter (Dune.field stanza "action") ~f:(fun actions ->
          List.iter actions ~f:(walk ~refused:false ~stdout:false ~stderr:false ~context:Fn.id)));
  { refusals = List.rev !refusals; problems = List.rev !problems }
