open Base

let parse ?documentation token : Utils.config_token_parse =
  Utils.parse_config_token ?documentation token

let cli suffix = "--ocannl_" ^ suffix

let () =
  Verdict.p_all ~min:4 "non-OCANNL documentation assignments have no accepted reading"
    [ "fastMathEnabled=false"; "mathMode=Safe"; "d=1"; "PARALLEL=0" ] ~f:(fun token ->
      match parse ~documentation:true token with
      | Utils.Rejected _ | Utils.Ambiguous { explicit = None; _ } -> true
      | _ -> false);
  Verdict.p_all ~min:2 "qualified removed keys retain their explicit normalized name"
    [ (cli "removed_key=1", "removed_key"); ("OCANNL_" ^ "REMOVED_KEY=1", "removed_key") ]
    ~f:(fun (token, key) ->
      match parse token with
      | Utils.Accepted parsed | Utils.Ambiguous { explicit = Some parsed; _ } ->
          String.equal parsed.token_key key
      | _ -> false);
  Verdict.p_all ~min:3 "alternate and concatenated values expose runtime candidates"
    [
      (cli "print_decimals_precision-7", "print_decimals_precision");
      (cli "backend_cuda=true", "backend");
      (cli "backendmetal", "backend");
    ]
    ~f:(fun (token, key) ->
      match parse token with
      | Utils.Ambiguous { candidates; _ } ->
          List.exists candidates ~f:(fun candidate -> String.equal candidate.Utils.token_key key)
      | _ -> false);
  let candidate_pairs =
    List.concat_map
      [ cli "print_decimals_precision-7"; cli "removed_key-7" ]
      ~f:(fun token ->
        match parse token with
        | Utils.Ambiguous { candidates; _ } ->
            List.map candidates ~f:(fun parsed -> (token, parsed))
        | _ -> [])
  in
  Verdict.p_all ~min:2 "ambiguous CLI candidates come only from runtime spelling prefixes"
    candidate_pairs ~f:(fun (token, parsed) ->
      List.exists (Utils.cmdline_var_prefixes ~qualified_only:true parsed.Utils.token_key)
        ~f:(fun prefix ->
          String.is_prefix token ~prefix && String.length token > String.length prefix));
  Verdict.p_all ~min:3 "mixed CLI names never acquire an explicit accepted reading"
    [
      cli "Print_Decimals_Precision=7";
      cli "print_decimals-precision=7";
      "--OCANNL_" ^ "print_decimals_precision=7";
    ]
    ~f:(fun token ->
      match parse token with
      | Utils.Ambiguous { explicit = None; _ } | Utils.Rejected _ -> true
      | _ -> false);
  Verdict.p_all ~min:3 "syntax rejection reports its reason and diagnostic key"
    [
      (cli "", Utils.Empty_qualified_key, None);
      (cli "?=1", Utils.Invalid_command_line_key, Some "?");
      ("OCANNL_backend=cc", Utils.Invalid_environment_key, None);
    ]
    ~f:(fun (token, reason, key) ->
      match parse token with
      | Utils.Rejected rejected ->
          Poly.equal (rejected.reason : Utils.config_token_rejection) reason
          && Option.equal String.equal key
               (Option.map rejected.qualified ~f:(fun parsed -> parsed.Utils.token_key))
      | _ -> false);
  Verdict.p "one-word documentation assignments require consumer judgment"
    (match parse ~documentation:true "backend=cc" with
    | Utils.Ambiguous { explicit = None; candidates = [ parsed ]; _ } ->
        Poly.equal parsed.token_shape Utils.Documentation_assignment_token
        && String.equal parsed.token_key "backend"
    | _ -> false);
  Verdict.p "bare documentation grammar is opt-in"
    (match parse "debug_log_from_routines=true" with
    | Utils.Rejected { reason = Utils.Not_configuration_syntax; _ } -> true
    | _ -> false)
