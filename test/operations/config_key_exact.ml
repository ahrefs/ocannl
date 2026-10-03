open Base
open Verdict.Claims

let matches key expected =
  Option.equal String.equal (Option.map (Utils.read_cmdline_var key) ~f:fst) (Some expected)

let check_grammar () =
  (* Derive every overlapping pair from the registry: newly added keys join this test without a
     disjointness ban or a second list of names. *)
  let keys = Set.to_list Utils.known_config_keys in
  let pairs =
    List.concat_map keys ~f:(fun shorter ->
        List.filter_map keys ~f:(fun longer ->
            Option.some_if
              (String.length longer > String.length shorter
              && String.is_prefix longer ~prefix:shorter)
              (shorter, longer)))
  in
  p "the registry exercises overlapping key names" (not (List.is_empty pairs));
  p_all "longer registered keys own every spelling and value separator" pairs
    ~f:(fun (shorter, longer) ->
      List.for_all (Utils.cmdline_var_names longer) ~f:(fun name ->
          List.for_all [ "="; "_"; "-"; "" ] ~f:(fun separator ->
              let arg = name ^ separator ^ "true" in
              Option.is_none (Utils.cmdline_arg_value shorter arg)
              && Option.equal String.equal (Utils.cmdline_arg_value longer arg) (Some "true"))));
  p_all "shorter keys retain all separators for unregistered values"
    (Utils.cmdline_var_names "online_softmax") ~f:(fun name ->
      List.for_all [ "="; "_"; "-"; "" ] ~f:(fun separator ->
          Option.equal String.equal
            (Utils.cmdline_arg_value "online_softmax" (name ^ separator ^ "true"))
            (Some "true")));
  p "an equals sign ends the key even when its value starts with a registered suffix"
    (Option.equal String.equal
       (Utils.cmdline_arg_value "online_softmax" "--ocannl_online_softmax=backward=true")
       (Some "backward=true"));
  p "an explicitly queried unregistered key retains its value grammar"
    (Option.equal String.equal
       (Utils.cmdline_arg_value "demo_key"
          (List.hd_exn (Utils.cmdline_var_names "demo_key") ^ "=custom"))
       (Some "custom"));
  p "the caller can override qualified-only lookup without changing classifier policy"
    (Option.is_none (Utils.cmdline_arg_value "profile" "--profile=custom")
    && Option.equal String.equal
         (Utils.cmdline_arg_value ~qualified_only:false "profile" "--profile=custom")
         (Some "custom")
    && not (Utils.cmdline_arg_is_config_key "--profile=custom"))

let () =
  match Stdlib.Sys.argv.(1) with
  | "acceptance" ->
      p "a backward-only command line leaves online_softmax unset"
        (Option.is_none (Utils.read_cmdline_var "online_softmax"));
      p "the backward key reads its own command-line value"
        (matches "online_softmax_backward" "true");
      check_grammar ()
  | "forward-first" ->
      p "the forward key before the block key reads independently"
        (matches "online_softmax" "true" && matches "online_softmax_block" "16")
  | "block-first" ->
      p "the block key before the forward key reads independently"
        (matches "online_softmax" "true" && matches "online_softmax_block" "16")
  | "first-wins" ->
      p "the first argument for the same key still wins" (matches "online_softmax" "false")
  | _ -> failwith "unknown config_key_exact case"
