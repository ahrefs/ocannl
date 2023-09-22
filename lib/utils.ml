open Base

module Set_O = struct
  let ( + ) = Set.union
  let ( - ) = Set.diff
  let ( & ) = Set.inter

  let ( -* ) s1 s2 =
    Set.of_sequence (Set.comparator_s s1) @@ Sequence.map ~f:Either.value @@ Set.symmetric_diff s1 s2
end

let no_ints = Set.empty (module Int)
let one_int = Set.singleton (module Int)

(** Retrieves [arg_name] argument from the command line or from an environment variable, returns
    [default] if none found. *)
let get_global_arg ~verbose ~default ~arg_name:n =
  if verbose then Stdio.printf "Retrieving commandline or environment variable <ocannl_%s>... %!" n;
  let variants = [ n; String.uppercase n ] in
  let env_variants =
    List.concat_map variants ~f:(fun n -> [ "ocannl_" ^ n; "OCANNL_" ^ n; "ocannl-" ^ n; "OCANNL-" ^ n ])
  in
  let cmd_variants = List.concat_map env_variants ~f:(fun n -> [ n; "-" ^ n; "--" ^ n ]) in
  let cmd_variants = List.concat_map cmd_variants ~f:(fun n -> [ n ^ "_"; n ^ "-"; n ^ "=" ]) in
  match
    Array.find_map (Core.Sys.get_argv ()) ~f:(fun arg ->
        List.find_map cmd_variants ~f:(fun prefix ->
            Option.some_if (String.is_prefix ~prefix arg) (prefix, arg)))
  with
  | Some (prefix, arg) ->
      let result = String.suffix arg (String.length prefix) in
      if verbose then Stdio.printf "found <%s> as commandline <%s>.\n%!" result arg;
      result
  | None -> (
      match
        List.find_map env_variants ~f:(fun env_n ->
            Option.map (Core.Sys.getenv env_n) ~f:(fun v -> (env_n, v)))
      with
      | Some (env_n, v) ->
          if verbose then Stdio.printf "found <%s> as environment variable <%s>.\n%!" v env_n;
          v
      | None ->
          if verbose then Stdio.printf "not found, using default <%s>.\n%!" default;
          default)

let rec union_find ~equal map ~key ~rank =
  match Map.find map key with
  | None -> (key, rank)
  | Some data -> if equal key data then (key, rank) else union_find ~equal map ~key:data ~rank:(rank + 1)

let union_add ~equal map k1 k2 =
  if equal k1 k2 then map
  else
    let root1, rank1 = union_find ~equal map ~key:k1 ~rank:0
    and root2, rank2 = union_find ~equal map ~key:k2 ~rank:0 in
    if rank1 < rank2 then Map.update map root1 ~f:(fun _ -> root2)
    else Map.update map root2 ~f:(fun _ -> root1)

(** Filters the list keeping the first occurrence of each element. *)
let unique_keep_first ~equal l =
  let rec loop acc = function
    | [] -> List.rev acc
    | hd :: tl -> if List.mem acc hd ~equal then loop acc tl else loop (hd :: acc) tl
  in
  loop [] l
