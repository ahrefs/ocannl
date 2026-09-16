(* The algebraic-rewrite tier over raw lowered code; the contract is in the interface. *)

open Base

type rewrite = { name : string; enabled : unit -> bool; apply : Low_level.t -> Low_level.t }

let tier : rewrite list =
  [ { name = "online_softmax"; enabled = Online_softmax.enabled; apply = Online_softmax.rewrite } ]

let max_rounds = 8

let apply (llc : Low_level.t) : Low_level.t =
  match List.filter tier ~f:(fun r -> r.enabled ()) with
  | [] -> llc
  | enabled ->
      let rec fixpoint round llc =
        let llc' = List.fold enabled ~init:llc ~f:(fun llc r -> r.apply llc) in
        if Low_level.equal llc' llc then llc'
        else if round >= max_rounds then
          invalid_arg
            (Printf.sprintf
               "Rewrites.apply: no fixpoint after %d rounds of [%s] -- a rewrite keeps changing \
                its own output, so it is not idempotent or does not consume its pattern"
               max_rounds
               (String.concat ~sep:"; " (List.map enabled ~f:(fun r -> r.name))))
        else fixpoint (round + 1) llc'
      in
      fixpoint 1 llc
