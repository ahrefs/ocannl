(* gh-ocannl-2243: diagnostic attribution only; all three stores remain refused. *)
open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let dim d = Row.get_default_dim ~d ()
let size dim size = Row.Dim_size { dim; size; origin = [] }

let caught f =
  match f () with
  | _ -> None
  | exception Row.Shape_error (msg, trace) ->
      let bt = Stdlib.Printexc.get_raw_backtrace () in
      Some (msg, trace, bt)

let stage1 cs = Row.solve_inequalities ~stage:Stage1 cs Row.empty_env

let head_is c = function
  | Some (_, Row.Constraint_failed got :: _, _) -> Poly.equal c got
  | _ -> false

let () =
  Stdlib.Printexc.record_backtrace true;
  Verdict.case "held sizes name their own relation" (fun () ->
      let first = size (Row.Var (Row.get_var ())) 7 in
      let second = size (dim 3) 4 in
      let error = caught (fun () -> stage1 [ first; second ]) in
      p "the inconsistent second size is refused" (Option.is_some error);
      p "the trace starts with the second size" (head_is second error));
  Verdict.case "held inverse keeps the nonempty-output boundary" (fun () ->
      let output = Row.Var (Row.get_var ()) in
      let window =
        Row.Affine
          {
            stride = 2;
            over = output;
            conv = Some { dilation = 1; kernel = dim 5; use_padding = false };
            stride_offset = 0;
          }
      in
      let relation = size window 3 in
      let inverse = caught (fun () -> stage1 [ relation ]) in
      let unifier =
        caught (fun () -> stage1 [ Row.Dim_eq { d1 = window; d2 = dim 3; origin = [] } ])
      in
      p "the unifier refuses the too-short input" (Option.is_some unifier);
      p "the held inverse refuses at Stage1 with the unifier message"
        (match (inverse, unifier) with
        | Some (a, _, _), Some (b, _, _) -> String.equal a b
        | _ -> false);
      p "the inverse names the held window size" (head_is relation inverse));
  Verdict.case "closing replay keeps the first failure and its raise site" (fun () ->
      let fresh () =
        Row.get_row_for_var (Row.provenance ~sh_id:40 ~kind:`Output) (Row.get_row_var ())
      in
      let fixture () =
        let v = fresh () and v2 = fresh () in
        (match v2.bcast with Row_var r -> Row.add_resolve_at_use r | Broadcastable -> ());
        let lhs = { v with beg_dims = [ dim 3 ] }
        and rhs = { v2 with dims = [ Row.Var (Row.get_var ~name:"y" ()) ] } in
        let closed = { (fresh ()) with bcast = Broadcastable; dims = [ dim 5 ] } in
        let cs =
          [
            Row.Shape_row (v2, []);
            Row.Row_eq { r1 = lhs; r2 = rhs; origin = [] };
            Row.Row_ineq { res = closed; opnd = v2; origin = [] };
          ]
        in
        List.fold [ Row.Stage1; Stage2; Stage3; Stage4; Stage5 ] ~init:(cs, Row.empty_env)
          ~f:(fun (cs, env) stage -> Row.solve_inequalities ~stage cs env)
      in
      let stages = [ Row.Stage6a; Stage6b; Stage7 ] in
      let cs, env = fixture () in
      let first =
        caught (fun () ->
            List.fold stages ~init:(cs, env) ~f:(fun (cs, env) stage ->
                Row.solve_inequalities ~stage cs env))
      in
      let cs, env = fixture () in
      Row.reset_trial_replays ();
      let replay = caught (fun () -> Row.solve_closing_stages ~stages cs env) in
      p "both attempts refuse, with exactly one replay"
        (Option.is_some replay && Row.trial_replays () = 1);
      p "the first failure payload is unchanged"
        (match (first, replay) with
        | Some (a, at, _), Some (b, bt, _) ->
            String.equal (Shape.error_to_string_hum a at) (Shape.error_to_string_hum b bt)
        | _ -> false);
      let frame = function
        | None -> ""
        | Some (_, _, bt) ->
            Stdlib.Printexc.raw_backtrace_to_string bt
            |> String.split_lines |> List.hd_exn |> String.split ~on:'"' |> List.hd_exn
      in
      let original = frame first and replayed = frame replay in
      Stdio.eprintf "first: %s\nreplayed: %s (not part of the golden)\n" original replayed;
      p "the first frame names the original solver raise function"
        (String.is_substring original ~substring:"Row.solve_inequalities"
        && String.equal original replayed))
