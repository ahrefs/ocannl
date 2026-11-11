open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims

(* gh-ocannl-2032: the numerator is solved incrementally, while its total remains stored on the row.
   A Shape_row already present at stage 2 used to re-emit the stale total forever. The other case
   follows Shape.finish_inference's stage-4 insertion instead. *)
let test ~early =
  let label = if early then "early Shape_row" else "stage-4 Shape_row" in
  let axes, stayed_lazy =
    Bounded_run.run label (fun () ->
        let forced = ref false in
        let x = Row.get_var () in
        let r = Row.get_row_for_var Row.empty_provenance (Row.get_row_var ()) in
        let origin = [] in
        let shape = Row.Shape_row (r, origin) in
        let constr =
          Row.Total_elems
            {
              numerator =
                Strided_var
                  {
                    coeff =
                      Utils.safe_lazy "two elements per block" (fun () ->
                          forced := true;
                          2);
                    var = x;
                    denom = 1;
                    round_up = true;
                  };
              divided_by = [];
              keep_axis = false;
            }
        in
        let solve stage cs env = Row.solve_inequalities ~stage cs env in
        let pending, env =
          solve Stage1
            (Row.Rows_constr { r = [ r ]; constr; origin } :: (if early then [ shape ] else []))
            Row.empty_env
        in
        let pending, env =
          solve Stage1
            (Row.Dim_eq { d1 = Var x; d2 = Row.get_default_dim ~d:3 (); origin } :: pending)
            env
        in
        let stayed_lazy = not !forced in
        let _, env =
          List.fold [ Row.Stage2; Stage3; Stage4; Stage5; Stage6a; Stage6b; Stage7 ]
            ~init:(pending, env) ~f:(fun (cs, env) stage ->
              let cs = if (not early) && Row.equal_stage stage Stage4 then shape :: cs else cs in
              solve stage cs env)
        in
        (Row.subst_row env r, stayed_lazy))
  in
  pf "%s: stage 1 leaves the coefficient lazy" label stayed_lazy;
  pf "%s: the row uses all six elements" label
    (match axes with
    | { beg_dims = []; dims = [ Dim { d = 6; _ } ]; bcast = Broadcastable; _ } -> true
    | _ -> false)

let () =
  test ~early:false;
  test ~early:true
