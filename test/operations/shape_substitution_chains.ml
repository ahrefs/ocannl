(** Long substitution chains through the shape solver (gh-ocannl-1904).

    A solution substitutes itself into the entries an index says it can change, rather than into the
    whole environment. Every scenario here runs under [Row.with_substitution_cross_check], which at
    each solution also runs the whole-environment substitution it replaced and fails unless the two
    leave the same environment and emit the same constraints, and again under
    [Row.with_whole_env_substitution], whose outcome must be the same. The scenarios aim at what the
    index must not miss: alias chains in both orders ending concrete and symbolic, aliases inside
    [Concat] and windows, a stored [Concat] only an unrelated solution normalizes, an occurs check
    reached through aliases, row chains on both flanks, bounds set before their endpoint resolves, a
    [Total_elems] row alive across a chain and a stage change, one no solution reaches, and two
    continuations of one saved environment. Drives [tensor/row.ml] directly, as
    [test_concat_dim_solver] does. *)

open! Base
open Ocannl
open Verdict.Claims

let origin : Row.constraint_origin list =
  [
    {
      lhs_name = "test";
      lhs_kind = `Output;
      rhs_name = "test";
      rhs_kind = `Output;
      operation = None;
    };
  ]

let dim d = Row.get_dim ~d ~basis:Row.default_basis ()
let eq d1 d2 : Row.constraint_ = Row.Dim_eq { d1; d2; origin }
let ineq res opnd : Row.constraint_ = Row.Dim_ineq { res; opnd; from_ = Sexp.List []; origin }
let row_var v = Row.get_row_for_var Row.empty_provenance v

let row ?(beg_dims = []) ?bcast dims : Row.t =
  {
    beg_dims;
    dims;
    bcast = Option.value_map bcast ~default:Row.Broadcastable ~f:(fun v -> Row.Row_var v);
    prov = Row.empty_provenance;
  }

let row_eq r1 r2 : Row.constraint_ = Row.Row_eq { r1; r2; origin }

(* One solve per constraint list, carrying the deferred constraints into the next: the shape of
   per-registration solving. *)
let solve_steps ?(stage = Row.Stage1) steps env =
  List.fold steps ~init:([], env) ~f:(fun (pending, env) cs ->
      Row.solve_inequalities ~stage (pending @ cs) env)

let vals env vars = List.map vars ~f:(Row.get_dim_val env)
let dims_of ds = List.map ds ~f:(fun d -> Option.try_with (fun () -> Row.dim_to_int_exn d))

(* A scenario's outcome, compared across the two substitutions: values, or the error's message. *)
type outcome = Values of int option list | Error of string

let equal_outcome (a : outcome) b = Poly.equal a b

let sexp_of_outcome = function
  | Values vs -> Sexp.List (Sexp.Atom "Values" :: List.map vs ~f:[%sexp_of: int option])
  | Error msg -> Sexp.List [ Sexp.Atom "Error"; Sexp.Atom msg ]

let outcome f = try Values (f ()) with Row.Shape_error (msg, _) -> Error msg

(* Runs [scenario] cross-checked and under whole-environment substitution, and claims the two agree;
   returns the cross-checked outcome and the indexed run's substitution visits. *)
let both name scenario =
  Row.reset_subst_visits ();
  let indexed = Row.with_substitution_cross_check (fun () -> outcome scenario) in
  let indexed_visits = Row.subst_visits () in
  Row.reset_subst_visits ();
  let whole = Row.with_whole_env_substitution (fun () -> outcome scenario) in
  let whole_visits = Row.subst_visits () in
  Stdio.eprintf "(not part of the golden) %s: %s; visits indexed %d, whole %d\n" name
    (Sexp.to_string_hum (sexp_of_outcome indexed))
    indexed_visits whole_visits;
  p
    (name ^ ": the indexed and the whole-environment substitution agree")
    (equal_outcome indexed whole);
  (indexed, indexed_visits, whole_visits)

let n = 200

(* [x_0 .. x_len] chained by equations, then the far end equated to [value]. Oldest-first solves
   [x_i := x_(i+1)], so each solution rewrites every earlier entry: the eager resolved environment's
   worst case. Newest-first solves [x_(i+1) := x_0], which nothing mentions yet. *)
let dim_chain ~newest_first ~len ~value () =
  let xs = List.init (len + 1) ~f:(fun _ -> Row.get_var ()) in
  let links =
    List.zip_exn (List.drop_last_exn xs) (List.tl_exn xs)
    |> List.map ~f:(fun (a, b) ->
        [ (if newest_first then eq (Var b) (Var a) else eq (Var a) (Var b)) ])
  in
  let last = if newest_first then List.hd_exn xs else List.last_exn xs in
  let _, env = solve_steps (links @ [ [ eq (Var last) value ] ]) Row.empty_env in
  (xs, env)

let () =
  Verdict.case "dimension alias chains" (fun () ->
      List.iter [ false; true ] ~f:(fun newest_first ->
          let order = if newest_first then "newest-first" else "oldest-first" in
          let a = if newest_first then "a " else "an " in
          let got, _, _ =
            both
              (a ^ order ^ " chain ending concrete")
              (fun () ->
                let xs, env = dim_chain ~newest_first ~len:n ~value:(dim 7) () in
                vals env xs)
          in
          p_all
            (a ^ order ^ " chain ending concrete solves every link")
            (match got with Values vs -> vs | Error _ -> [ None ])
            ~f:(Option.equal Int.equal (Some 7)));
      let sym, _ = Train.IDX.get_static_symbol ~static_range:8 Train.IDX.empty in
      let got, _, _ =
        both "a chain ending symbolic" (fun () ->
            let xs, env = dim_chain ~newest_first:false ~len:n ~value:(Row.get_sym_dim sym) () in
            List.map xs ~f:(fun x ->
                match Row.get_dim_from_env env x with Some (Sym _) -> Some 1 | _ -> None))
      in
      p_all "a chain ending symbolic solves every link to the symbol"
        (match got with Values vs -> vs | Error _ -> [ None ])
        ~f:(Option.equal Int.equal (Some 1)));
  Verdict.case "aliases inside concatenations and windows" (fun () ->
      let chain_to value =
        let xs = List.init 50 ~f:(fun _ -> Row.get_var ()) in
        let links =
          List.zip_exn (List.drop_last_exn xs) (List.tl_exn xs)
          |> List.map ~f:(fun (a, b) -> eq (Var a) (Var b))
        in
        (List.hd_exn xs, links @ [ eq (Var (List.last_exn xs)) value ])
      in
      let got, _, _ =
        both "a concatenation and a window over alias chains" (fun () ->
            let v = Row.get_var () and o = Row.get_var () and o2 = Row.get_var () in
            let a, a_links = chain_to (dim 2) and b, b_links = chain_to (dim 3) in
            let i, i_links = chain_to (dim 4) and i2, i2_links = chain_to (dim 5) in
            let k, k_links = chain_to (dim 3) in
            let window =
              Row.Affine
                {
                  stride = 1;
                  over = Var i2;
                  conv = Some { dilation = 1; kernel = Var k; use_padding = false };
                  stride_offset = 0;
                }
            in
            let _, env =
              solve_steps
                ([
                   [ eq (Var v) (Concat [ Var a; Var b ]) ];
                   [
                     eq (Var o)
                       (Affine { stride = 2; over = Var i; conv = None; stride_offset = 0 });
                   ];
                   [ eq (Var o2) window ];
                 ]
                @ List.map (a_links @ b_links @ i_links @ k_links @ i2_links) ~f:List.return)
                Row.empty_env
            in
            vals env [ v; o; o2 ])
      in
      p
        "the concatenation sums its resolved components, the strided window doubles its, and the \
         valid window adds its kernel's span"
        (equal_outcome got (Values [ Some 5; Some 8; Some 7 ]));
      let got, _, _ =
        both "a concatenation of solved parts" (fun () ->
            let w = Row.get_var () and u = Row.get_var () in
            let _, env = solve_steps [ [ eq (Var w) (Concat [ dim 2; dim 3 ]) ] ] Row.empty_env in
            let before = Row.get_dim_val env w in
            let _, env = solve_steps [ [ eq (Var u) (dim 1) ] ] env in
            [ before; Row.get_dim_val env w ])
      in
      p "a concatenation of solved parts is stored as its sum"
        (equal_outcome got (Values [ Some 5; Some 5 ])));
  Verdict.case "an occurs check reached through aliases" (fun () ->
      let got, _, _ =
        both "an indirect cycle" (fun () ->
            let x = Row.get_var () and y = Row.get_var () in
            let zs = List.init 50 ~f:(fun _ -> Row.get_var ()) in
            let links =
              List.zip_exn (y :: List.drop_last_exn zs) zs
              |> List.map ~f:(fun (a, b) -> [ eq (Var a) (Var b) ])
            in
            let _, env =
              solve_steps
                ([
                   [
                     eq (Var x)
                       (Affine { stride = 2; over = Var y; conv = None; stride_offset = 0 });
                   ];
                 ]
                @ links
                @ [ [ eq (Var (List.last_exn zs)) (Var x) ] ])
                Row.empty_env
            in
            vals env [ x ])
      in
      p "the cycle closed through the alias chain fails the occurs check"
        (match got with
        | Error msg -> String.is_substring msg ~substring:"occurs check"
        | _ -> false));
  Verdict.case "row chains on both flanks" (fun () ->
      let len = 60 in
      List.iter [ false; true ] ~f:(fun newest_first ->
          let order = if newest_first then "newest-first" else "oldest-first" in
          let a = if newest_first then "a " else "an " in
          let got, _, _ =
            both
              (a ^ order ^ " row chain")
              (fun () ->
                let rs = List.init (len + 1) ~f:(fun _ -> Row.get_row_var ()) in
                let links =
                  List.mapi
                    (List.zip_exn (List.drop_last_exn rs) (List.tl_exn rs))
                    ~f:(fun i (a, b) ->
                      let outer, inner = if newest_first then (b, a) else (a, b) in
                      [
                        row_eq (row_var outer)
                          (row ~beg_dims:[ dim (i + 1) ] ~bcast:inner [ dim (1000 + i) ]);
                      ])
                in
                let last = if newest_first then List.hd_exn rs else List.last_exn rs in
                let _, env =
                  solve_steps (links @ [ [ row_eq (row_var last) (row []) ] ]) Row.empty_env
                in
                let first = if newest_first then List.last_exn rs else List.hd_exn rs in
                let r = Row.subst_row env (row_var first) in
                dims_of r.beg_dims @ [ None ] @ dims_of r.dims)
          in
          let expected =
            let beg = List.init len ~f:(fun i -> Some (i + 1))
            and dims = List.init len ~f:(fun i -> Some (1000 + i)) in
            if newest_first then List.rev beg @ [ None ] @ dims else beg @ [ None ] @ List.rev dims
          in
          p
            (a ^ order ^ " row chain composes both flanks in order")
            (equal_outcome got (Values expected))));
  Verdict.case "bounds set before their endpoint resolves" (fun () ->
      let bounded ~a_value () =
        let a = Row.get_var () and b = Row.get_var () in
        let cs = List.init 50 ~f:(fun _ -> Row.get_var ()) in
        let links =
          List.zip_exn (b :: List.drop_last_exn cs) cs
          |> List.map ~f:(fun (x, y) -> [ eq (Var x) (Var y) ])
        in
        let _, env =
          solve_steps
            ([ [ ineq (Var a) (Var b) ]; [ eq (Var a) (dim a_value) ] ]
            @ links
            @ [ [ eq (Var (List.last_exn cs)) (dim 3) ] ])
            Row.empty_env
        in
        vals env [ a; b ]
      in
      let compatible, _, _ = both "a compatible bound" (bounded ~a_value:3) in
      p "a bound met by the endpoint's eventual value holds"
        (equal_outcome compatible (Values [ Some 3; Some 3 ]));
      let incompatible, _, _ = both "an incompatible bound" (bounded ~a_value:5) in
      p "a bound the endpoint's eventual value violates fails"
        (match incompatible with Error _ -> true | Values _ -> false));
  Verdict.case "a Total_elems row across a chain and a stage change" (fun () ->
      let got, _, _ =
        both "a Total_elems row" (fun () ->
            let r = Row.get_row_var () and x = Row.get_var () and y = Row.get_var () in
            let constr =
              Row.Total_elems { numerator = Num_elems 12; divided_by = [ x ]; keep_axis = false }
            in
            let _, env =
              solve_steps
                [ [ Row.Rows_constr { r = [ row_var r ]; constr; origin } ] ]
                Row.empty_env
            in
            let xs, env' = dim_chain ~newest_first:true ~len:50 ~value:(dim 2) () in
            ignore (xs : Row.dim_var list);
            ignore (env' : Row.environment);
            let unrelated =
              List.init 50 ~f:(fun i -> [ eq (Var (Row.get_var ())) (dim (i + 1)) ])
            in
            let pending, env = solve_steps (unrelated @ [ [ eq (Var x) (dim 3) ] ]) env in
            let _, env =
              Row.solve_inequalities ~stage:Stage2
                (pending @ [ row_eq (row_var r) (row [ Var y ]) ])
                env
            in
            vals env [ x; y ])
      in
      p
        "the constraint survives the unrelated solutions and solves the row once its divisor is \
         known"
        (equal_outcome got (Values [ Some 3; Some 4 ])));
  (* gh-ocannl-1977: a row outside a solution stays out of it: it used to re-emit at every dimension
     solution, reversing its divisors and staying pending for the next one. *)
  Verdict.case "a Total_elems row the solutions do not reach" (fun () ->
      let visits_with row =
        Row.reset_subst_visits ();
        let r = Row.get_row_var () and x = Row.get_var () and y = Row.get_var () in
        let constr =
          Row.Total_elems { numerator = Num_elems 12; divided_by = [ x; y ]; keep_axis = false }
        in
        let start =
          if row then [ [ Row.Rows_constr { r = [ row_var r ]; constr; origin } ] ] else []
        in
        let unrelated = List.init 51 ~f:(fun i -> [ eq (Var (Row.get_var ())) (dim (i + 1)) ]) in
        let _, env = solve_steps (start @ unrelated) Row.empty_env in
        (Row.subst_visits (), Row.find_row_entry env r, x, y)
      in
      let bare, _, _, _ = visits_with false in
      let visits, entry, x, y = visits_with true in
      p "fifty unrelated solutions visit it once" (visits - bare <= 2);
      p "and leave its divisors as they were"
        (match entry with
        | Some (Bounds_row { constr = Total_elems { divided_by = [ a; b ]; _ }; _ }) ->
            Row.equal_dim_var a x && Row.equal_dim_var b y
        | _ -> false));
  Verdict.case "a Total_elems numerator solved before the stage that reads it" (fun () ->
      let closes ~unrelated =
        let r = Row.get_row_var () and x = Row.get_var () in
        let constr =
          Row.Total_elems
            {
              numerator =
                Strided_var
                  { coeff = Utils.safe_value "gh-1977" 2; var = x; denom = 1; round_up = false };
              divided_by = [];
              keep_axis = false;
            }
        in
        let pending, env =
          solve_steps
            [
              [
                Row.Rows_constr { r = [ row_var r ]; constr; origin };
                Row.Shape_row (row_var r, origin);
              ];
              [ eq (Var x) (dim 3) ];
            ]
            Row.empty_env
        in
        let pending, env =
          Row.solve_inequalities ~stage:Stage2
            (pending @ if unrelated then [ eq (Var (Row.get_var ())) (dim 9) ] else [])
            env
        in
        let _, env =
          List.fold [ Row.Stage3; Stage4; Stage5; Stage6a; Stage6b; Stage7 ] ~init:(pending, env)
            ~f:(fun (pending, env) stage -> Row.solve_inequalities ~stage pending env)
        in
        List.map (Row.subst_row env (row_var r)).dims ~f:Row.dim_to_int_exn
      in
      p "it needs no unrelated solution to wake it"
        (List.equal Int.equal (closes ~unrelated:false) (closes ~unrelated:true)));
  Verdict.case "two continuations of one saved environment" (fun () ->
      let got, _, _ =
        both "two continuations" (fun () ->
            let xs = List.init 100 ~f:(fun _ -> Row.get_var ()) in
            let links =
              List.zip_exn (List.drop_last_exn xs) (List.tl_exn xs)
              |> List.map ~f:(fun (a, b) -> [ eq (Var b) (Var a) ])
            in
            let _, saved = solve_steps links Row.empty_env in
            let x0 = List.hd_exn xs in
            let _, env3 = solve_steps [ [ eq (Var x0) (dim 3) ] ] saved in
            let _, env4 = solve_steps [ [ eq (Var x0) (dim 4) ] ] saved in
            let _, again = solve_steps [ [ eq (Var x0) (dim 3) ] ] saved in
            let all v env =
              Some
                (if List.for_all (vals env xs) ~f:(Option.equal Int.equal (Some v)) then 1 else 0)
            in
            let unsolved = Some (if List.for_all (vals saved xs) ~f:Option.is_none then 1 else 0) in
            [ all 3 env3; all 4 env4; all 3 again; unsolved ])
      in
      p "each continuation solves the chain its own way, and the saved environment stays unsolved"
        (equal_outcome got (Values [ Some 1; Some 1; Some 1; Some 1 ])));
  Verdict.case "substitution work along a chain" (fun () ->
      let visits ~newest_first len =
        Row.reset_subst_visits ();
        ignore (dim_chain ~newest_first ~len ~value:(dim 7) ());
        let indexed = Row.subst_visits () in
        Row.reset_subst_visits ();
        ignore (Row.with_whole_env_substitution (dim_chain ~newest_first ~len ~value:(dim 7)));
        let whole = Row.subst_visits () in
        Stdio.eprintf "(not part of the golden) %s chain of %d: visits indexed %d, whole %d\n"
          (if newest_first then "newest-first" else "oldest-first")
          len indexed whole;
        (Float.of_int indexed, Float.of_int whole)
      in
      let leg =
        Verdict.leg ~aggregation:`Environment
          ~when_:(not (Row.substitution_cross_checked ()))
          ~on:"shape_substitution_cross_check=true" ()
      in
      let i1, w1 = visits ~newest_first:true n and i2, w2 = visits ~newest_first:true (2 * n) in
      let o1, ow1 = visits ~newest_first:false n and o2, ow2 = visits ~newest_first:false (2 * n) in
      in_leg ~leg "a newest-first chain twice as long takes 1.8x-2.2x the indexed visits"
        (fun label -> p label (Float.between (i2 /. i1) ~low:1.8 ~high:2.2));
      in_leg ~leg "and at least 3x the whole-environment visits" (fun label ->
          p label Float.(w2 /. w1 >= 3.));
      (* Oldest-first, every entry of this environment mentions each new solution, so both
         substitutions rewrite them all: quadratic in the chain's length, the eagerly resolved
         environment's worst case (gh-ocannl-1904 measured no such chain in the model workloads). *)
      in_leg ~leg
        "an oldest-first chain, the eager environment's worst case, takes no more indexed visits \
         than whole-environment ones" (fun label -> p label Float.(o1 <= ow1 && o2 <= ow2)))
