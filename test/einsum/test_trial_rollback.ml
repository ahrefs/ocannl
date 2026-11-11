(* gh-ocannl-2129: a refused trial solve leaves no trace. The row solver tries equations it may drop
   -- [solve_dim_ineq]'s equality merge of two bounds, an open window's candidate tags and the probe
   vetting them -- and a [Var = Var] merge shares a use-site mark ([resolve_at_use_dim]) before a
   later part of the same trial fails. The mark decides how the variable closes: marked, an
   unconstrained dim with a bound closes to it at stage 6; unmarked, it closes minimal at stage 7.
   So a leaked mark changes the shape silently. Each case below is run with a mark that only the
   refused trial could share, against the same system with nothing marked, and a control shows the
   closing does tell a marked variable apart.

   The window sites (a candidate's solve, the probe vetting it) are rolled back too, but no system
   here makes their leak visible: the marks a refused candidate shares come from equations the held
   system implies whatever the tag. Then the order claim, a pin of the issue's other premise that
   did not reproduce: windows, bounds, shared kernels and window pairs held over one variable settle
   alike in every order, with candidates vetted on their own consequences only. *)

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
let tagged d basis = Row.get_dim ~d ~basis ()
let eq d1 d2 : Row.constraint_ = Row.Dim_eq { d1; d2; origin }
let ineq res opnd : Row.constraint_ = Row.Dim_ineq { res; opnd; from_ = Sexp.Atom "test"; origin }
let closing = [ Row.Stage5; Stage6a; Stage6b; Stage7 ]

(* What closes [vs]: a shape row holding them, as a tensor's would. *)
let close vs : Row.constraint_ =
  Row.Shape_row
    ( {
        beg_dims = [];
        dims = List.map vs ~f:(fun v -> Row.Var v);
        bcast = Broadcastable;
        prov = Row.provenance ~sh_id:2129 ~kind:`Output;
      },
      origin )

let run batches =
  List.fold batches
    ~init:(Ok ([], Row.empty_env))
    ~f:(fun acc (stage, cs) ->
      Result.bind acc ~f:(fun (deferred, env) ->
          match Row.solve_inequalities ~stage (deferred @ cs) env with
          | r -> Ok r
          | exception Row.Shape_error (msg, _) -> Error msg))

let value env v =
  match Row.get_dim_from_env env v with
  | Some (Dim { d; basis; _ }) -> Printf.sprintf "%d of %s" d basis
  | Some _ -> "unsized"
  | None -> "open"

let show = function Ok (_, env) -> fun v -> value env v | Error msg -> fun _ -> "refused: " ^ msg

(* merge_by_equality, refused: [a; c; 5] against the bound [b; 3]. The pairing arm equates a = b,
   sharing a's mark, before the residual [c; 5] exceeds [3]. *)
let merge_refused ~mark =
  let v = Row.get_var () and b = Row.get_var () and a = Row.get_var () and c = Row.get_var () in
  if mark then Row.add_resolve_at_use_dim a;
  let r =
    run
      ([
         ( Row.Stage4,
           [
             ineq (Row.Concat [ Var b; dim 3 ]) (Var v);
             ineq (Row.Concat [ Var a; Var c; dim 5 ]) (Var v);
           ] );
         (Stage4, [ ineq (dim 7) (Var b); close [ b ] ]);
       ]
      @ List.map closing ~f:(fun s -> (s, [])))
  in
  (show r v, show r b, Row.is_resolve_at_use_dim b)

(* merge_by_equality, postponed below stage 4: [a; c] against [b; d] pairs a = b and c = d, then the
   merge waits; a is then solved, so the real merge at stage 4 never pairs a with b. *)
let merge_postponed ~mark =
  let v = Row.get_var () and b = Row.get_var () and d = Row.get_var () in
  let a = Row.get_var () and c = Row.get_var () in
  if mark then Row.add_resolve_at_use_dim a;
  let r =
    run
      ([
         ( Row.Stage1,
           [
             ineq (Row.Concat [ Var b; Var d ]) (Var v); ineq (Row.Concat [ Var a; Var c ]) (Var v);
           ] );
         (Stage1, [ eq (Var a) (dim 2) ]);
         (Stage2, []);
         (Stage3, []);
         (Stage4, [ ineq (dim 7) (Var b); close [ b ] ]);
       ]
      @ List.map closing ~f:(fun s -> (s, [])))
  in
  (show r v, show r b, Row.is_resolve_at_use_dim b)

(* The control: [b] marked itself closes to its bound, so the claims below can fail. *)
let marked_directly () =
  let b = Row.get_var () in
  Row.add_resolve_at_use_dim b;
  show
    (run
       ([ (Row.Stage4, [ ineq (dim 7) (Var b); close [ b ] ]) ]
       @ List.map closing ~f:(fun s -> (s, []))))
    b

let () =
  Verdict.case "a refused or postponed equality merge of two bounds leaves no mark" (fun () ->
      p "control: a variable marked itself closes to its bound of 7"
        (String.equal (marked_directly ()) "7 of default");
      let v, b, b_marked = merge_refused ~mark:true and v', b', _ = merge_refused ~mark:false in
      p "refused at stage 4: [a; c; 5] against [b; 3] demotes the operand" (String.equal v v');
      p "and b, merged with the marked a inside the refused attempt, stays unmarked" (not b_marked);
      p "so it closes as with nothing marked" (String.equal b b' && String.equal b "1 of bcast_if_1");
      let v, b, b_marked = merge_postponed ~mark:true and v', b', _ = merge_postponed ~mark:false in
      p "postponed below stage 4: [a; c] against [b; d], a solved before the merge applies"
        (String.equal v v');
      p "and b, merged with the marked a inside the postponed attempt, stays unmarked"
        (not b_marked);
      p "so it closes as with nothing marked" (String.equal b b' && String.equal b "1 of bcast_if_1"))

(* Search: held windows over one output (or a shared kernel), every order. *)
let window ?(stride = 1) ~kernel over =
  Row.Affine
    { stride; over; conv = Some { dilation = 1; kernel; use_padding = false }; stride_offset = 0 }

let tags = [ Row.default_basis; Row.bcast_if_1; "a"; "h" ]

let rec permutations = function
  | [] -> [ [] ]
  | xs ->
      List.concat_mapi xs ~f:(fun i x ->
          let rest = List.filteri xs ~f:(fun j _ -> j <> i) in
          List.map (permutations rest) ~f:(fun perm -> x :: perm))

type item = { name : string; mk : Row.dim_var -> Row.dim_var -> Row.dim_var -> Row.constraint_ }

let pool =
  let wins =
    List.concat_map tags ~f:(fun kt ->
        List.concat_map tags ~f:(fun it ->
            List.map
              [ (1, 3, 3); (1, 1, 1); (2, 3, 5) ]
              ~f:(fun (stride, ks, is) ->
                {
                  name = Printf.sprintf "w(s%d,k%d %s)(v)=%d %s" stride ks kt is it;
                  mk =
                    (fun v _ _ -> eq (window ~stride ~kernel:(tagged ks kt) (Var v)) (tagged is it));
                })))
  and bounds =
    List.concat_map tags ~f:(fun t ->
        List.map [ 1; 3 ] ~f:(fun d ->
            { name = Printf.sprintf "v>=%d %s" d t; mk = (fun v _ _ -> ineq (tagged d t) (Var v)) }))
  and kernels =
    List.concat_map tags ~f:(fun et ->
        List.map tags ~f:(fun it ->
            {
              name = Printf.sprintf "w(k)(1 %s)=3 %s" et it;
              mk = (fun _ _ k -> eq (window ~kernel:(Var k) (tagged 1 et)) (tagged 3 it));
            }))
  and pairs =
    List.concat_map tags ~f:(fun kt ->
        [
          {
            name = Printf.sprintf "w(k3 %s)(v)=w(k3 %s)(u)" kt kt;
            mk =
              (fun v u _ ->
                eq (window ~kernel:(tagged 3 kt) (Var v)) (window ~kernel:(tagged 3 kt) (Var u)));
          };
        ])
  in
  wins @ bounds @ kernels @ pairs

let outcome items order =
  let v = Row.get_var () and u = Row.get_var () and k = Row.get_var () in
  let cs = List.map order ~f:(fun i -> (List.nth_exn items i).mk v u k) in
  let r =
    run
      (List.map cs ~f:(fun c -> (Row.Stage1, [ c ]))
      @ List.map [ Row.Stage2; Stage3; Stage4 ] ~f:(fun s -> (s, []))
      @ List.map closing ~f:(fun s -> (s, [])))
  in
  match r with
  | Error _ -> "refused"
  | Ok (left, env) ->
      Printf.sprintf "v=%s u=%s k=%s%s" (value env v) (value env u) (value env k)
        (if List.is_empty left then "" else " +deferred")

let order_dependent items =
  let outs = List.map (permutations (List.init (List.length items) ~f:Fn.id)) ~f:(outcome items) in
  if List.for_all outs ~f:(String.equal (List.hd_exn outs)) then false
  else (
    Stdio.eprintf "%s => %s (not part of the golden)\n"
      (String.concat ~sep:" ; " (List.map items ~f:(fun i -> i.name)))
      (String.concat ~sep:" | " outs);
    true)

let rec choose k = function
  | _ when k = 0 -> [ [] ]
  | [] -> []
  | x :: xs -> List.map (choose (k - 1) xs) ~f:(fun c -> x :: c) @ choose k xs

let () =
  Verdict.case "held windows over one variable settle alike in every order" (fun () ->
      p_none ~min:2000 "two constraints: windows, bounds, a shared kernel, a window pair"
        (choose 2 pool) ~f:order_dependent;
      let held =
        List.filter pool ~f:(fun i ->
            not
              (String.is_prefix i.name ~prefix:"v>="
              || String.is_prefix i.name ~prefix:"w(s1,k1"
              || String.is_prefix i.name ~prefix:"w(s2"))
      in
      p_none ~min:5000 "three open windows, shared kernels or pairs" (choose 3 held)
        ~f:order_dependent)
