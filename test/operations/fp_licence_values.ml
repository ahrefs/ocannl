(* gh-ocannl-998: the floating-point licences, executed. Each licence changes the value of its own
   discriminating witnesses and of no other licence's; and because the licensed algebra runs once,
   on the raw lowering ([Ir.Fp_algebra]), a placement flip cannot change a result at uniform f32: an
   inlined producer is never contracted with its consumer.

   cc only, with the C compiler's own contraction pinned off ([cc_backend_fp_contract=off]): the
   claims are about OCANNL's algebra, and a compiler free to fuse [a * b + c] -- the platform
   default of clang, nvcc, hipcc and Metal -- would fuse the inlined read itself. Elsewhere every
   claim is skipped. Mixed-precision witnesses are left out: on cc a narrow product is computed in
   f32 ([narrow_compute_f32]), so where it rounds is a codegen question, not a licence's. *)
open Base
open Verdict.Claims
open Ll_test
module LL = Ir.Low_level
module Ops = Ir.Ops

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cc = List.mem [ "cc"; "multidev_cc" ] backend_name ~equal:String.equal
let () = Stdio.eprintf "fp-licence values backend: %s\n" backend_name
let () = Unix.putenv "OCANNL_CC_BACKEND_FP_CONTRACT" "off"
let claim label f = if on_cc then p label (f ()) else Verdict.skipped ~backend:backend_name label

let claim_all label xs ~f =
  if on_cc then p_all label xs ~f else Verdict.skipped ~backend:backend_name label

let bitwise a b =
  if Float.is_nan a || Float.is_nan b then Float.is_nan a && Float.is_nan b
  else Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let p2 n = Float.ldexp 1. n
let licence_name = [ "fp_contract"; "fp_reassociate"; "fp_finite_math" ]

let licence_of = function
  | "fp_contract" -> { LL.Fp_licence.exact with contract = true }
  | "fp_reassociate" -> { LL.Fp_licence.exact with reassociate = true }
  | _ -> { LL.Fp_licence.exact with finite_math = true }

(* The inputs, by slot. All exact in f32. *)
let near_one = 1. +. p2 (-12)

let inputs =
  [|
    near_one (* 0 *);
    near_one (* 1 *);
    -.(1. +. p2 (-11)) (* 2 *);
    1. (* 3 *);
    p2 24 (* 4 *);
    p2 24 (* 5 *);
    p2 100 (* 6 *);
    p2 (-100) (* 7 *);
    Float.infinity (* 8 *);
    -0. (* 9 *);
    Float.nan (* 10 *);
  |]

let () =
  let n_in = Array.length inputs in
  let mk = node_factory ~first_id:1099800 ~dims:[| n_in |] () in
  let input = mk "fpv_input" in
  materialize input;
  let x i = get input [| fixed i |] in
  let sub = binop Ops.Sub and div = binop Ops.Div in
  (* (licence that changes it, what it shows, expression) *)
  let witnesses =
    [
      ("fp_contract", "contraction: a * b + c rounds once", add (mul (x 0) (x 1)) (x 2));
      ( "fp_reassociate",
        "constant reassociation: (1 + 2^24) - 2^24",
        add (add (x 3) (c (p2 24))) (c (-.p2 24)) );
      ("fp_reassociate", "catastrophic cancellation: 1 + (2^24 - 2^24)", add (x 3) (sub (x 4) (x 4)));
      ("fp_reassociate", "asymmetric subtraction: 1 - (2^24 - 2^24)", sub (x 3) (sub (x 4) (x 5)));
      ("fp_reassociate", "overflow: 2^100 * (2^100 / 2^100)", mul (x 6) (div (x 6) (x 6)));
      ("fp_reassociate", "underflow: 2^-100 / (2^-100 / 2^-100)", div (x 7) (div (x 7) (x 7)));
      ("fp_finite_math", "infinity: 0 * inf", mul (c 0.) (x 8));
      ("fp_finite_math", "NaN: 0 * NaN", mul (c 0.) (x 10));
      ("fp_finite_math", "signed zero: -0 + 0", add (x 9) (c 0.));
    ]
  in
  let n_out = List.length witnesses in
  let run ~first_id name licence =
    let omk = node_factory ~first_id ~dims:[| n_out |] () in
    let output = omk ("fpv_out_" ^ name) in
    materialize output;
    let raw =
      List.foldi witnesses ~init:LL.Noop ~f:(fun k acc (_, _, rhs) ->
          seq acc (set output [| fixed k |] rhs))
    in
    let name = "fpv_" ^ name in
    let o = optimize ~materialized:[ input; output ] ~name (Ir.Fp_algebra.rewrite licence raw) in
    List.hd_exn (execute ~name o ~seed:[ (input, inputs); (output, blank n_out) ] ~read:[ output ])
  in
  let runs =
    lazy
      ( run ~first_id:1099810 "exact" LL.Fp_licence.exact,
        List.mapi licence_name ~f:(fun i key ->
            (key, run ~first_id:(1099820 + (10 * i)) key (licence_of key))) )
  in
  let ieee = [ 0.; 0.; 1.; 1.; p2 100; p2 (-100); Float.nan; Float.nan; 0. ] in
  claim_all "the exact run computes the IEEE values"
    (List.mapi ieee ~f:(fun k want -> (k, want)))
    ~f:(fun (k, want) ->
      let exact, _ = Lazy.force runs in
      bitwise exact.(k) want);
  let cells =
    List.concat_map licence_name ~f:(fun key ->
        List.mapi witnesses ~f:(fun k (owner, what, _) -> (key, k, owner, what)))
  in
  claim_all "each licence changes its own witnesses and leaves the others' values" cells
    ~f:(fun (key, k, owner, what) ->
      let exact, licensed = Lazy.force runs in
      let got = (List.Assoc.find_exn licensed key ~equal:String.equal).(k) in
      let changed = not (bitwise got exact.(k)) in
      let ok = Bool.equal changed (String.equal key owner) in
      if not ok then
        Stdio.eprintf "(not part of the golden) %s under %s: got %h, exact %h\n" what key got
          exact.(k);
      ok);
  (* Placement neutrality: a producer [y := a * b] read by [z := y + c], inlined or materialized. *)
  let contract = { LL.Fp_licence.exact with contract = true } in
  let leg name ~virtual_producer =
    let lmk =
      node_factory ~first_id:(if virtual_producer then 1099860 else 1099870) ~dims:[| 1 |] ()
    in
    let y = lmk ("fpv_y_" ^ name) and z = lmk ("fpv_z_" ^ name) in
    materialize z;
    if virtual_producer then virtualize y else materialize y;
    let raw =
      seq
        (set y [| fixed 0 |] (mul (x 0) (x 1)))
        (set z [| fixed 0 |] (add (get y [| fixed 0 |]) (x 2)))
    in
    let o =
      optimize
        ~materialized:([ input; z ] @ if virtual_producer then [] else [ y ])
        ~name:("fpv_" ^ name)
        (Ir.Fp_algebra.rewrite contract raw)
    in
    (o, y, z)
  in
  let read (o, _, z) ~name =
    (List.hd_exn (execute ~name o ~seed:[ (input, inputs); (z, blank 1) ] ~read:[ z ])).(0)
  in
  let legs =
    lazy
      (let inlined = leg "inlined" ~virtual_producer:true in
       let stored = leg "stored" ~virtual_producer:false in
       let fused_o, _, fused_z = inlined in
       (* The old pipeline's order: contract AFTER virtualization. *)
       let late =
         ({ fused_o with LL.llc = Ir.Fp_algebra.rewrite contract fused_o.LL.llc }, (), fused_z)
       in
       ( inlined,
         read inlined ~name:"fpv_inlined",
         read stored ~name:"fpv_stored",
         let o, _, z = late in
         (List.hd_exn
            (execute ~name:"fpv_late" o ~seed:[ (input, inputs); (z, blank 1) ] ~read:[ z ])).(0) ))
  in
  claim "the inlined leg's producer is virtual" (fun () ->
      let (o, y, _), _, _, _ = Lazy.force legs in
      known_virtual o y);
  claim "an inlined and a materialized producer agree bitwise under the contraction licence"
    (fun () ->
      let _, inlined, stored, _ = Lazy.force legs in
      bitwise inlined stored && bitwise stored 0.);
  claim "contracting after virtualization, the old order, would fuse the inlined read" (fun () ->
      let _, _, stored, late = Lazy.force legs in
      bitwise late (p2 (-24)) && not (bitwise late stored));
  claim "fusion is detectable: the single-assignment control rounds once" (fun () ->
      let exact, licensed = Lazy.force runs in
      bitwise (List.Assoc.find_exn licensed "fp_contract" ~equal:String.equal).(0) (p2 (-24))
      && bitwise exact.(0) 0.)
