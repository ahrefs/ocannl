(* gh-ocannl-998: each floating-point licence controls only its own rewrites, which run once, on the
   raw lowering ([Ir.Fp_algebra], a member of the [Ir.Rewrites] tier); [simplify_llc] -- after
   virtualization and again after scheduling -- is exact. The executed witnesses are in
   [fp_licence_values]. *)
open Base
open Verdict.Claims
open Ll_test
module LL = Ir.Low_level
module Ops = Ir.Ops

let licences =
  let open LL.Fp_licence in
  [
    ("fp_contract", { exact with contract = true });
    ("fp_reassociate", { exact with reassociate = true });
    ("fp_finite_math", { exact with finite_math = true });
  ]

let licence_vars = [ "OCANNL_FP_CONTRACT"; "OCANNL_FP_REASSOCIATE"; "OCANNL_FP_FINITE_MATH" ]

(* Every licence key set to [value] (["true"] / ["false"]) around [f]. *)
let under value f =
  let previous = List.map licence_vars ~f:(fun var -> (var, Stdlib.Sys.getenv_opt var)) in
  List.iter licence_vars ~f:(fun var -> Unix.putenv var value);
  Stdlib.Fun.protect f ~finally:(fun () ->
      List.iter previous ~f:(fun (var, v) -> Unix.putenv var (Option.value v ~default:"")))

let () =
  let mk = node_factory ~first_id:99800 ~dims:[| 1 |] () in
  let input = mk "input" and output = mk "output" in
  let x = get input [| fixed 0 |] in
  let sub = binop Ops.Sub and div = binop Ops.Div in
  (* Each witness is discriminating for its own licence alone. *)
  let cases =
    [
      ("fp_contract", "contraction", add (mul x x) x);
      ("fp_reassociate", "constant reassociation", add (add x (c 2.)) (c 3.));
      ("fp_reassociate", "subtraction re-parenthesizing", add x (sub x x));
      ("fp_reassociate", "multiplication/division interchange", mul x (div x x));
      ("fp_reassociate", "integer-power unrolling", binop Ops.ToPowOf x (c 3.));
      ("fp_finite_math", "zero product", mul (c 0.) x);
      ("fp_finite_math", "positive-zero addition", add x (c 0.));
    ]
  in
  let rhs_of = function LL.Set { llsc; _ } -> llsc | _ -> failwith "lost the assignment" in
  let rewrite licence rhs = rhs_of (Ir.Fp_algebra.rewrite licence (set_at output (fixed 0) rhs)) in
  let exact rhs = rhs_of (LL.simplify_llc [] (set_at output (fixed 0) rhs)) in
  let licence_of key = List.Assoc.find_exn licences key ~equal:String.equal in
  p_all "the post-virtualization simplifier is exact on every floating-point witness" cases
    ~f:(fun (_, _, rhs) -> LL.equal_scalar_t rhs (exact rhs));
  p_all "each licence rewrites its own witnesses" cases ~f:(fun (key, _, rhs) ->
      not (LL.equal_scalar_t rhs (rewrite (licence_of key) rhs)));
  let pairs =
    List.concat_map licences ~f:(fun (key, licence) ->
        List.map cases ~f:(fun (owner, _, rhs) -> (key, licence, owner, rhs)))
  in
  p_all "each licence leaves the other licences' witnesses alone" pairs
    ~f:(fun (key, licence, owner, rhs) ->
      String.equal key owner || LL.equal_scalar_t rhs (rewrite licence rhs));
  let every = ("all", LL.Fp_licence.all) :: licences in
  p_all "the tier member reaches its fixpoint in one application"
    (List.cartesian_product every cases) ~f:(fun ((_, licence), (_, _, rhs)) ->
      let once = rewrite licence rhs in
      LL.equal_scalar_t once (rewrite licence once));
  p "the tier member is disabled with every licence off"
    (under "false" (fun () -> not (Ir.Fp_algebra.enabled ())));
  p_all "the exact zero identities x - (+0) and x + (-0) fold unlicensed"
    [ sub x (c 0.); add x (c (-0.)); add (c (-0.)) x ]
    ~f:(fun rhs -> LL.equal_scalar_t x (exact rhs));
  p "constant folding is exact and runs unlicensed"
    (LL.equal_scalar_t (c 5.) (exact (add (c 2.) (c 3.))));
  let imk = node_factory ~prec:Ops.int64 ~first_id:99900 ~dims:[| 1 |] () in
  let integer_out = imk "integer_out" and integer_input = imk "integer_input" in
  let i = get integer_input [| fixed 0 |] in
  let ibinop op a b = LL.Binop (op, (a, Ops.int64), (b, Ops.int64)) in
  let isimplify rhs = rhs_of (LL.simplify_llc [] (set_at integer_out (fixed 0) rhs)) in
  p "the integer zero identity is exact canonicalization"
    (LL.equal_scalar_t i (isimplify (ibinop Ops.Add i (c 0.))));
  let integer_constants = [ (Ops.Add, 5.); (Ops.Mul, 6.) ] in
  p_all "integer constant reassociation is exact canonicalization" integer_constants
    ~f:(fun (op, combined) ->
      LL.equal_scalar_t
        (isimplify (ibinop op (ibinop op i (c 2.)) (c 3.)))
        (ibinop op (c combined) i));
  p "the tier member leaves integer expressions to the canonicalizer"
    (let rhs = ibinop Ops.Add (ibinop Ops.Add i (c 2.)) (c 3.) in
     LL.equal_scalar_t rhs
       (rhs_of (Ir.Fp_algebra.rewrite LL.Fp_licence.all (set_at integer_out (fixed 0) rhs))));
  (* The reverse of the post-unroll witness the licences once needed a cache tag for: unrolling
     exposes fixed scalar indices only AFTER the source digest was computed, and the schedule's
     re-simplification no longer consults any licence. *)
  let smk = node_factory ~first_id:100000 ~dims:[| 3 |] () in
  let source = smk ~dims:[| 1 |] "cache_source" and target = smk "cache_target" in
  materialize source;
  materialize target;
  let axis = sym () in
  let raw =
    loop ~from_:1 ~upto:2 axis
      (set target [| iter axis |] (add (add (get source [| fixed 0 |]) (embed axis)) (c 3.)))
  in
  let opt = optimize ~name:"af_cache_unroll" raw in
  let scheduled_arm value =
    under value (fun () ->
        let scheduled = Ir.Schedule.apply [ Ir.Schedule.Unroll { axis; materialize = true } ] opt in
        ( Ir.Schedule_cache.digest (Ir.Schedule_cache.canonicalize scheduled),
          Ir.Schedule_cache.numerics_tag () ))
  in
  let code_off, tag_off = scheduled_arm "false" and code_on, tag_on = scheduled_arm "true" in
  p "identical code schedules identically whatever the licences" (String.equal code_off code_on);
  p "the cache numerics tag does not depend on the licences" (String.equal tag_off tag_on);
  (* What the cache keys on instead: the raw tier's rewrite is in the code the digest names. *)
  let digest_arm value ~name raw =
    under value (fun () ->
        Ir.Schedule_cache.digest
          (Ir.Schedule_cache.canonicalize (optimize ~name (Ir.Rewrites.apply raw))))
  in
  let untouched =
    loop ~upto:2 axis (set target [| iter axis |] (mul (get source [| fixed 0 |]) (embed axis)))
  in
  let touched =
    loop ~upto:2 axis
      (set target [| iter axis |] (add (mul (get source [| fixed 0 |]) (embed axis)) (c 3.)))
  in
  p "a routine no licence touches keeps one cache identity"
    (String.equal
       (digest_arm "false" ~name:"af_untouched" untouched)
       (digest_arm "true" ~name:"af_untouched" untouched));
  p "a routine the licences rewrite is separated by its code digest"
    (not
       (String.equal
          (digest_arm "false" ~name:"af_touched" touched)
          (digest_arm "true" ~name:"af_touched" touched)))
