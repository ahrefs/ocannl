(* gh-ocannl-998: each floating-point licence controls only its own rewrites. *)
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
      ("fp_finite_math", "signed-zero addition", add x (c 0.));
    ]
  in
  let simplify licence rhs =
    match LL.simplify_llc ~licence [] (set_at output (fixed 0) rhs) with
    | LL.Set { llsc; _ } -> llsc
    | _ -> failwith "simplification lost the assignment"
  in
  let licence_of key = List.Assoc.find_exn licences key ~equal:String.equal in
  p_all "the exact simplifier preserves each floating-point witness" cases ~f:(fun (_, _, rhs) ->
      LL.equal_scalar_t rhs (simplify LL.Fp_licence.exact rhs));
  p_all "each licence rewrites its own witnesses" cases ~f:(fun (key, _, rhs) ->
      not (LL.equal_scalar_t rhs (simplify (licence_of key) rhs)));
  let pairs =
    List.concat_map licences ~f:(fun (key, licence) ->
        List.map cases ~f:(fun (owner, _, rhs) -> (key, licence, owner, rhs)))
  in
  p_all "each licence leaves the other licences' witnesses alone" pairs
    ~f:(fun (key, licence, owner, rhs) ->
      String.equal key owner || LL.equal_scalar_t rhs (simplify licence rhs));
  p_all "unqualified simplification follows the configured licences" cases ~f:(fun (_, _, rhs) ->
      LL.equal_scalar_t
        (simplify (LL.Fp_licence.of_config ()) rhs)
        (match LL.simplify_llc [] (set_at output (fixed 0) rhs) with
        | LL.Set { llsc; _ } -> llsc
        | _ -> failwith "simplification lost the assignment"));
  p "constant folding is exact and runs unlicensed"
    (LL.equal_scalar_t (c 5.) (simplify LL.Fp_licence.exact (add (c 2.) (c 3.))));
  let imk = node_factory ~prec:Ops.int64 ~first_id:99900 ~dims:[| 1 |] () in
  let integer_out = imk "integer_out" and integer_input = imk "integer_input" in
  let i = get integer_input [| fixed 0 |] in
  let ibinop op a b = LL.Binop (op, (a, Ops.int64), (b, Ops.int64)) in
  let isimplify rhs =
    match LL.simplify_llc ~licence:LL.Fp_licence.exact [] (set_at integer_out (fixed 0) rhs) with
    | LL.Set { llsc; _ } -> llsc
    | _ -> failwith "integer simplification lost the assignment"
  in
  p "the integer zero identity runs unlicensed"
    (LL.equal_scalar_t i (isimplify (ibinop Ops.Add i (c 0.))));
  let integer_constants = [ (Ops.Add, 5.); (Ops.Mul, 6.) ] in
  p_all "integer constant reassociation runs unlicensed" integer_constants ~f:(fun (op, combined) ->
      LL.equal_scalar_t
        (isimplify (ibinop op (ibinop op i (c 2.)) (c 3.)))
        (ibinop op (c combined) i));
  (* Unrolling exposes fixed scalar indices only AFTER the source digest was computed. *)
  let smk = node_factory ~first_id:100000 ~dims:[| 3 |] () in
  let source = smk ~dims:[| 1 |] "cache_source" and target = smk "cache_target" in
  materialize source;
  materialize target;
  let axis = sym () in
  let raw =
    loop ~from_:1 ~upto:2 axis
      (set target [| iter axis |] (add (add (get source [| fixed 0 |]) (embed axis)) (c 3.)))
  in
  let var = "OCANNL_FP_REASSOCIATE" in
  let previous = Stdlib.Sys.getenv_opt var in
  let under value f =
    Unix.putenv var value;
    Stdlib.Fun.protect f ~finally:(fun () -> Unix.putenv var (Option.value previous ~default:""))
  in
  let cache_arm value =
    under value (fun () ->
        let opt = optimize ~name:"af_cache_unroll" raw in
        let base = Ir.Schedule_cache.digest (Ir.Schedule_cache.canonicalize opt) in
        let scheduled = Ir.Schedule.apply [ Ir.Schedule.Unroll { axis; materialize = true } ] opt in
        ( base,
          Ir.Schedule_cache.digest (Ir.Schedule_cache.canonicalize scheduled),
          Ir.Schedule_cache.numerics_tag () ))
  in
  let base_off, code_off, tag_off = cache_arm "false" in
  let base_on, code_on, tag_on = cache_arm "true" in
  p "the unroll witness has identical pre-schedule digests across licences"
    (String.equal base_off base_on);
  p "unrolling exposes licence-dependent constant reassociation"
    (not (String.equal code_off code_on));
  p "the cache numerics tag separates identical sources with different scheduled algebra"
    (not (String.equal tag_off tag_on));
  let original_tag =
    String.prefix
      (Stdlib.Digest.to_hex (Stdlib.Digest.string (Ir.Numerics.fingerprint (Ir.Numerics.get ()))))
      8
  in
  p "the all-licensed cache tag remains compatible with existing entries"
    (String.equal original_tag (under "true" Ir.Schedule_cache.numerics_tag))
