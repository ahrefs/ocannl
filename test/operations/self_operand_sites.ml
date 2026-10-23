(* gh-ocannl-1497: every place the library names one tensor twice as the operands of one operation,
   executed against an oracle computed here on the host.

   A tensor used twice in one step ([m * m], [m +* "a, b; b, c => a, c" m]) used to share its
   projection ids between the occurrences, so a self-product silently computed its diagonal
   (gh-ocannl-1476); [Shape.dealias_operands] now gives every repeated occurrence its own copy of
   the shape. [self_concat] pins the fix on the concatenation and on the two products it was found
   with. This test pins it library-wide, in three parts:

   1. The library's own self-operand sites, DERIVED from the code: the ppxlib parse of every [.ml]
   under [lib/] and [tensor/] that this test's rule globs, read for an infix-operator application
   inside a [%op]/[%cd] extension (structure-level [let%op], expression-level [[%cd ...]]; an [%oc]
   escape inside one is plain OCaml again) whose two operands name the same tensor by an identifier
   or a field path ([x *. x], [centered *. centered], [p.grad * p.grad ~logic:...]), and for the
   concatenation operator, a tuple operand naming one tensor twice ([(w, w) ++^ ...]). An operand is
   read through the einsum annotations the PPX accepts beside it: a spec string before or after it,
   a capture list, a spec identifier before it (the binary einsum operators only), and a
   [~logic]/[~projections] label. The scan's boundary, so a miss past it is a known deferral: a
   self-operand passed through a named function ([TDSL.pointmul x x ()], [where c x x]), one tensor
   bound to two different names, and a repeated COMPOUND operand ([(x + 1) *. (x + 1)], which inside
   [%op] builds two tensors, not one twice) are not read. The scanner's rules are checked on
   synthetic sources first: each form it must report, beside the nearest text it must not (host
   floats outside an extension, a type expression, a string literal, two different operands, an
   [%oc] escape).

   2. One executed case per site, keyed by (file, top-level binding, operator, operand). The claims
   pin the relationship both ways: every site the scan finds has a case, and every case names a site
   the library still has. Each case runs forward AND backward -- the backward of a pointwise
   self-product accumulates two contributions into one gradient -- on inputs that differ in every
   cell and are asymmetric across the axes, against closed forms computed here in double precision.
   These are pointwise or same-spec sites: their equations unify the copies' ids again, so they do
   not depend on the de-aliasing; they are pinned here because until now they were checked only
   through tests of other things.

   3. The operator families where the de-aliasing is load-bearing: the matrix product [*] and every
   binary einsum operator the parser generates specs for ([Einsum_parser]'s list, so a new operator
   fails here until it gets an oracle), each applied to one 3x3 matrix twice. Small integers, so
   every sum is exact and the comparison is exact. With [dealias_operands] reverted these compute
   their diagonal and fail.

   Float readbacks in part 2 are device reductions, so they reach stdout only as two-sided tolerance
   claims ([p_close]); a failure's numbers go to stderr. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module IDX = Train.IDX

(* ---------------------------------------------------------------------------------------------- *)
(* Part 1: the scanner. *)

type site = { file : string; binding : string; op : string; operand : string }

let compare_site a b =
  List.compare String.compare
    [ a.file; a.binding; a.op; a.operand ]
    [ b.file; b.binding; b.op; b.operand ]

let equal_site a b = compare_site a b = 0
let site_string s = Printf.sprintf "%s: %s: %s %s %s" s.file s.binding s.operand s.op s.operand

module Scan = struct
  open Ppxlib

  let binary_einsum_ops = Einsum_parser.binary_operators_with_generated_specs
  let concat_op = Einsum_parser.concat_operator_with_generated_specs
  let is_operator name = (not (String.is_empty name)) && String.mem "!$%&*+-./:<=>?@^|~#" name.[0]

  (* The tensor an identifier or a field path names: [x], [p.grad], [M.t]. *)
  let rec path (e : expression) =
    match e.pexp_desc with
    | Pexp_ident { txt; _ } -> Some (String.concat ~sep:"." (Longident.flatten_exn txt))
    | Pexp_field (r, { txt; _ }) ->
        Option.map (path r) ~f:(fun p ->
            p ^ "." ^ String.concat ~sep:"." (Longident.flatten_exn txt))
    | Pexp_constraint (e, _) -> path e
    | _ -> None

  (* An einsum annotation beside an operand: the spec string or the capture list. *)
  let is_annotation (e : expression) =
    match e.pexp_desc with
    | Pexp_constant (Pconst_string _) -> true
    | Pexp_construct ({ txt = Lident ("::" | "[]"); _ }, _) -> true
    | _ -> false

  (* The tensor an operand names once its annotations are stripped: ["spec" m] and ["spec" m ["d"]]
     (infix spec), [m "spec" ["d"]] (mixfix spec), [spec m] (a spec identifier, only after a binary
     einsum operator), [m ~logic:"..."] ([%cd]'s projections). [None] for anything else. *)
  let operand ~einsum (e : expression) =
    match e.pexp_desc with
    | Pexp_apply (head, args) -> (
        let labels_ok =
          List.for_all args ~f:(fun (l, _) ->
              match l with
              | Nolabel -> true
              | Labelled ("logic" | "projections") -> true
              | Labelled _ | Optional _ -> false)
        in
        let positional =
          List.filter_map args ~f:(function
            | Nolabel, a when not (is_annotation a) -> Some a
            | _ -> None)
        in
        if not labels_ok then None
        else
          match positional with
          | [ a ] when is_annotation head -> path a
          | [ a ] when einsum && Option.is_some (path head) -> path a
          | [] -> path head
          | _ -> None)
    | _ -> path e

  let name_of_pattern (p : pattern) =
    match p.ppat_desc with
    | Ppat_var { txt; _ } -> txt
    | Ppat_constraint ({ ppat_desc = Ppat_var { txt; _ }; _ }, _) -> txt
    | _ -> "_"

  let sites ~file source =
    let found = ref [] in
    let binding = ref "_" and in_ext = ref false in
    let with_ref r v f =
      let saved = !r in
      r := v;
      Exn.protect ~f ~finally:(fun () -> r := saved)
    in
    let record op operand = found := { file; binding = !binding; op; operand } :: !found in
    let walker =
      object (self)
        inherit Ast_traverse.iter as super

        method! structure_item si =
          match si.pstr_desc with
          | Pstr_value (_, vbs) ->
              List.iter vbs ~f:(fun vb ->
                  with_ref binding (name_of_pattern vb.pvb_pat) (fun () -> self#value_binding vb))
          | Pstr_extension (({ txt = "op" | "cd"; _ }, PStr items), _) ->
              with_ref in_ext true (fun () -> List.iter items ~f:self#structure_item)
          | _ -> super#structure_item si

        method! expression e =
          match e.pexp_desc with
          | Pexp_extension ({ txt = "op" | "cd"; _ }, payload) ->
              with_ref in_ext true (fun () -> self#payload payload)
          | Pexp_extension ({ txt = "oc"; _ }, payload) ->
              with_ref in_ext false (fun () -> self#payload payload)
          | Pexp_apply
              ( { pexp_desc = Pexp_ident { txt = Lident op; _ }; _ },
                [ (Nolabel, lhs); (Nolabel, rhs) ] )
            when !in_ext && is_operator op ->
              (if String.equal op concat_op then
                 match lhs.pexp_desc with
                 | Pexp_tuple elems ->
                     let names = List.filter_map elems ~f:(operand ~einsum:false) in
                     List.find_all_dups names ~compare:String.compare
                     |> List.iter ~f:(fun name -> record op name)
                 | _ -> ()
               else
                 let einsum = List.mem binary_einsum_ops op ~equal:String.equal in
                 match (operand ~einsum:false lhs, operand ~einsum rhs) with
                 | Some a, Some b when String.equal a b -> record op a
                 | _ -> ());
              super#expression e
          | _ -> super#expression e
      end
    in
    walker#structure (Parse.implementation (Lexing.from_string source));
    List.dedup_and_sort !found ~compare:compare_site
end

(* The scanner's rules, on synthetic sources: each site it must report, and beside it the nearest
   text it must not. *)
let () =
  let source =
    {|
let%op square x = x *. x
let%op cube x = 0.5 *. (x *. x *. x)
let%op gram m = m +* "a, b; b, c => a, c" m
let%op gram_mixfix m = m +* m "a, b; b, c => a, c" ["c"]
let%op gram_ident ~spec m = m +* spec m
let%op cat w = (w, w) ++^ "s, c; s, d => s, c^d"
let norm n p = [%cd n =+ p.grad * p.grad ~logic:"...|...->...; ...|...->... => |->0"]
let host d = d *. d
type pair = int * int
let%op text x = "x *. x" ^ x
let%op different x y = x *. y
let%op cat_distinct w v = (w, v) ++^ "s, c; s, d => s, c^d"
let%op escaped x = [%oc x *. x]
let%op compound x = (x + 1) *. (x + 1)
let%op through_function x = TDSL.pointmul x x ()
|}
  in
  let got = Scan.sites ~file:"synthetic.ml" source in
  let want =
    [
      ("square", "*.", "x");
      ("cube", "*.", "x");
      ("gram", "+*", "m");
      ("gram_mixfix", "+*", "m");
      ("gram_ident", "+*", "m");
      ("cat", "++^", "w");
      ("norm", "*", "p.grad");
    ]
    |> List.map ~f:(fun (binding, op, operand) -> { file = "synthetic.ml"; binding; op; operand })
    |> List.sort ~compare:compare_site
  in
  p "scanner: reports exactly the synthetic sites, and none of their nearest non-sites"
    (List.equal equal_site got want);
  if not (List.equal equal_site got want) then
    Stdio.eprintf "scanner (not part of the golden): got\n  %s\n%!"
      (String.concat ~sep:"\n  " (List.map got ~f:site_string))

(* The corpus: the [.ml] files the rule passes (its [lib/*.ml] and [tensor/*.ml] globs), read as
   paths relative to the repository root. In the build directory those globs also match dune's
   preprocessed [<module>.pp.ml]: a marshalled AST, recognized by its content (the OCaml AST magic
   number), not by its name, and claimed below to have its source text beside it in the corpus. *)
let library_sites, corpus =
  let files =
    Array.to_list (Sys.get_argv ())
    |> List.tl_exn
    |> List.filter ~f:(fun f -> String.is_suffix f ~suffix:".ml")
    |> List.map ~f:(fun f -> (f, Stdio.In_channel.read_all f))
  in
  let relative f =
    match String.chop_prefix f ~prefix:"../../" with
    | Some r -> r
    | None -> failwith ("self_operand_sites: corpus path outside ../../: " ^ f)
  in
  let asts, sources =
    List.partition_tf files ~f:(fun (_, content) -> String.is_prefix content ~prefix:"Caml1999M")
  in
  let sources = List.map sources ~f:(fun (f, content) -> (relative f, content)) in
  (* Emptiness passes: a build whose globs match no AST has none to skip. *)
  p_all "every binary AST the corpus globs matched is a preprocessed source the corpus parses"
    ~min:0 asts ~f:(fun (f, _) ->
      match String.chop_suffix (relative f) ~suffix:".pp.ml" with
      | Some stem -> List.Assoc.mem sources (stem ^ ".ml") ~equal:String.equal
      | None -> false);
  ( List.concat_map sources ~f:(fun (file, content) -> Scan.sites ~file content)
    |> List.sort ~compare:compare_site,
    List.map sources ~f:fst )

(* ---------------------------------------------------------------------------------------------- *)
(* Part 2: one executed case per library site. *)

let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

(* Two-sided, relative above magnitude one: a misread cell is off by O(1). *)
let tol w = 1e-4 *. Float.max 1. (Float.abs w)

(* Cell [i] of a [dims]-shaped input: distinct in every cell, not symmetric under any axis swap, and
   centered so both signs occur. *)
let input_values n = Array.init n ~f:(fun i -> (Float.of_int ((i * 7 % 11) + i) /. 4.) -. 2.)
let upstream_weights n = Array.init n ~f:(fun i -> Float.of_int (((i * 5) + 3) % 9) -. 4. +. 0.5)
let numel dims = List.fold dims ~init:1 ~f:( * )

(* [x] (differentiable) and the constant [w] over [b | o], and the context after one forward and
   backward of [sum (f x *. w)], with [f x] and [x]'s gradient materialized. *)
let run_loss ~b ~o f =
  let n = numel (b @ o) in
  let xv = input_values n and wv = upstream_weights n in
  let at values idx =
    values.(Array.fold2_exn idx (Array.of_list (b @ o)) ~init:0 ~f:(fun acc i d -> (acc * d) + i))
  in
  let x =
    Operation.init ~l:"x" ~prec:Ir.Ops.single ~b ~o ~f:(at xv) ~grad_spec:Tensor.Require_grad ()
  in
  let w = NTDSL.init ~l:"w" ~prec:Ir.Ops.single ~b ~o ~f:(at wv) () in
  let y = f x in
  let%op loss = (y *. w) ++ "... | ... => | ->0" in
  Train.set_materialized y.Tensor.value;
  Train.set_materialized (grad_of x);
  let update = Train.grad_update loss in
  let ctx = Train.init_params (Context.auto ()) IDX.empty loss in
  let ctx, routine = Context.compile ctx update IDX.empty in
  let ctx = Context.run ctx routine in
  (xv, wv, Context.get_values ctx y.value, Context.get_values ctx (grad_of x))

(* Normalization over the groups [group i] partitions the cells into (gamma = 1, beta = 0): the
   forward, and the input gradient under upstream [g], dx = (g - mean g - y * mean (g y)) / sigma
   per group. *)
let normalization_oracle ~epsilon ~group xv g =
  let n = Array.length xv in
  let groups = 1 + Array.fold (Array.init n ~f:group) ~init:0 ~f:Int.max in
  let mean_of v =
    let sum = Array.create ~len:groups 0. and count = Array.create ~len:groups 0. in
    Array.iteri v ~f:(fun i x ->
        sum.(group i) <- sum.(group i) +. x;
        count.(group i) <- count.(group i) +. 1.);
    Array.map2_exn sum count ~f:( /. )
  in
  let mean = mean_of xv in
  let variance = mean_of (Array.mapi xv ~f:(fun i x -> Float.square (x -. mean.(group i)))) in
  let sigma = Array.map variance ~f:(fun v -> Float.sqrt (v +. epsilon)) in
  let y = Array.mapi xv ~f:(fun i x -> (x -. mean.(group i)) /. sigma.(group i)) in
  let mean_g = mean_of g and mean_gy = mean_of (Array.mapi g ~f:(fun i gi -> gi *. y.(i))) in
  let dx =
    Array.mapi g ~f:(fun i gi ->
        let k = group i in
        (gi -. mean_g.(k) -. (y.(i) *. mean_gy.(k))) /. sigma.(k))
  in
  (y, dx)

let check_forward_backward label (y, dx) (y_want, dx_want) =
  pf_close "%s: the forward matches the host oracle" label y y_want ~tol;
  pf_close "%s: the input gradient matches the host oracle" label dx dx_want ~tol

(* [x *. x *. x] in the tanh-approximated GELU. *)
let gelu_case label =
  let b = [ 2 ] and o = [ 3; 4 ] in
  let xv, wv, y, dx = run_loss ~b ~o Nn_blocks.gelu in
  let k = 0.7978845608028654 and c = 0.044715 in
  let gelu x = 0.5 *. x *. (1. +. Float.tanh (k *. (x +. (c *. x *. x *. x)))) in
  let gelu' x =
    let t = Float.tanh (k *. (x +. (c *. x *. x *. x))) in
    (0.5 *. (1. +. t)) +. (0.5 *. x *. (1. -. (t *. t)) *. k *. (1. +. (3. *. c *. x *. x)))
  in
  check_forward_backward label (y, dx)
    (Array.map xv ~f:gelu, Array.mapi xv ~f:(fun i x -> gelu' x *. wv.(i)))

let epsilon = 0.125

(* [centered *. centered] in the variance: over the output axes, per batch row. *)
let layer_norm_case label =
  let b = [ 3 ] and o = [ 2; 4 ] in
  let ln = Nn_blocks.layer_norm ~label:[ "ln" ] ~epsilon () in
  let xv, wv, y, dx = run_loss ~b ~o ln in
  let per_row = numel o in
  check_forward_backward label (y, dx)
    (normalization_oracle ~epsilon ~group:(fun i -> i / per_row) xv wv)

(* [centered *. centered] in the batch statistics: over the batch (and spatial) axes, per channel.
   The batch statistics are the ones a training step ([~train_step:(Some 0)]) normalizes with. *)
let batch_norm_case ~spatial label =
  let channels = 3 in
  let b, o, make =
    if spatial then ([ 2 ], [ 2; 3; channels ], Nn_blocks.batch_norm2d)
    else ([ 2; 3 ], [ channels ], Nn_blocks.batch_norm1d)
  in
  let layer = make ~label:[ "bn" ] ~epsilon ?momentum:None () in
  let xv, wv, y, dx = run_loss ~b ~o (fun x -> layer ~train_step:(Some 0) x) in
  check_forward_backward label (y, dx)
    (normalization_oracle ~epsilon ~group:(fun i -> i % channels) xv wv)

(* [p.grad * p.grad] in the global gradient norm: two registered parameters of different shapes,
   trained by [sum (p *. c)], so each gradient is its [c] exactly and the norm is sqrt (sum c^2). *)
let grad_l2_norm_case label =
  let c1 = upstream_weights 4 and c2 = Array.map (upstream_weights 6) ~f:(fun v -> v +. 0.25) in
  let p1 = TDSL.param ~values:(input_values 4) "p1" () in
  let p2 =
    let nd =
      Ir.Ndarray.init_array ~debug:"p2" Ir.Ops.single ~dims:[| 2; 3 |] ~padding:None ~f:(fun idx ->
          (input_values 6).((idx.(0) * 3) + idx.(1)))
    in
    TDSL.reshape_param ~l:"p2" ~i:[ 3 ] ~o:[ 2 ] nd ()
  in
  let k1 = NTDSL.init ~l:"c1" ~prec:Ir.Ops.single ~o:[ 4 ] ~f:(fun idx -> c1.(idx.(0))) () in
  let k2 =
    NTDSL.init ~l:"c2" ~prec:Ir.Ops.single ~i:[ 3 ] ~o:[ 2 ]
      ~f:(fun idx -> c2.((idx.(0) * 3) + idx.(1)))
      ()
  in
  let%op loss = (p1 *. k1) ++ "... | ... => | ->0" + ((p2 *. k2) ++ "... | ... -> ... => | ->0") in
  let norm, norm_comp = Train.grad_l2_norm loss in
  List.iter [ p1; p2 ] ~f:(fun p -> Train.set_materialized (grad_of p));
  let step = Ir.Assignments.sequence [ Train.grad_update loss; norm_comp ] in
  let ctx = Train.init_params (Context.auto ()) IDX.empty loss in
  let ctx, routine = Context.compile ctx step IDX.empty in
  let ctx = Context.run ctx routine in
  pf_close "%s: each parameter's gradient is its upstream constant" label
    (Array.append (Context.get_values ctx (grad_of p1)) (Context.get_values ctx (grad_of p2)))
    (Array.append c1 c2) ~tol;
  let sum_sq = Array.fold (Array.append c1 c2) ~init:0. ~f:(fun acc v -> acc +. (v *. v)) in
  pf_close "%s: the norm is the square root of every gradient cell's square, summed" label
    (Context.get_values ctx norm.Tensor.value)
    [| Float.sqrt sum_sq |]
    ~tol

(* The executed case of each library site, keyed by the site it covers. *)
let cases =
  let site file binding op operand = { file; binding; op; operand } in
  [
    (site "lib/nn_blocks.ml" "gelu" "*." "x", gelu_case);
    (site "lib/nn_blocks.ml" "layer_norm" "*." "centered", layer_norm_case);
    (site "lib/nn_blocks.ml" "batch_norm1d" "*." "centered", batch_norm_case ~spatial:false);
    (site "lib/nn_blocks.ml" "batch_norm2d" "*." "centered", batch_norm_case ~spatial:true);
    (site "lib/train.ml" "grad_l2_norm" "*" "p.grad", grad_l2_norm_case);
  ]

let () =
  p_exists "the corpus holds the library's tensor-code sources" corpus ~f:(fun f ->
      String.equal f "lib/nn_blocks.ml");
  Stdio.print_endline "Library self-operand sites:";
  List.iter library_sites ~f:(fun s -> Stdio.printf "  %s\n" (site_string s));
  let covered = List.map cases ~f:fst in
  p_all "every self-operand site in the library has an executed case here" library_sites
    ~f:(fun s -> List.mem covered s ~equal:equal_site);
  p_all "every executed case names a self-operand site the library still has" covered ~f:(fun s ->
      List.mem library_sites s ~equal:equal_site);
  List.iter cases ~f:(fun (s, case) ->
      Tensor.unsafe_reinitialize ();
      case (site_string s))

(* ---------------------------------------------------------------------------------------------- *)
(* Part 3: the operator families where the de-aliasing is load-bearing. *)

let () =
  let n = 3 in
  (* Distinct, and [mv a b <> mv b a]: a product that collapsed onto the diagonal, or read one
     operand transposed, differs from the oracle. *)
  let mv a b = Float.of_int ((a * 5) + (b * b) + 1) in
  let values = Array.init (n * n) ~f:(fun i -> mv (i / n) (i % n)) in
  let reduce ~init ~f ~combine =
    Array.init (n * n) ~f:(fun idx ->
        let a = idx / n and c = idx % n in
        List.fold (List.init n ~f:Fn.id) ~init ~f:(fun acc k -> combine acc (f (mv a k) (mv k c))))
  in
  let sum_of f = reduce ~init:0. ~f ~combine:( +. ) in
  let family ~label ~matrix_axes ~oracle make =
    Tensor.unsafe_reinitialize ();
    let m =
      if matrix_axes then
        Tensor.ndarray ~grad_spec:Tensor.Prohibit_grad values ~batch_dims:[] ~input_dims:[ n ]
          ~output_dims:[ n ] ()
      else
        Tensor.ndarray ~grad_spec:Tensor.Prohibit_grad values ~batch_dims:[] ~input_dims:[]
          ~output_dims:[ n; n ] ()
    in
    let out = make m in
    Train.set_materialized out.Tensor.value;
    let ctx = Train.forward_once (Context.auto ()) out in
    pf_all2 "%s: one matrix as both operands computes the full product, not its diagonal" label
      (Context.get_values ctx out.value)
      oracle ~f:Float.equal
  in
  family ~label:"m * m" ~matrix_axes:true ~oracle:(sum_of ( *. )) (fun m -> TDSL.O.(m * m));
  List.iter Einsum_parser.binary_operators_with_generated_specs ~f:(fun op ->
      let label = Printf.sprintf "m %s \"a, b; b, c => a, c\" m" op in
      let spec = "a, b; b, c => a, c" in
      match op with
      | "+*" ->
          family ~label ~matrix_axes:false ~oracle:(sum_of ( *. )) (fun m ->
              TDSL.einsum spec m m ())
      | "+++" ->
          family ~label ~matrix_axes:false ~oracle:(sum_of ( +. )) (fun m ->
              TDSL.outer_sum spec m m ())
      | "@^+" ->
          family ~label ~matrix_axes:false
            ~oracle:(reduce ~init:Float.neg_infinity ~f:( +. ) ~combine:Float.max) (fun m ->
              TDSL.tropical spec m m ())
      | op -> fail ("no self-operand oracle for the binary einsum operator " ^ op))
