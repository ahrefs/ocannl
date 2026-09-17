(* gh-ocannl-774: proof soundness is checked against enumerated finite images, and init elision is
   checked through real Assignments lowering against explicit neutral initialization. *)
open Base
module Idx = Ir.Indexing
module Aff = Ir.Affine
module A = Ir.Assignments
module Tn = Ir.Tnode
module LL = Ir.Low_level
open Verdict.Claims

let aff symbols offset = Idx.affine ~symbols ~offset
let sym () = Idx.get_symbol ()

let proj components lhs_dims project_lhs : Idx.projections =
  {
    components;
    lhs_dims;
    project_lhs;
    rhs_dims = [||];
    project_rhs = [||];
    extent_syms = [];
    debug_info = { spec = "surjectivity"; derived_for = Sexp.Atom ""; trace = [] };
  }

let () =
  let i = sym () and j = sym () in
  p "normalization cancels repeated symbols"
    (Idx.equal_axis_index (aff [ (2, i); (-2, i) ] 7) (Idx.Fixed_idx 7));
  p "normalization collapses a unit coefficient"
    (Idx.equal_axis_index (aff [ (0, j); (2, i); (-1, i) ] 0) (Idx.Iterator i));
  p "normalization keeps a nonzero offset"
    (match aff [ (1, i) ] 2 with Idx.Affine _ -> true | _ -> false);
  p "normalization is independent of term order"
    (Idx.equal_axis_index (aff [ (2, j); (3, i) ] 4) (aff [ (3, i); (2, j) ] 4));
  let encoded =
    Sexp.List
      [
        Sexp.Atom "Affine";
        Sexp.List
          [ Sexp.Atom "symbols"; Sexp.List [ Sexp.List [ Sexp.Atom "1"; Idx.sexp_of_symbol i ] ] ];
        Sexp.List [ Sexp.Atom "offset"; Sexp.Atom "0" ];
      ]
  in
  p "deserialization also normalizes affine indices"
    (Idx.equal_axis_index (Idx.axis_index_of_sexp encoded) (Idx.Iterator i));
  let check name dims comps idcs expected =
    p name (Bool.equal (Aff.is_surjective (proj comps dims idcs)) expected)
  in
  check "iterator must reach the last cell" [| 4 |] [| [ (3, i) ] |] [| Idx.Iterator i |] false;
  check "positive offset leaves a prefix gap" [| 4 |] [| [ (3, i) ] |] [| aff [ (1, i) ] 1 |] false;
  check "negative offset cannot prove coverage" [| 4 |]
    [| [ (4, i) ] |]
    [| aff [ (1, i) ] (-1) |]
    false;
  check "reversal covers every cell" [| 4 |] [| [ (4, i) ] |] [| aff [ (-1, i) ] 3 |] true;
  check "stride holes are refused" [| 7 |] [| [ (4, i) ] |] [| aff [ (2, i) ] 0 |] false;
  check "dense overlapping sum is surjective" [| 5 |]
    [| [ (3, i) ]; [ (3, j) ] |]
    [| aff [ (1, i); (1, j) ] 0 |]
    true;
  check "diagonal dependence leaves holes" [| 3; 3 |]
    [| [ (3, i) ] |]
    [| Idx.Iterator i; Idx.Iterator i |]
    false;
  check "out-of-axis offsets cannot cancel through flattening" [| 2; 3 |]
    [| [ (2, i) ]; [ (3, j) ] |]
    [| aff [ (1, i) ] (-1); aff [ (1, j) ] 3 |]
    false;
  check "flattened Sub_axis covers the whole buffer" [| 2; 3 |]
    [| [ (6, i) ] |]
    [| Idx.Sub_axis; Idx.Iterator i |]
    true;
  check "trailing Sub_axis stride leaves holes" [| 3; 2 |]
    [| [ (3, i) ] |]
    [| Idx.Iterator i; Idx.Sub_axis |]
    false;
  check "an empty loop cannot cover a nonempty target" [| 1 |]
    [| [ (0, i) ] |]
    [| Idx.Fixed_idx 0 |] false;
  check "empty target coverage is vacuous" [| 0 |] [| [ (0, i) ] |] [| Idx.Iterator i |] true;
  check "static symbols are refused" [| 3 |] [||] [| Idx.Iterator i |] false;
  check "overflow cannot prove coverage" [| Int.max_value; 2 |]
    [| [ (2, i) ] |]
    [| Idx.Iterator i; Idx.Fixed_idx 0 |]
    false;
  let a = sym () and b = sym () and c = sym () and d = sym () in
  let components = [| [ (2, a); (3, b) ]; [ (1, c); (2, d) ] |] in
  check "two independent concat axes cover their product" [| 5; 3 |] components
    [| Idx.Concat [ a; b ]; Idx.Concat [ c; d ] |]
    true;
  check "repeated concat axis preserves diagonal dependence" [| 5; 5 |] components
    [| Idx.Concat [ a; b ]; Idx.Concat [ a; b ] |]
    false;
  check "partial and complete concat coordinates cannot be mixed" [| 2; 5 |] components
    [| Idx.Iterator a; Idx.Concat [ a; b ] |]
    false;
  let square_components = [| [ (2, a); (2, b) ]; [ (2, c); (2, d) ] |] in
  let square = proj square_components [| 4; 4 |] [| Idx.Concat [ a; b ]; Idx.Concat [ c; d ] |] in
  let corner x y = [| Idx.Iterator x; Idx.Iterator y |] in
  let diagonal_blocks = { square with project_rhs = [| corner a c; corner b d |] } in
  p "two diagonal RHS blocks do not cover the off-diagonal blocks"
    (not (Aff.is_surjective diagonal_blocks));
  let full_blocks =
    { square with project_rhs = [| corner a c; corner a d; corner b c; corner b d |] }
  in
  p "four RHS blocks cover two concatenated axes" (Aff.is_surjective full_blocks);
  let guarded =
    {
      (proj [| [ (3, i) ] |] [| 3 |] [| Idx.Iterator i |]) with
      extent_syms = [ (Some i, fst (Idx.get_static_symbol ~static_range:3 Idx.Empty)) ];
    }
  in
  p "runtime extent guard cannot justify full-buffer init elision" (not (Aff.is_surjective guarded));
  (* Independent enumerator: a false positive misses at least one concrete cell. Includes negative
     coefficients, offsets, repeated symbols, gaps, collisions and target bounds. *)
  let cases =
    List.concat_map (List.range 1 5) ~f:(fun n ->
        List.concat_map (List.range 1 5) ~f:(fun m ->
            List.concat_map (List.range (-3) 4) ~f:(fun ci ->
                List.concat_map (List.range (-3) 4) ~f:(fun cj ->
                    List.concat_map (List.range (-2) 4) ~f:(fun offset ->
                        List.map (List.range 1 13) ~f:(fun dim -> (n, m, ci, cj, offset, dim)))))))
  in
  p_all "every admitted bounded affine image covers its target" cases
    ~f:(fun (n, m, ci, cj, offset, dim) ->
      let pr = proj [| [ (n, i) ]; [ (m, j) ] |] [| dim |] [| aff [ (ci, i); (cj, j) ] offset |] in
      if not (Aff.is_surjective pr) then true
      else
        let image = Array.create ~len:dim false in
        for x = 0 to n - 1 do
          for y = 0 to m - 1 do
            let v = (ci * x) + (cj * y) + offset in
            if v >= 0 && v < dim then image.(v) <- true
          done
        done;
        Array.for_all image ~f:Fn.id)

let next_id = ref 25000

let node dims label =
  Int.incr next_id;
  let n =
    Tn.create (Tn.Specified Ir.Ops.single) ~id:!next_id ~label:[ label ]
      ~unpadded_dims:(lazy dims)
      ~padding:(lazy None)
      ()
  in
  Tn.update_memory_mode n Tn.On_device (Site "99:test-setup");
  n

let execute_case ?make_projection ?(reverse = false) name ~n ~m ~dims ~indices ~address ~elide =
  let expected = Array.create ~len:(Array.fold dims ~init:1 ~f:( * )) 0. in
  let values = Array.init (n * m) ~f:(fun k -> Float.of_int (11 + (10 * (k / m)) + (k % m))) in
  for x = 0 to n - 1 do
    for y = 0 to m - 1 do
      let a = address x y in
      expected.(a) <- expected.(a) +. values.((x * m) + y)
    done
  done;
  let run explicit_init =
    let i = sym () and j = sym () in
    let src = node [| n; m |] (name ^ "_src") and dst = node dims (name ^ "_dst") in
    let projection =
      {
        (proj [| [ (n, i) ]; [ (m, j) ] |] dims (indices i j)) with
        rhs_dims = [| [| n; m |] |];
        project_rhs = [| [| Idx.Iterator i; Idx.Iterator j |] |];
      }
    in
    let projection = match make_projection with None -> projection | Some f -> f () in
    let scatter =
      A.Accum_op
        {
          initialize_neutral = not explicit_init;
          accum = Ir.Ops.Add;
          lhs = (if reverse then src else dst);
          rhs =
            (if reverse then A.Rev_sides { op = Ir.Ops.Identity; lhses = [| A.Node dst |] }
             else if Array.exists projection.components ~f:(fun comp -> List.length comp > 1) then
               A.Block { op = Ir.Ops.Identity; rhses = [| A.Node src |] }
             else A.Unop { op = Ir.Ops.Identity; rhs = A.Node src });
          projections = lazy projection;
          projections_debug = name;
        }
    in
    let asgns =
      if explicit_init then
        A.Seq (A.Fetch { array = dst; fetch_op = Constant 0.; dims = lazy dims }, scatter)
      else scatter
    in
    let llc = A.to_low_level asgns in
    let zeros = ref 0 in
    Ll_test.walk llc ~on_stmt:(function
      | LL.Zero_out tn when Tn.equal tn dst -> Int.incr zeros
      | _ -> ());
    if not explicit_init then p (name ^ ": initialization decision") (Bool.equal (!zeros = 0) elide);
    let ctx = Context.auto () in
    Stdio.eprintf "%s: backend=%s\n%!" name (Context.backend_name ctx);
    let ctx = Context.set_values ctx src values in
    let ctx = Context.set_values ctx dst (Array.create ~len:(Array.length expected) 12345.) in
    let comp = { A.asgns; embedded_nodes = Set.singleton (module Tn) dst } in
    let ctx, routine =
      Context.compile
        ~name:(name ^ if explicit_init then "_reference" else "_candidate")
        ctx comp Idx.Empty
    in
    let ctx = Context.run ctx routine in
    Context.get_values ctx dst
  in
  let actual = run false and reference = run true in
  p_all2 (name ^ ": executed values match the enumerated scatter") actual expected ~f:Float.equal;
  p_all2 (name ^ ": elision agrees with explicit initialization") actual reference ~f:Float.equal

let () =
  execute_case "coverage_reverse" ~n:3 ~m:2 ~dims:[| 6 |]
    ~indices:(fun i j -> [| aff [ (-2, i); (-1, j) ] 5 |])
    ~address:(fun i j -> 5 - (2 * i) - j)
    ~elide:true;
  execute_case "coverage_prefix_gap" ~n:3 ~m:2 ~dims:[| 7 |]
    ~indices:(fun i j -> [| aff [ (2, i); (1, j) ] 1 |])
    ~address:(fun i j -> 1 + (2 * i) + j)
    ~elide:false;
  execute_case "coverage_stride_gap" ~n:3 ~m:2 ~dims:[| 8 |]
    ~indices:(fun i j -> [| aff [ (3, i); (1, j) ] 0 |])
    ~address:(fun i j -> (3 * i) + j)
    ~elide:false;
  execute_case "coverage_collision" ~n:3 ~m:2 ~dims:[| 4 |]
    ~indices:(fun i j -> [| aff [ (1, i); (1, j) ] 0 |])
    ~address:(fun i j -> i + j)
    ~elide:false;
  execute_case "coverage_sub_axis" ~n:3 ~m:2 ~dims:[| 2; 3 |]
    ~indices:(fun i j -> [| Idx.Sub_axis; aff [ (2, i); (1, j) ] 0 |])
    ~address:(fun i j -> (2 * i) + j)
    ~elide:true;
  execute_case "coverage_diagonal" ~n:3 ~m:1 ~dims:[| 3; 3 |]
    ~indices:(fun i _ -> [| Idx.Iterator i; Idx.Iterator i |])
    ~address:(fun i _ -> 4 * i)
    ~elide:false;
  execute_case "coverage_two_concats" ~n:5 ~m:3 ~dims:[| 5; 3 |]
    ~indices:(fun i j -> [| Idx.Iterator i; Idx.Iterator j |])
    ~address:(fun i j -> (3 * i) + j)
    ~elide:true
    ~make_projection:(fun () ->
      let a = sym () and b = sym () and c = sym () and d = sym () in
      let indices = [| Idx.Concat [ a; b ]; Idx.Concat [ c; d ] |] in
      {
        (proj [| [ (2, a); (3, b) ]; [ (1, c); (2, d) ] |] [| 5; 3 |] indices) with
        rhs_dims = [| [| 5; 3 |] |];
        project_rhs = [| indices |];
      })

let () =
  let i = sym () in
  let src = node [| 6 |] "padded_flat_src" in
  Int.incr next_id;
  let dst =
    Tn.create (Tn.Specified Ir.Ops.single) ~id:!next_id ~label:[ "padded_flat_dst" ]
      ~unpadded_dims:(lazy [| 2; 3 |])
      ~padding:
        (lazy (Some ([| Ir.Ops.{ left = 1; right = 0 }; Ir.Ops.{ left = 0; right = 0 } |], 0.)))
      ()
  in
  let projection =
    {
      (proj [| [ (6, i) ] |] [| 2; 3 |] [| Idx.Sub_axis; Idx.Iterator i |]) with
      rhs_dims = [| [| 6 |] |];
      project_rhs = [| [| Idx.Iterator i |] |];
    }
  in
  let refused =
    try
      ignore
        (A.to_low_level
           (A.Accum_op
              {
                initialize_neutral = true;
                accum = Ir.Ops.Add;
                lhs = dst;
                rhs = A.Unop { op = Ir.Ops.Identity; rhs = A.Node src };
                projections = lazy projection;
                projections_debug = "padded_flat";
              })
          : LL.t);
      false
    with Utils.User_error message -> String.is_substring message ~substring:"Flattened Sub_axis"
  in
  p "padded flattened layout is refused explicitly" refused

let () =
  let make_projection reverse () =
    let a = sym () and b = sym () and c = sym () and d = sym () in
    let components = [| [ (2, a); (2, b) ]; [ (2, c); (2, d) ] |] in
    let full = [| Idx.Concat [ a; b ]; Idx.Concat [ c; d ] |] in
    let partial = [| Idx.Iterator a; Idx.Iterator c |] in
    if reverse then
      {
        (proj components [| 2; 2 |] partial) with
        rhs_dims = [| [| 4; 4 |] |];
        project_rhs = [| full |];
      }
    else
      {
        (proj components [| 4; 4 |] full) with
        rhs_dims = [| [| 2; 2 |] |];
        project_rhs = [| partial |];
      }
  in
  let run reverse name =
    execute_case name ~n:2 ~m:2 ~dims:[| 4; 4 |]
      ~indices:(fun i j -> [| Idx.Iterator i; Idx.Iterator j |])
      ~address:(fun i j -> (4 * i) + j)
      ~elide:false ~reverse ~make_projection:(make_projection reverse)
  in
  run false "coverage_sparse_block";
  run true "coverage_sparse_reverse"

let () =
  let name = "coverage_dependent_reverse_selector" in
  let run explicit_init =
    let a = sym () and b = sym () in
    let src = node [| 4; 2 |] (name ^ "_src") in
    let dsts = Array.init 2 ~f:(fun k -> node [| 2 |] (name ^ Int.to_string k)) in
    let projection =
      {
        (proj [| [ (2, a); (2, b) ] |] [| 4; 2 |] [| Idx.Concat [ a; b ]; Idx.Iterator b |]) with
        rhs_dims = [| [| 2 |]; [| 2 |] |];
        project_rhs = [| [| Idx.Iterator a |]; [| Idx.Iterator b |] |];
      }
    in
    let scatter =
      A.Accum_op
        {
          initialize_neutral = not explicit_init;
          accum = Ir.Ops.Add;
          lhs = src;
          rhs = A.Rev_sides { op = Ir.Ops.Identity; lhses = Array.map dsts ~f:(fun n -> A.Node n) };
          projections = lazy projection;
          projections_debug = name;
        }
    in
    let asgns =
      if explicit_init then
        Array.fold dsts ~init:scatter ~f:(fun rest dst ->
            A.Seq (A.Fetch { array = dst; fetch_op = Constant 0.; dims = lazy [| 2 |] }, rest))
      else scatter
    in
    let zeros = ref 0 in
    Ll_test.walk (A.to_low_level asgns) ~on_stmt:(function
      | LL.Zero_out tn when Tn.equal tn dsts.(0) -> Int.incr zeros
      | _ -> ());
    if not explicit_init then p (name ^ ": unavailable source retains initialization") (!zeros > 0);
    let ctx = Context.auto () in
    Stdio.eprintf "%s: backend=%s\n%!" name (Context.backend_name ctx);
    let ctx = Context.set_values ctx src [| 11.; 12.; 21.; 22.; 31.; 32.; 41.; 42. |] in
    let ctx =
      Array.fold dsts ~init:ctx ~f:(fun ctx dst -> Context.set_values ctx dst [| 12345.; 12345. |])
    in
    let comp = { A.asgns; embedded_nodes = Set.of_array (module Tn) dsts } in
    let ctx, routine =
      Context.compile
        ~name:(name ^ if explicit_init then "_reference" else "_candidate")
        ctx comp Idx.Empty
    in
    let ctx = Context.run ctx routine in
    Array.concat_map dsts ~f:(Context.get_values ctx)
  in
  let actual = run false and reference = run true in
  p_all2 (name ^ ": executed selected values") actual [| 0.; 0.; 31.; 42. |] ~f:Float.equal;
  p_all2 (name ^ ": explicit initialization agrees") actual reference ~f:Float.equal
