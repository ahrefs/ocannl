(* gh-ocannl-2102: every identifier cc declares in a kernel is either a node's own, a loop index, a
   scope local, or a name the node-name rules reserve -- derived from the rendered kernels, not from
   a hand-kept list.

   A node's code name is minted by [C_syntax.get_code_ident], which gives a label in a reserved
   namespace (the [ident_blacklist], the implementation's [__x] / [_X], the [ocannl_] and [tmma_]
   prefixes, a trailing [__], the scope-local [v<digits>_] spelling) the disambiguated
   [n<id>_<label>] form. Every fixed temporary a renderer declares must sit in one of those
   namespaces, or a user label spelled like it shadows it, or is shadowed: on cc a node labelled
   [tmma_b__] once read the register tile's own uninitialized pointer (gh-ocannl-2012). The rules
   are a list kept by hand, so this test derives the declarations instead:

   1. It renders the three kernel kinds whose temporaries the rules exist for -- register-tiled (an
   f32 tile with full blocks, one with row and column tails, one widening bf16 operands), vectorized
   (elementwise and reducing loops, each with a remainder) and scope-local (a declared local and an
   inlined [Local_scope]) -- and reads each kernel's declarations off its source with
   [Generated.declarations], whose allowlist of declaration forms fails closed: a statement it
   cannot classify is a failure here, not a skipped line.

   2. It sorts them: a parameter must be one of the routine's nodes (counted apart, never probed); a
   variable spelled like a loop index of the routine's code is a loop index (counted apart: the
   rules do not cover loop indices, gh-ocannl-2103); a variable spelled [Low_level.scope_local_name]
   of a scope of the code is a scope local; everything else, typedef names included, is a fixed
   temporary.

   3. It asks the minter itself whether the rules cover each scope local and fixed temporary: one
   probe routine holds a node labelled after each of them, and a covered name is one whose node took
   another code name. A plainly labelled probe node keeping its label is the control that the probe
   can answer "uncovered".

   The negative controls run every time, on the vectorized kernel's source with lines added: three
   unreserved temporaries -- declared plainly, with a type name the reader does not know, and by a
   call of a macro the source defines -- must come back as exactly the uncovered names; and a second
   declarator, a function declaration, a [#define] in a kernel, a call of a token-pasting macro and
   a declaring macro inside an expression must each come back unrecognized.

   cc only: the GPU dialects prefix every node identifier ([t_]), so a temporary there cannot meet a
   node pointer, and the register tile and vectorized renderings are the C backends'. The vector
   width is pinned to 16 bytes so the renderings are the same on every host; the derived names and
   counts go to stderr, not into the golden, as a codegen change may add or drop a temporary without
   breaking the rule. *)

open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Idx = Ir.Indexing
module Tn = Ir.Tnode
module Generated = Test_utils.Generated
open Verdict.Claims

let () = Unix.putenv "OCANNL_CC_VECTOR_BYTES" "16"
let ctx = lazy (Context.cpu ())
let facts = Context.backend_facts_of_name "cc"
let get = Ll_test.get
let it = Ll_test.iter
let fma a b d = LL.Ternop (Ops.FMA, (a, Ops.single), (b, Ops.single), (d, Ops.single))

type kernel = { name : string; o : LL.optimized; nodes : Tn.t list }

(* {1 The kernels} *)

(* One [Tile_mma] over [d += a * b] ([m x k] times [k x n]) in a lane loop, its serial micro-kernel
   the fallback. *)
let tile ~name ~first_id ~m ~n ~k ~ab_prec =
  let mk ~prec ~dims label =
    let tn = Ll_test.node_factory ~prec ~first_id ~dims () (name ^ "_" ^ label) in
    Ll_test.materialize tn;
    tn
  in
  let d = mk ~prec:Ops.single ~dims:[| m; n |] "d" in
  let a = mk ~prec:ab_prec ~dims:[| m; k |] "a" in
  let b = mk ~prec:ab_prec ~dims:[| k; n |] "b" in
  let i = Ll_test.sym () and j = Ll_test.sym () and l = Ll_test.sym () in
  let fallback =
    Ll_test.loop_n i m @@ Ll_test.loop_n j n @@ Ll_test.loop_n l k
    @@ Ll_test.set d
         [| it i; it j |]
         (fma (get a [| it i; it l |]) (get b [| it l; it j |]) (get d [| it i; it j |]))
  in
  let lane = Ll_test.sym () in
  let origin = [| Ll_test.fixed 0; Ll_test.fixed 0 |] in
  let o =
    Ll_test.optimize_scoped ~materialized:[ d; a; b ] ~name ~raw:fallback
      (LL.For_loop
         {
           index = lane;
           from_ = 0;
           to_ = 0;
           axis = LL.Workgroup;
           body =
             Ll_test.tile_mma ~m ~n ~k ~lane ~d:(d, origin) ~a:(a, origin) ~b:(b, origin) fallback;
         })
  in
  { name; o; nodes = [ d; a; b ] }

(* Two elementwise loops and two reductions, each retyped [Vectorized], over an extent with a
   remainder past every lane count. *)
let vectorized ~first_id =
  let width = 18 in
  let name = "dt_vectorized" in
  let mk = Ll_test.node_factory ~first_id ~dims:[| width |] () in
  let mk1 = Ll_test.node_factory ~first_id:(first_id + 50) ~dims:[| 1 |] () in
  let a = mk "dt_vec_a" and b = mk "dt_vec_b" and x = mk "dt_vec_x" and y = mk "dt_vec_y" in
  let s1 = mk1 "dt_vec_s1" and s2 = mk1 "dt_vec_s2" in
  let nodes = [ a; b; x; y; s1; s2 ] in
  List.iter nodes ~f:Ll_test.materialize;
  let i = Ll_test.sym () and j = Ll_test.sym () and k = Ll_test.sym () and l = Ll_test.sym () in
  let zero = Ll_test.fixed 0 in
  let llc =
    LL.unflat_lines
      [
        Ll_test.loop_n i width
          (Ll_test.set_at x (it i)
             (Ll_test.add (Ll_test.mul (get a [| it i |]) (get b [| it i |])) (Ll_test.c 1.)));
        Ll_test.loop_n j width
          (Ll_test.set_at y (it j)
             (Ll_test.add (get a [| it j |]) (Ll_test.mul (Ll_test.c 3.) (get b [| it j |]))));
        Ll_test.loop_n k width
          (Ll_test.set_at s1 zero (Ll_test.add (get s1 [| zero |]) (get a [| it k |])));
        Ll_test.loop_n l width
          (Ll_test.set_at s2 zero (Ll_test.add (get s2 [| zero |]) (get b [| it l |])));
      ]
  in
  let o =
    Ir.Schedule.apply
      (List.map [ i; j; k; l ] ~f:(fun axis -> Ir.Schedule.Retype { axis; ty = LL.Vectorized }))
      (Ll_test.optimize ~materialized:nodes ~name llc)
  in
  { name; o; nodes }

(* A declared local and an inlined [Local_scope] accumulating over an inner loop, each read by the
   store, reaching the backend as written through the scoped seam (optimization would fold
   single-assignment locals away, and codegen inlines a single-assignment scope as an
   expression). *)
let scope_local ~first_id =
  let n = 8 in
  let name = "dt_scope_local" in
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let one = mk "dt_loc_one" and two = mk "dt_loc_two" in
  let a = mk "dt_loc_a" and x = mk "dt_loc_x" in
  List.iter [ a; x ] ~f:Ll_test.materialize;
  List.iter [ one; two ] ~f:Ll_test.virtualize;
  let id1 = LL.get_scope one in
  let id2 = LL.get_scope two in
  let i = Ll_test.sym () and r = Ll_test.sym () in
  let at = it i in
  let v1 = Ll_test.add (get a [| at |]) (Ll_test.c 1.) in
  let v2 = Ll_test.mul (get a [| at |]) (Ll_test.c 2.) in
  let raw = Ll_test.loop_n i n (Ll_test.set_at x at (Ll_test.add v1 v2)) in
  let accumulated =
    Ll_builders.scope_of id2
      (LL.unflat_lines
         [
           LL.Set_local (id2, Ll_test.c 0.);
           Ll_test.loop_n r 2 (LL.Set_local (id2, Ll_test.add (LL.Get_local id2) (get a [| at |])));
         ])
  in
  let scoped =
    Ll_test.loop_n i n
      (LL.unflat_lines
         (Ll_test.define_local id1 v1
         @ [ Ll_test.set_at x at (Ll_test.add (LL.Get_local id1) accumulated) ]))
  in
  let o = Ll_test.optimize_scoped ~materialized:[ a; x ] ~name ~raw scoped in
  { name; o; nodes = [ a; x ] }

(* {1 Sorting the declarations} *)

let loop_idents (llc : LL.t) =
  let acc = ref [] in
  Ll_test.walk llc ~on_stmt:(function
    | LL.For_loop { index; _ } -> acc := Idx.symbol_ident index :: !acc
    | _ -> ());
  !acc

let scope_ids (llc : LL.t) =
  let acc = ref [] in
  Ll_test.walk llc
    ~on_scope:(fun id -> acc := id :: !acc)
    ~on_stmt:(function LL.Declare_local { id; _ } -> acc := id :: !acc | _ -> ());
  List.dedup_and_sort !acc ~compare:LL.Scope_id.compare

let code_name tn =
  match tn.Tn.code_name with
  | Some name -> name
  | None -> failwith "declared_temporaries: a node of a compiled routine has no code name"

type sorted = {
  all : string list;  (** Every declared identifier. *)
  params : string list;
  stray_params : string list;  (** Parameters that are no node of the routine. *)
  loop_indices : string list;
  scope_locals : string list;
  temporaries : string list;
  unrecognized : (int * string) list;
}

let sort ~(k : kernel) ~source =
  let { Generated.declared; unrecognized } = Generated.declarations source in
  let node_idents = List.map k.nodes ~f:(fun tn -> Generated.node_ident ~facts (code_name tn)) in
  let loops = loop_idents k.o.LL.llc in
  let locals =
    List.map (scope_ids k.o.LL.llc) ~f:(fun id -> LL.scope_local_name ~node:(code_name id.LL.tn) id)
  in
  let mem l x = List.mem l x ~equal:String.equal in
  let params, body =
    List.partition_tf declared ~f:(fun d ->
        match d.Generated.site with Generated.Parameter -> true | _ -> false)
  in
  let params = List.map params ~f:(fun d -> d.Generated.ident) in
  let names =
    List.map body ~f:(fun d -> d.Generated.ident) |> List.dedup_and_sort ~compare:String.compare
  in
  {
    all = List.map declared ~f:(fun d -> d.Generated.ident);
    params = List.filter params ~f:(mem node_idents);
    stray_params = List.filter params ~f:(Fn.non (mem node_idents));
    loop_indices = List.filter names ~f:(mem loops);
    scope_locals = List.filter names ~f:(mem locals);
    temporaries = List.filter names ~f:(fun x -> not (mem loops x || mem locals x));
    unrecognized;
  }

(* {1 The minter's answer} *)

(* One routine with a node labelled after each name, written once; a name is covered when its node
   took another code name. [probe_plain] is a label no rule reserves, so it must keep its name. *)
let plain_label = "probe_plain"

let covered_by_rules names =
  let mk = Ll_test.node_factory ~first_id:2102900 ~dims:[| 1 |] () in
  let probes = List.map (plain_label :: names) ~f:(fun label -> (label, mk label)) in
  List.iter probes ~f:(fun (_, tn) -> Ll_test.materialize tn);
  let llc =
    LL.unflat_lines
      (List.map probes ~f:(fun (_, tn) -> Ll_test.set_at tn (Ll_test.fixed 0) (Ll_test.c 1.)))
  in
  let name = "dt_probe" in
  let o = Ll_test.optimize ~materialized:(List.map probes ~f:snd) ~name llc in
  ignore (Ll_test.link ~ctx:(Lazy.force ctx) ~name o : Context.t * Context.routine);
  List.map probes ~f:(fun (label, tn) -> (label, not (String.equal (code_name tn) label)))

(* {1 The negative controls} *)

(* [source] with [defines] ahead of it and [stmts] added as the first statements after each kernel's
   main-logic marker. *)
let with_statements ?(defines = []) source stmts =
  defines
  @ (String.split_lines source
    |> List.concat_map ~f:(fun l ->
        if String.is_substring l ~substring:Ir.C_syntax.main_logic_marker then l :: stmts else [ l ])
    )
  |> String.concat ~sep:"\n"

(* Two macros the controls define ahead of a kernel: one declaring its second argument, one pasting
   the name it declares. *)
let control_defines = [ "#define DT_DECLARE(T, x) T x = 0"; "#define DT_PASTE(x) float dt_##x = 0" ]

(* Unreserved temporaries the control adds: plainly, with a type name the reader does not know, and
   through a macro call. Sorted, as the uncovered names are compared against them. *)
let unreserved_controls =
  [
    ("dt_macro_tmp", "    DT_DECLARE(float, dt_macro_tmp);");
    ("dt_unknown_tmp", "    dt_unknown_t dt_unknown_tmp = 0;");
    ("dt_unreserved_tmp", "    float dt_unreserved_tmp = 0.0f;");
  ]

let () =
  Utils.settings.output_debug_files_in_build_directory <- true;
  Generated.init ~backend_name:"cc";
  let kernels =
    [
      ( "register tile, f32 full blocks",
        tile ~name:"dt_tile_full" ~first_id:2102000 ~m:8 ~n:8 ~k:8 ~ab_prec:Ops.single );
      ( "register tile, f32 row and column tails",
        tile ~name:"dt_tile_tails" ~first_id:2102100 ~m:6 ~n:19 ~k:5 ~ab_prec:Ops.single );
      ( "register tile, bf16 operands",
        tile ~name:"dt_tile_bf16" ~first_id:2102200 ~m:7 ~n:8 ~k:2 ~ab_prec:Ops.bfloat16 );
      ("vectorized loops", vectorized ~first_id:2102300);
      ("scope locals", scope_local ~first_id:2102400);
    ]
  in
  let rendered =
    List.map kernels ~f:(fun (what, k) ->
        let _, routine = Ll_test.link ~ctx:(Lazy.force ctx) ~name:k.name k.o in
        (what, k, routine, Generated.read k.name))
  in
  (* Each kernel renders the form it stands for, so no leg passes over a declined rendering. *)
  List.iter rendered ~f:(fun (what, k, routine, source) ->
      if String.is_prefix k.name ~prefix:"dt_tile" then
        p (what ^ ": the tile renders tensorized") (Ll_test.rendered_tensorized routine.Context.mma)
      else if String.equal k.name "dt_vectorized" then
        p
          (what ^ ": all four loops render the vector form")
          (List.length
             (String.substr_index_all source ~may_overlap:false ~pattern:"{ /* Vectorized ")
          = 4));
  let sorted = List.map rendered ~f:(fun (what, k, _, source) -> (what, k, sort ~k ~source)) in
  let control_source =
    let _, k, _, source =
      List.find_exn rendered ~f:(fun (_, k, _, _) -> String.equal k.name "dt_vectorized")
    in
    (k, source)
  in
  let control =
    let k, source = control_source in
    sort ~k
      ~source:
        (with_statements ~defines:control_defines source (List.map unreserved_controls ~f:snd))
  in
  let to_probe =
    List.concat_map sorted ~f:(fun (_, _, s) -> s.scope_locals @ s.temporaries)
    @ control.temporaries
    |> List.dedup_and_sort ~compare:String.compare
  in
  let answers = covered_by_rules to_probe in
  let covered name = List.Assoc.find_exn answers name ~equal:String.equal in
  p "the probe keeps an unreserved label's name, so it can answer uncovered"
    (not (covered plain_label));
  List.iter sorted ~f:(fun (what, k, s) ->
      Stdio.eprintf
        "%s (%s; not part of the golden): %d node parameters, loop indices [%s], scope locals \
         [%s], fixed temporaries [%s]\n"
        what k.name (List.length s.params)
        (String.concat ~sep:" " s.loop_indices)
        (String.concat ~sep:" " s.scope_locals)
        (String.concat ~sep:" " s.temporaries);
      List.iter s.unrecognized ~f:(fun (line, text) ->
          Stdio.eprintf "  unrecognized at line %d: %s\n" line text);
      p_empty (what ^ ": every statement takes an allowlisted form") ~over:s.all s.unrecognized;
      List.iter s.stray_params ~f:(Stdio.eprintf "  stray parameter: %s\n");
      p_empty (what ^ ": every parameter is a node of the routine") ~over:s.all s.stray_params;
      p (what ^ ": declares node parameters") (not (List.is_empty s.params));
      List.iter s.temporaries ~f:(fun x ->
          if not (covered x) then Stdio.eprintf "  unreserved temporary: %s\n" x);
      if String.equal k.name "dt_scope_local" then
        p_all
          (what ^ ": declares both scope locals, each reserved")
          s.scope_locals ~min:2 ~f:covered);
  List.iter sorted ~f:(fun (what, k, s) ->
      if not (String.equal k.name "dt_scope_local") then
        p (what ^ ": declares fixed temporaries") (not (List.is_empty s.temporaries)));
  p_all "every fixed temporary the kernels declare is reserved"
    (List.concat_map sorted ~f:(fun (_, _, s) -> s.temporaries)
    |> List.dedup_and_sort ~compare:String.compare)
    ~f:covered;
  (* Negative controls. *)
  p_empty "control: the added temporaries read as allowlisted declarations" ~over:control.all
    control.unrecognized;
  p
    "control: the added unreserved temporaries -- plain, of an unknown type, declared by a macro \
     call -- are exactly what comes back uncovered"
    (List.equal String.equal
       (List.filter control.temporaries ~f:(Fn.non covered))
       (List.map unreserved_controls ~f:fst));
  let k, source = control_source in
  let refused stmt =
    not
      (List.is_empty
         (sort ~k ~source:(with_statements ~defines:control_defines source [ stmt ])).unrecognized)
  in
  p "control: a second declarator is unrecognized" (refused "    float dt_y = 0, dt_z = 1;");
  p "control: a function declaration is unrecognized" (refused "    float dt_f(int dt_p);");
  p "control: a macro definition in a kernel is unrecognized" (refused "#define dt_macro 1");
  p "control: a call of a macro that pastes is unrecognized" (refused "    DT_PASTE(pasted);");
  p "control: a declaring macro inside an expression is unrecognized"
    (refused "    dt_s = DT_DECLARE(float, dt_q);")
