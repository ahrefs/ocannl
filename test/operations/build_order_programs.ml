(* gh-ocannl-1974: what a compile decides for a convolution and for a transformer block is a
   function of the program -- not of how many nodes the session minted before it, nor of the order
   its parameters and data were created in.

   [flip_order_minting] part 1 runs the uid-shift check on one MLP step. Here two more programs -- a
   padded two-conv training step (padding resets, window projections) and a two-block transformer
   encoder step (attention, layer norms sharing labels across layers) -- are each built four ways:
   after minting 0 or 3 unrelated nodes (a uid shift: every Tnode-keyed hash table iterates
   differently, creation order is kept), and with their layers and data created first-to-last or
   last-to-first (creation order reversed: every [Set]/[Map] over nodes and every [Tn.compare] sort
   iterates differently). The layers are applied in one order either way, so it is one program.

   Compared across the builds: the emitted source up to minted identifiers, every node's placement,
   the schedule cache's digest of each lowered segment, the loss the step computes, and the memory
   layout (gh-ocannl-2051): the pool rank and offset of every node the step's link lays out, the
   scorer's footprint and the allocator's link delta. The default parameter init draws from a stream
   keyed by the parameter's session id, which the reversed build changes, so every parameter is
   given values by its label instead.

   Last, hand-built: the schedule cache's digest lists a program's companion nodes (workgroup-shared
   tiles, simdgroup fragments, swizzled and pipelined tiles) by their position in the code, not in
   the order the schedule minted them. *)

open Base
open Stdio
open Ocannl
open Nn_blocks.DSL_modules
module IDX = Train.IDX
module Tn = Ir.Tnode
module LL = Ir.Low_level
module SC = Ir.Schedule_cache
module Generated = Test_utils.Generated
open Verdict.Claims

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let () = Generated.init ~backend_name

type program = {
  loss : Tensor.t;
  roles : (string * Tn.t) list;
      (** Nodes the fixture names, whose relative creation order the build's [rev] reverses. *)
}

(* Creates [makers]' values first-to-last, or last-to-first under [rev], and returns them in the
   list's order either way. *)
let create ~rev makers =
  let made = List.map (if rev then List.rev makers else makers) ~f:(fun f -> f ()) in
  if rev then List.rev made else made

let conv ~rev ~channels () =
  let c1_out, c2_out = channels in
  let data () =
    let x_nd =
      Ir.Ndarray.init_array ~debug:"xs" Ir.Ops.single ~dims:[| 2; 6; 6; 2 |] ~padding:None
        ~f:(fun idx ->
          Float.of_int ((Array.fold idx ~init:7 ~f:(fun acc i -> (acc * 31) + i) % 13) - 6) /. 4.)
    in
    `Data (TDSL.rebatch ~l:"xs" x_nd ())
  in
  let layer label out_channels () =
    `Layer (Nn_blocks.conv2d ~label:[ label ] ~use_padding:true ~out_channels ())
  in
  match create ~rev [ data; layer "c1" c1_out; layer "c2" c2_out ] with
  | [ `Data xs; `Layer c1; `Layer c2 ] ->
      let%op h1 = relu (c1 xs) in
      let%op h2 = c2 h1 in
      let%op loss = (h2 *. h2) ++ "...|... => |->0" in
      { loss; roles = [ ("xs", xs.Tensor.value); ("h1", h1.Tensor.value) ] }
  | _ -> assert false

let transformer ~rev () =
  let data () =
    `Data
      (TDSL.range_of_shape ~label:[ "x" ] ~batch_dims:[ 2; 8 ] ~input_dims:[] ~output_dims:[ 16 ] ())
  in
  let block l () =
    `Layer
      (Nn_blocks.transformer_encoder_block ~label:[ l ] ~num_heads:2 ~d_k:8 ~d_v:8 ~d_ff:32 ()
         ~train_step:None)
  in
  match create ~rev [ data; block "l0"; block "l1" ] with
  | [ `Data x; `Layer b0; `Layer b1 ] ->
      let%op x' = x /. 256. in
      let h = b1 (b0 x') in
      (* Not [h *. h]: a layer norm's output has that sum whatever the weights are. *)
      let%op loss = (h *. x') ++ "... | ... => 0" in
      { loss; roles = [ ("x", x.Tensor.value); ("scaled", x'.Tensor.value) ] }
  | _ -> assert false

type outcome = {
  source : string;
  placements : string list;  (** ["<label> <mode>"], sorted: session ids differ across orders. *)
  digests : string list;
  loss_value : float;
  order : int list;  (** The program's parameters' creation ranks, in label order. *)
  hash_order : string list;
  param_labels : string list;
  layout : string list;
      (** ["<label> <pool rank> <offset>"] of every node the step's own link laid out, sorted. *)
  param_layout : string list;  (** Held parameters, laid out by initialization. *)
  fresh : int;  (** How many store nodes the step's own link laid out. *)
  unpositioned : string list;
      (** The step's newly laid out nodes its raw code never accesses: their layout order falls back
          to creation order. *)
  scored : string;  (** The scorer's footprint, [Memory_budget.footprint] before the compile. *)
  linked : string;  (** The allocator's delta, [Context.link_delta] of the compiled segments. *)
  liveness_planned : bool;
}

let label tn = String.concat ~sep:"." tn.Tn.label

(* ["<label> <pool rank> <offset>"] for each node of [nodes] [ctx] holds a buffer for, sorted: pool
   ids count per device across contexts, so a pool is named by its rank among the pools these nodes
   occupy. *)
let layout_of ctx nodes =
  let located =
    List.filter_map nodes ~f:(fun tn ->
        Option.map (Context.buffer_location ctx tn) ~f:(fun loc -> (label tn, loc)))
  in
  let pools =
    List.map located ~f:(fun (_, (pool, _)) -> pool) |> List.dedup_and_sort ~compare:Int.compare
  in
  let rank pool = fst (List.findi_exn pools ~f:(fun _ p -> p = pool)) in
  List.map located ~f:(fun (l, (pool, offset)) -> Printf.sprintf "%s %d %d" l (rank pool) offset)
  |> List.sort ~compare:String.compare

let observe ?(scheduled = false) ~name ~mint ~rev make =
  for _ = 1 to mint do
    ignore (NTDSL.O.(!.1.0) : Tensor.t)
  done;
  Utils.settings.fixed_state_for_init <- Some 3;
  Tensor.unsafe_reinitialize ();
  let { loss; roles } = make ~rev () in
  let step = Train.grad_update loss in
  let ctx = Train.init_params (Context.auto ()) IDX.empty loss in
  let params =
    Set.to_list loss.Tensor.params
    |> List.map ~f:(fun t -> t.Tensor.value)
    |> List.sort ~compare:(fun a b -> String.compare (label a) (label b))
  in
  let ctx =
    List.foldi params ~init:ctx ~f:(fun k ctx tn ->
        Context.set_values ctx tn
          (Array.init (Tn.num_elems tn) ~f:(fun i ->
               0.25 *. Float.sin (Float.of_int ((17 * (k + 1)) + (3 * i))))))
  in
  let scored = Memory_budget.footprint ctx step IDX.empty in
  let pre_compile = ctx in
  let captured = ref [] in
  Generated.arm name;
  Generated.arm (name ^ "__seg");
  let ctx, routine =
    Context.compile ~name
      ~lowered_transform:(fun opt ->
        captured := [ opt ];
        if scheduled then
          Ir.Schedule.maybe_default_schedules ~facts:(Context.backend_facts ctx)
            ~limits:(Context.hardware_limits ctx) ~static_indices:[] opt
        else [ opt ])
      ctx step IDX.empty
  in
  let lowered = List.hd_exn !captured in
  let linked = Context.link_delta pre_compile lowered routine.segments in
  let ctx = Context.run ctx routine in
  let loss_value = (Context.get_values ctx loss.Tensor.value).(0) in
  let source = if List.length routine.segments > 1 then name ^ "__seg" else name in
  let nodes = Hashtbl.keys lowered.LL.traced_store in
  let fresh =
    List.filter nodes ~f:(fun tn -> Context.mem ctx tn && not (Context.mem pre_compile tn))
  in
  let rank tn = List.count (params @ List.map roles ~f:snd) ~f:(fun o -> Tn.compare o tn < 0) in
  let outcome =
    {
      source = Generated.normalize_minted ~facts:(Context.backend_facts ctx) (Generated.read source);
      placements =
        List.map nodes ~f:(fun tn ->
            Printf.sprintf "%s %s" (label tn)
              (Tn.debug_memory_mode (Tn.Placements.get (Context.placements ctx) tn)))
        |> List.sort ~compare:String.compare;
      digests = List.map routine.segments ~f:(fun seg -> SC.digest (SC.canonicalize seg));
      loss_value;
      order = List.map (params @ List.map roles ~f:snd) ~f:rank;
      hash_order = List.map nodes ~f:label;
      param_labels = List.map params ~f:label;
      layout = layout_of ctx fresh;
      param_layout = layout_of ctx params;
      fresh = List.length fresh;
      unpositioned =
        (let positions = LL.first_access_positions lowered.LL.source in
         List.filter fresh ~f:(fun tn -> not (Hashtbl.mem positions tn)) |> List.map ~f:label);
      scored = Sexp.to_string (LL.sexp_of_footprint scored);
      linked = Sexp.to_string (Context.sexp_of_link_delta linked);
      liveness_planned = linked.Context.ld_liveness_planned;
    }
  in
  eprintf
    "%s mint %d rev %b (not part of the golden): %d segment(s), %d store nodes, %d params, loss \
     %.9g, creation ranks %s, scored %s, linked %s\n\
     %!"
    name mint rev (List.length routine.segments) (List.length nodes) (List.length params) loss_value
    (String.concat ~sep:" " (List.map outcome.order ~f:Int.to_string))
    outcome.scored outcome.linked;
  Context.release ctx;
  outcome

let tol want = 1e-5 *. Float.abs want

(* gh-ocannl-2051: the layout planners receive a routine's nodes in its raw code's access order, not
   in creation order -- so every held node sits at the same pool rank and offset, and the scorer and
   the allocator report the same footprint, whichever order the program's nodes were created in. *)
let layout_claims name ~base ~rest =
  List.iter rest ~f:(fun (what, o) ->
      if not (List.equal String.equal o.layout base.layout) then
        eprintf "%s: the %s build's layout differs:\n%s\n----\n%s\n" name what
          (String.concat ~sep:"\n" base.layout)
          (String.concat ~sep:"\n" o.layout));
  p
    (name ^ ": control: the step lays out nodes of its own")
    (base.fresh > 0 && List.length base.layout = base.fresh);
  p_all (name ^ ": every build lays the step's nodes out at the same pool rank and offset") rest
    ~f:(fun (_, o) -> List.equal String.equal o.layout base.layout);
  p
    (name ^ ": control: initialization lays out every parameter")
    ((not (List.is_empty base.param_labels))
    && List.length base.param_layout = List.length base.param_labels);
  List.iter rest ~f:(fun (what, o) ->
      if not (List.equal String.equal o.param_layout base.param_layout) then
        eprintf "%s: the %s build's parameter layout differs:\n%s\n----\n%s\n" name what
          (String.concat ~sep:"\n" base.param_layout)
          (String.concat ~sep:"\n" o.param_layout));
  p_all (name ^ ": every build lays parameters out at the same pool rank and offset") rest
    ~f:(fun (_, o) -> List.equal String.equal o.param_layout base.param_layout);
  p_all (name ^ ": every build scores the same footprint") rest ~f:(fun (_, o) ->
      String.equal o.scored base.scored);
  p_all (name ^ ": every build's link delta is the same") rest ~f:(fun (_, o) ->
      String.equal o.linked base.linked);
  List.iter (("base", base) :: rest) ~f:(fun (what, o) ->
      if not (List.is_empty o.unpositioned) then
        eprintf "%s: the %s build lays out nodes its raw code never accesses: %s\n" name what
          (String.concat ~sep:" " o.unpositioned));
  (* Emptiness is the passing case: a node the raw code accesses never falls back to creation order.
     The control above keeps the step's own nodes nonempty. *)
  p_all (name ^ ": every node the step lays out is ordered by the step's raw code")
    (("base", base) :: rest) ~f:(fun (_, o) -> List.is_empty o.unpositioned)

let check name make ~changed =
  let build ?(make = make) ~mint ~rev tag =
    eprintf "%s, %s build:\n%!" name tag;
    observe ~name:("bop_" ^ name) ~mint ~rev make
  in
  let base = build ~mint:0 ~rev:false "m0" in
  let shifted = build ~mint:3 ~rev:false "m3" in
  let reversed = build ~mint:0 ~rev:true "m0r" in
  let both = build ~mint:3 ~rev:true "m3r" in
  let other = build ~make:changed ~mint:0 ~rev:false "changed" in
  let rest = [ ("shifted", shifted); ("reversed", reversed); ("shifted and reversed", both) ] in
  p
    (name ^ ": control: the uid shift changes a node-keyed table's iteration order")
    (not (List.equal String.equal base.hash_order shifted.hash_order));
  p
    (name ^ ": control: the shift keeps the nodes' creation order")
    (List.equal Int.equal base.order shifted.order);
  p
    (name ^ ": control: the reversed build creates its nodes in another order")
    (not (List.equal Int.equal base.order reversed.order));
  p_pairwise_distinct
    (name ^ ": the parameters' labels tell them apart")
    base.param_labels ~equal:String.equal ~to_string:Fn.id;
  p
    (name ^ ": the step computes a loss off zero")
    Float.(is_finite base.loss_value && base.loss_value > 1e-6);
  p
    (name ^ ": control: the changed program computes another loss")
    Float.(other.loss_value > 1.001 * base.loss_value || other.loss_value < 0.999 * base.loss_value);
  List.iter rest ~f:(fun (what, o) ->
      if not (String.equal o.source base.source) then
        eprintf "%s: the %s build's source differs:\n%s\n----\n%s\n" name what base.source o.source);
  p_all (name ^ ": every build emits the same source") rest ~f:(fun (_, o) ->
      String.equal o.source base.source);
  p_all (name ^ ": every build places every node the same") rest ~f:(fun (_, o) ->
      List.equal String.equal o.placements base.placements);
  p_all (name ^ ": every build's segments digest the same") rest ~f:(fun (_, o) ->
      List.equal String.equal o.digests base.digests);
  layout_claims name ~base ~rest;
  p_all (name ^ ": every build computes the same loss") rest ~f:(fun (what, o) ->
      Verdict.within ~what_ran:what (name ^ ": the loss") [| o.loss_value |] [| base.loss_value |]
        ~tol);
  (* First-appearance renaming must not be what makes the sources equal. *)
  p
    (name ^ ": control: a changed program emits another source, and digests apart")
    ((not (String.equal other.source base.source))
    && not (List.equal String.equal other.digests base.digests));
  (* Laid out alone, the two conv steps coincide: every size pads to the same 32-byte slots. *)
  p
    (name ^ ": control: a changed program lays out or links differently")
    ((not (List.equal String.equal other.layout base.layout))
    || not (String.equal other.linked base.linked))

(* Same graph, same seed and counters: initializing its two draws together in forward order must
   agree bit for bit with initializing each separately in creation order. *)
let () =
  Tensor.unsafe_reinitialize ();
  Tensor.set_random_seed ~seed:42 ();
  let first_draw = TDSL.param "init_first" ~output_dims:[ 8 ] () in
  let second_draw = TDSL.param "init_second" ~output_dims:[ 8 ] () in
  let%op out = second_draw + first_draw in
  let together = Tensor.init_params out in
  let separate =
    Ir.Assignments.sequence [ Tensor.init_params first_draw; Tensor.init_params second_draw ]
  in
  let evaluate name comp nodes =
    let base = Context.auto () in
    let ctx, routine = Context.compile ~name base comp IDX.empty in
    let ctx = Context.run ctx routine in
    let values = List.map nodes ~f:(Context.get_values ctx) in
    Context.release ctx;
    Context.release base;
    values
  in
  let actual = evaluate "bop_init_program" together [ first_draw.value; second_draw.value ] in
  let reference = evaluate "bop_init_creation" separate [ first_draw.value; second_draw.value ] in
  p_all "initialization order preserves fixed-program random draws bit for bit"
    (List.zip_exn actual reference) ~f:(fun (a, b) -> Array.equal Float.equal a b);
  let writes =
    Ir.Assignments.fold_leaves together.asgns ~init:[] ~f:(fun acc leaf ->
        Set.to_list (Ir.Assignments.collect_written leaf) @ acc)
    |> List.rev
  in
  let position node = fst (List.findi_exn writes ~f:(fun _ n -> Tn.equal n node)) in
  p "independent parameters initialize in forward operand order"
    (position second_draw.value < position first_draw.value);
  Tensor.unsafe_reinitialize ();
  let source = TDSL.param "init_source" ~output_dims:[ 8 ] () in
  let shared = NTDSL.pointmul source (NTDSL.number 2.) () in
  let first = TDSL.param "init_owner" ~param_init:(NTDSL.add shared (NTDSL.number 1.)) () in
  let second = TDSL.param "init_reader" ~param_init:(NTDSL.add shared (NTDSL.number 3.)) () in
  let%op out = second + first in
  let values =
    evaluate "bop_init_dependencies" (Tensor.init_params out)
      [ source.value; first.value; second.value ]
  in
  match values with
  | [ source; first; second ] ->
      p_all "initializer dependencies precede readers despite reversed forward access"
        (List.zip_exn [ first; second ] [ 1.; 3. ])
        ~f:(fun (actual, bias) ->
          Verdict.within "initializer value" actual
            (Array.map source ~f:(fun x -> (2. *. x) +. bias))
            ~tol:(fun _ -> 1e-6))
  | _ -> assert false

(* Slice parents are materialization outputs of collect_nodes_guess_output, but still execution
   dependencies. A precision mismatch forces copies so reading before the owner cannot hide behind
   aliasing the buffer the owner later fills. *)
let () =
  Verdict.case "slice initializer dependencies" (fun () ->
      Tensor.unsafe_reinitialize ();
      let index, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
      let data =
        NTDSL.init ~l:"init_slice_data" ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 8 ]
          ~f:(fun i -> Float.of_int ((10 * i.(0)) + i.(1) + 1))
          ()
      in
      let shared = NTDSL.pointmul data (NTDSL.number 2.) () in
      let first = TDSL.param "init_slice_owner" ~param_init:(NTDSL.slice index shared) () in
      let second = TDSL.param "init_slice_reader" ~param_init:(NTDSL.slice index shared) () in
      List.iter [ first; second ] ~f:(fun t -> Tn.update_prec t.Tensor.value Ir.Ops.half);
      let%op out = second + first in
      let base = Context.auto () in
      let ctx, routine =
        Context.compile ~name:"bop_init_slices" base (Tensor.init_params out) bindings
      in
      IDX.find_exn routine.bindings index := 1;
      let init_ctx = Context.run ctx routine in
      p_all "slice initializer regression exercises copies" [ first; second ] ~f:(fun t ->
          not (Ir.Assignments.slice_alias_eligible ~array:t.Tensor.value ~sliced:shared.value));
      Train.set_materialized out.value;
      let ctx, forward =
        Context.compile ~name:"bop_init_slices_value" init_ctx (Tensor.consume_forward_code out)
          bindings
      in
      IDX.find_exn forward.bindings index := 1;
      let ctx = Context.run ctx forward in
      p "slice parent initialization precedes every copying reader"
        (Array.equal Float.equal
           (Context.get_values ctx out.value)
           (Array.init 8 ~f:(fun i -> Float.of_int (44 + (4 * i)))));
      Context.release ctx;
      Context.release init_ctx;
      Context.release base)

let () =
  check "conv" (conv ~channels:(3, 2)) ~changed:(conv ~channels:(2, 3));
  check "transformer" transformer ~changed:(fun ~rev () ->
      let p = transformer ~rev () in
      let loss = p.loss in
      let%op loss = loss + loss in
      { p with loss })

(* === The digest's companion sections ===

   [first[i] = i + 1; second[i] = first[i] + 1], with [second] created before [first] under [rev],
   and both marked as the schedule marks its tiles. On master the sections listed their members in
   the order of the set or map holding them -- uid order -- so the two builds digested apart. *)
let companion_digest ~rev ?(depth = 2) ?(swizzle = LL.Swizzle_elem) () =
  let open Ll_test in
  let mk = node_factory ~first_id:(if rev then 1_974_200 else 1_974_100) ~dims:[| 4 |] () in
  let first, second =
    if rev then
      let second = mk "second" in
      (mk "first", second)
    else
      let first = mk "first" in
      (first, mk "second")
  in
  List.iter [ first; second ] ~f:materialize;
  let s = sym () and t = sym () in
  let llc =
    seq
      (loop_n s 4 (set_at first (iter s) (tick s)))
      (loop_n t 4 (set_at second (iter t) (add (get first [| iter t |]) (c 1.))))
  in
  let o = optimize ~name:(if rev then "bop_companions_rev" else "bop_companions") llc in
  let both = Set.of_list (module Tn) [ first; second ] in
  let map_of data = Map.of_alist_exn (module Tn) [ (first, data); (second, data) ] in
  let marked =
    {
      o with
      LL.workgroup_shared = both;
      simdgroup_fragments = both;
      swizzled = map_of swizzle;
      pipelined = map_of { LL.pt_depth = depth; pt_rotor = s };
    }
  in
  (Tn.compare first second < 0, SC.digest (SC.canonicalize marked))

let () =
  let first_older, digest = companion_digest ~rev:false () in
  let first_older', digest' = companion_digest ~rev:true () in
  p "companions: control: the two builds create the pair in opposite orders"
    (first_older && not first_older');
  p "companions: the digest is the same whichever was created first" (String.equal digest digest');
  p "companions: control: another pipeline depth digests apart"
    (not (String.equal digest (snd (companion_digest ~rev:false ~depth:3 ()))));
  p "companions: control: another swizzle kind digests apart"
    (not (String.equal digest (snd (companion_digest ~rev:false ~swizzle:LL.Swizzle_b128 ()))))

(* === A decisive bump-packing pair (gh-ocannl-2051) ===

   [small =: y + y; big =: x + x], with [small] 4 bytes and [big] 64, the two outputs created in
   either order and written in one. Bump packing pads each offset to [Ops.buffer_alignment] and the
   pool ends at the last item, so the order the planner receives the pair in shows in the offsets
   AND in the pool's size: small first ends at alignment + 64, big first at 64 + 4. Laid out in the
   code's order, both builds put [small] first. *)
let make_tensor label vals =
  let open Bigarray in
  let ga = Genarray.create Float32 c_layout [| Array.length vals |] in
  Array.iteri vals ~f:(fun i v -> Genarray.set ga [| i |] v);
  Tensor.term
    ~init_data:(Reshape (Ir.Ndarray.as_array Ir.Ops.Single ga))
    ~grad_spec:Tensor.Prohibit_grad ~label:[ label ] ~batch_dims:[] ~input_dims:[]
    ~output_dims:[ Array.length vals ]
    ()

let bump_pair ~rev =
  Tensor.unsafe_reinitialize ();
  let x = make_tensor "x" (Array.init 16 ~f:(fun i -> Float.of_int ((7 * i % 11) - 5))) in
  let y = make_tensor "y" [| 2.5 |] in
  let make_small () = make_tensor "small" [| 0. |] in
  let make_big () = make_tensor "big" (Array.create ~len:16 0.) in
  let small, big =
    if rev then
      let big = make_big () in
      (make_small (), big)
    else
      let small = make_small () in
      (small, make_big ())
  in
  let%cd comp =
    small =: y + y;
    big =: x + x
  in
  let ctx = Context.auto () in
  let scored = Memory_budget.footprint ctx comp IDX.empty in
  let pre_compile = ctx in
  let captured = ref [] in
  let ctx, routine =
    Context.compile ~name:"bop_bump"
      ~lowered_transform:(fun opt ->
        captured := [ opt ];
        [ opt ])
      ctx comp IDX.empty
  in
  let linked = Context.link_delta pre_compile (List.hd_exn !captured) routine.segments in
  let ctx = Context.run ctx routine in
  let loc t = Option.value_exn (Context.buffer_location ctx t.Tensor.value) in
  let values = (Context.get_values ctx small.value, Context.get_values ctx big.value) in
  let result =
    ( Tn.compare small.value big.value < 0,
      loc small,
      loc big,
      scored.LL.fp_working,
      linked.Context.ld_working_bytes,
      values,
      Array.map (Context.get_values ctx x.value) ~f:(fun v -> v +. v) )
  in
  Context.release ctx;
  result

let () =
  let small_older, (sp, so), (bp, bo), scored, linked, (sv, bv), want_big = bump_pair ~rev:false in
  let small_older', (sp', so'), (bp', bo'), scored', linked', (sv', bv'), _ = bump_pair ~rev:true in
  let align = Ir.Ops.buffer_alignment in
  p "bump: control: the two builds create the outputs in opposite orders"
    (small_older && not small_older');
  p "bump: control: the outputs share one working pool" (sp = bp && sp' = bp');
  p "bump: the output written first sits at offset 0 in either build" (so = 0 && so' = 0);
  p "bump: the other output follows at the alignment in either build" (bo = align && bo' = align);
  p "bump: the scored working pool is the same in either build, alignment + 64 bytes"
    (scored = align + 64 && scored' = scored);
  p "bump: the linked working pool is the same in either build, alignment + 64 bytes"
    (linked = align + 64 && linked' = linked);
  p "bump: both builds compute small = y + y" (Float.equal sv.(0) 5. && Float.equal sv'.(0) 5.);
  p "bump: both builds compute big = x + x"
    (Array.equal Float.equal bv want_big
    && Array.equal Float.equal bv' want_big
    && Array.exists want_big ~f:(fun v -> Float.(v <> 0.)))

(* === The liveness arena (gh-ocannl-2051) ===

   Under [buffer_aliasing] the working pool is laid out by the greedy arena, which breaks equal-size
   ties by input order. The transformer step's first-to-last and last-to-first builds again, under
   the default schedules (the liveness fold over the unscheduled step plans nothing), last in this
   executable: the gate is read per call, and flipped through the environment. The conv step has no
   arena candidate: its delta is the gradients, which it exports. *)
let () =
  Unix.putenv "OCANNL_BUFFER_ALIASING" "true";
  let leg name make =
    let build ~rev tag =
      eprintf "%s, %s build under buffer_aliasing:\n%!" name tag;
      observe ~scheduled:true ~name:("bop_arena_" ^ name) ~mint:0 ~rev make
    in
    let base = build ~rev:false "m0" in
    let reversed = build ~rev:true "m0r" in
    let name = name ^ " under buffer_aliasing" in
    p (name ^ ": control: the step's delta is liveness-planned") base.liveness_planned;
    layout_claims name ~base ~rest:[ ("reversed", reversed) ];
    p
      (name ^ ": the reversed build computes the same loss")
      (Verdict.within ~what_ran:"reversed" (name ^ ": the loss") [| reversed.loss_value |]
         [| base.loss_value |] ~tol)
  in
  leg "transformer" transformer
