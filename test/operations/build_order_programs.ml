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
   the schedule cache's digest of each lowered segment, and the loss the step computes. The default
   parameter init draws from a stream keyed by the parameter's session id, which the reversed build
   changes, so every parameter is given values by its label instead.

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
}

let label tn = String.concat ~sep:"." tn.Tn.label

let observe ~name ~mint ~rev make =
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
  let captured = ref [] in
  Generated.arm name;
  Generated.arm (name ^ "__seg");
  let ctx, routine =
    Context.compile ~name
      ~lowered_transform:(fun opt ->
        captured := [ opt ];
        [ opt ])
      ctx step IDX.empty
  in
  let lowered = List.hd_exn !captured in
  let ctx = Context.run ctx routine in
  let loss_value = (Context.get_values ctx loss.Tensor.value).(0) in
  let source = if List.length routine.segments > 1 then name ^ "__seg" else name in
  let nodes = Hashtbl.keys lowered.LL.traced_store in
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
    }
  in
  eprintf
    "%s mint %d rev %b (not part of the golden): %d segment(s), %d store nodes, %d params, loss \
     %.9g, creation ranks %s\n\
     %!"
    name mint rev (List.length routine.segments) (List.length nodes) (List.length params) loss_value
    (String.concat ~sep:" " (List.map outcome.order ~f:Int.to_string));
  Context.release ctx;
  outcome

let tol want = 1e-5 *. Float.abs want

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
  p_all (name ^ ": every build computes the same loss") rest ~f:(fun (what, o) ->
      Verdict.within ~what_ran:what (name ^ ": the loss") [| o.loss_value |] [| base.loss_value |]
        ~tol);
  (* First-appearance renaming must not be what makes the sources equal. *)
  p
    (name ^ ": control: a changed program emits another source, and digests apart")
    ((not (String.equal other.source base.source))
    && not (List.equal String.equal other.digests base.digests))

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
