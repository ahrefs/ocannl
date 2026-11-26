(* gh-ocannl-2472: narrow-float ([half], [bfloat16], [fp8]) routine-log arguments widen to [float]
   through the backend's own conversion before guarding and before [styled_log_arg].

   Under [debug_log_from_routines=true], [log_value_args] used to hand narrow-float expressions
   straight to [(g ? d : 0)] and [styled_log_arg] ([(int)(d * 1000.0)] under
   [prefer_backend_uniformity=true], or bare [d] to variadic [%g] without it). On HIP and CUDA
   [__half], [__hip_bfloat16], [__nv_bfloat16], [__hip_fp8_e5m2] and [__nv_fp8_e5m2] fail
   compilation on operator [*] / [?:] ambiguity or variadic passing; on cc and Metal the software
   storage types ([uint16_t], [uint8_t]) compiled as integers and logged raw bit encodings instead
   of float values.

   This test compiles, executes and checks the stream-logged values of four routine shapes -- [Set],
   bounds-guarded [Where] inside [Set], kept [Set_local] scope, and [Set_from_vec] RNG draw -- at
   [half], [bfloat16] and [fp8], under both [prefer_backend_uniformity=true] ([%de-3]) and
   [prefer_backend_uniformity=false] ([%g] on cc/CUDA/HIP; [%de-3] on Metal). Every input and output
   is exact in f16, bf16 and e5m2 fp8, so executed values and logged decimal strings agree on every
   backend. *)

open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Idx = Ir.Indexing
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let u4 = Ops.uint4x32
let get = Ll_test.get
let n = 4
let init_out = Array.create ~len:n (-4.0)
let precisions = [ ("half", Ops.half); ("bf16", Ops.bfloat16); ("fp8", Ops.fp8) ]
let modes = [ ("uniform", true); ("printf", false) ]
let uses_scaled_int ~uniform = uniform || String.equal backend_name "metal"
let expected_text ~uniform ~scaled ~plain = if uses_scaled_int ~uniform then scaled else plain

let stream_log_lines () =
  let dir = Utils.log_files_dir () in
  Stdlib.Sys.readdir dir |> Array.to_list
  |> List.filter ~f:(String.is_suffix ~suffix:".log")
  |> List.sort ~compare:String.compare
  |> List.concat_map ~f:(fun f -> Stdio.In_channel.read_lines (Stdlib.Filename.concat dir f))

(* A value line of a store or local assignment, ["<lhs>{=<old>} = <new> = <expr>"]: its [<new>]. *)
let stored_value line =
  match String.substr_index line ~pattern:"} = " with
  | None -> None
  | Some i -> (
      let rest = String.drop_prefix line (i + 4) in
      match String.substr_index rest ~pattern:" = " with
      | None -> None
      | Some j -> Some (String.prefix rest j))

let has_operand_annotation line ~operand ~want =
  let needle = "{=" ^ want ^ "}" in
  let n = String.length line in
  let op_len = String.length operand in
  let rec skip_parens i = if i < n && Char.equal line.[i] ')' then skip_parens (i + 1) else i in
  List.exists (String.substr_index_all line ~may_overlap:false ~pattern:operand) ~f:(fun i ->
      let j = skip_parens (i + op_len) in
      String.is_substring_at line ~pos:j ~substring:needle)

let has_logged_value ?(operands = []) lines ~prefix ~want =
  let ok =
    List.exists lines ~f:(fun line ->
        String.is_prefix line ~prefix
        && Option.equal String.equal (stored_value line) (Some want)
        && List.for_all operands ~f:(fun (operand, want_op) ->
            has_operand_annotation line ~operand ~want:want_op))
  in
  (if not ok then
     let sample =
       List.filter lines ~f:(String.is_prefix ~prefix) |> fun l ->
       List.take l 3 |> String.concat ~sep:" | "
     in
     Stdio.eprintf "missing logged value prefix=%S want=%S (sample: %s)\n%!" prefix want sample);
  ok

let run_in_mode ~uniform ctx ~name o ~seed ~read =
  Utils.with_stated_config
    [ ("prefer_backend_uniformity", Bool.to_string uniform) ]
    (fun () ->
      let ctx', _routine = Ll_test.run_routine ~ctx:!ctx ~name o ~seed in
      ctx := ctx';
      let values = List.map read ~f:(Context.get_values ctx') in
      Context.sync ctx';
      values)

let check_set ctx ~uniform ~mode_tag ~first_id (prec_tag, prec) =
  let tag = Printf.sprintf "rlnf_%s_%s_set" mode_tag prec_tag in
  let mk = Ll_test.node_factory ~prec ~first_id ~dims:[| n |] () in
  let a = mk (tag ^ "_a") and b = mk (tag ^ "_b") and out = mk (tag ^ "_out") in
  List.iter [ a; b; out ] ~f:Ll_test.materialize;
  let i = Ll_test.sym () in
  let at = Ll_test.iter i in
  let llc =
    Ll_test.loop_n i n (Ll_test.set_at out at (Ll_test.add_own (get a [| at |]) (get b [| at |])))
  in
  let o = Ll_test.optimize ~materialized:[ a; b; out ] ~name:tag llc in
  let av = [| 1.5; -2.0; 0.5; 3.0 |] in
  let bv = [| 0.5; 1.0; -0.25; 1.0 |] in
  let want = [| 2.0; -1.0; 0.25; 4.0 |] in
  let got =
    match
      run_in_mode ~uniform ctx ~name:tag o ~seed:[ (a, av); (b, bv); (out, init_out) ] ~read:[ out ]
    with
    | [ v ] -> v
    | _ -> assert false
  in
  let values_ok = Array.equal Float.equal got want in
  if not values_ok then Stdio.eprintf "%s: values mismatch\n%!" tag;
  let lines = stream_log_lines () in
  let out_name = Ir.Tnode.label_name out in
  let log_ok =
    has_logged_value lines ~prefix:(out_name ^ "[0]{")
      ~want:(expected_text ~uniform ~scaled:"2000e-3" ~plain:"2")
    && has_logged_value lines ~prefix:(out_name ^ "[1]{")
         ~want:(expected_text ~uniform ~scaled:"-1000e-3" ~plain:"-1")
    && has_logged_value lines ~prefix:(out_name ^ "[2]{")
         ~want:(expected_text ~uniform ~scaled:"250e-3" ~plain:"0.25")
  in
  values_ok && log_ok

let check_where ctx ~uniform ~mode_tag ~first_id (prec_tag, prec) =
  let tag = Printf.sprintf "rlnf_%s_%s_where" mode_tag prec_tag in
  let mk = Ll_test.node_factory ~prec ~first_id ~dims:[| n |] () in
  (* [a] is shorter than the loop extent, so the debug log's [Where] guard must short-circuit its
     read while pairing the widened expression with [0]. *)
  let a = mk ~dims:[| 2 |] (tag ^ "_a") and b = mk (tag ^ "_b") and out = mk (tag ^ "_out") in
  List.iter [ a; b; out ] ~f:Ll_test.materialize;
  let i = Ll_test.sym () in
  let at = Ll_test.iter i in
  let cond = Ll_test.lt (Ll_test.embed i) (Ll_test.ic 2) in
  let llc =
    Ll_test.loop_n i n
      (Ll_test.set_at out at (Ll_test.where_own cond (get a [| at |]) (get b [| at |])))
  in
  let o = Ll_test.optimize ~materialized:[ a; b; out ] ~name:tag llc in
  let av = [| 1.5; -2.0 |] in
  let bv = [| 0.5; 1.0; -0.25; 3.0 |] in
  let want = [| 1.5; -2.0; -0.25; 3.0 |] in
  let got =
    match
      run_in_mode ~uniform ctx ~name:tag o ~seed:[ (a, av); (b, bv); (out, init_out) ] ~read:[ out ]
    with
    | [ v ] -> v
    | _ -> assert false
  in
  let values_ok = Array.equal Float.equal got want in
  if not values_ok then Stdio.eprintf "%s: values mismatch\n%!" tag;
  let lines = stream_log_lines () in
  let a_name = Ir.Tnode.label_name a in
  let b_name = Ir.Tnode.label_name b in
  let out_name = Ir.Tnode.label_name out in
  let zero = expected_text ~uniform ~scaled:"0e-3" ~plain:"0" in
  let val_1_5 = expected_text ~uniform ~scaled:"1500e-3" ~plain:"1.5" in
  let val_3 = expected_text ~uniform ~scaled:"3000e-3" ~plain:"3" in
  let log_ok =
    has_logged_value lines ~prefix:(out_name ^ "[0]{") ~want:val_1_5
      ~operands:[ (a_name ^ "[0]", val_1_5); (b_name ^ "[0]", zero) ]
    && has_logged_value lines ~prefix:(out_name ^ "[3]{") ~want:val_3
         ~operands:[ (a_name ^ "[3]", zero); (b_name ^ "[3]", val_3) ]
  in
  values_ok && log_ok

let check_set_local ctx ~uniform ~mode_tag ~first_id (prec_tag, prec) =
  let tag = Printf.sprintf "rlnf_%s_%s_local" mode_tag prec_tag in
  let mk = Ll_test.node_factory ~prec ~first_id ~dims:[| n |] () in
  let a = mk (tag ^ "_a") and out = mk (tag ^ "_out") and tmp = mk (tag ^ "_tmp") in
  List.iter [ a; out ] ~f:Ll_test.materialize;
  Ll_test.virtualize tmp;
  let i = Ll_test.sym () in
  let at = Ll_test.iter i in
  let id = LL.get_scope tmp in
  (* Two assignments keep the scope from collapsing into the enclosing store. *)
  let body =
    Ll_test.seq
      (LL.Set_local (id, get a [| at |]))
      (LL.Set_local (id, Ll_test.add_own (LL.Get_local id) (Ll_test.c 1.0)))
  in
  let scoped = Ll_test.loop_n i n (Ll_test.set_at out at (Ll_test.scope_of id body)) in
  let raw =
    Ll_test.loop_n i n (Ll_test.set_at out at (Ll_test.add_own (get a [| at |]) (Ll_test.c 1.0)))
  in
  let o = Ll_test.optimize_scoped ~materialized:[ a; out ] ~name:tag ~raw scoped in
  let av = [| 1.0; -2.0; 0.5; 3.0 |] in
  let want = [| 2.0; -1.0; 1.5; 4.0 |] in
  let got =
    match run_in_mode ~uniform ctx ~name:tag o ~seed:[ (a, av); (out, init_out) ] ~read:[ out ] with
    | [ v ] -> v
    | _ -> assert false
  in
  let values_ok = Array.equal Float.equal got want in
  if not values_ok then Stdio.eprintf "%s: values mismatch\n%!" tag;
  let lines = stream_log_lines () in
  let local_prefix = LL.scope_local_prefix id.scope_id in
  let log_ok =
    has_logged_value lines ~prefix:local_prefix
      ~want:(expected_text ~uniform ~scaled:"2000e-3" ~plain:"2")
  in
  values_ok && log_ok

let check_set_from_vec ctx ~uniform ~mode_tag ~first_id (prec_tag, prec) =
  let tag = Printf.sprintf "rlnf_%s_%s_vec" mode_tag prec_tag in
  let length = Ops.vec_unop_lanes prec in
  let mk = Ll_test.node_factory ~prec ~first_id ~dims:[| length |] () in
  let out = mk (tag ^ "_out") in
  Ll_test.materialize out;
  let iprec = Ops.index_prec () in
  let threefry a b = LL.Binop (Ops.Threefry4x32_light, a, b) in
  let block =
    threefry
      (threefry (LL.Constant_bits 7L, u4) (LL.Constant_bits 42L, u4), u4)
      (LL.Embed_index (Idx.Fixed_idx 0), iprec)
  in
  let llc =
    Ll_test.set_from_vec ~length ~vec_unop:Ops.Uint4x32_to_prec_uniform out [| Idx.Fixed_idx 0 |]
      (block, u4)
  in
  let o = Ll_test.optimize ~materialized:[ out ] ~name:tag llc in
  let seed_out = Array.create ~len:length (-4.0) in
  let got =
    match run_in_mode ~uniform ctx ~name:tag o ~seed:[ (out, seed_out) ] ~read:[ out ] with
    | [ v ] -> v
    | _ -> assert false
  in
  let values_ok =
    Array.length got = length
    && Array.for_all got ~f:(fun v -> Float.(0. <= v && v < 1.))
    && not (Array.for_all got ~f:(Float.equal got.(0)))
  in
  if not values_ok then Stdio.eprintf "%s: RNG draws invalid\n%!" tag;
  let lines = stream_log_lines () in
  let out_name = Ir.Tnode.label_name out in
  let prefix = out_name ^ "[0]{" in
  let marker = " = vec_result.v[0] = " in
  let logged_lane0 =
    List.find_map lines ~f:(fun line ->
        if not (String.is_prefix line ~prefix) then None
        else
          match String.substr_index line ~pattern:marker with
          | None -> None
          | Some i ->
              Float.of_string_opt
                (String.strip (String.drop_prefix line (i + String.length marker))))
  in
  let log_ok =
    match logged_lane0 with
    | None ->
        Stdio.eprintf "%s: missing logged lane0 line (prefix=%S)\n%!" tag prefix;
        false
    | Some logged ->
        Verdict.within ~what_ran:tag "logged lane0" [| logged |] [| got.(0) |] ~tol:(fun _ -> 1e-3)
  in
  values_ok && log_ok

let () =
  if not (Utils.debug_log_from_routines ()) then
    fail "this test runs under --ocannl_debug_log_from_routines=true (see its dune rule)";
  Stdio.eprintf "routine_log_narrow_floats: backend %s (not part of the golden)\n%!" backend_name;
  let ctx = ref (Context.auto ()) in
  let shapes =
    [
      ("Set", check_set, 0);
      ("bounds-guarded Where", check_where, 300);
      ("Set_local", check_set_local, 600);
      ("Set_from_vec", check_set_from_vec, 900);
    ]
  in
  List.iteri modes ~f:(fun mi (mode_tag, uniform) ->
      let base_id = 2472000 + (mi * 2000) in
      List.iter shapes ~f:(fun (shape, check, offset) ->
          p_alli ~min:3
            (Printf.sprintf "%s: narrow-float %s compiles, executes, and logs numeric values"
               mode_tag shape) precisions ~f:(fun idx p_item ->
              check ctx ~uniform ~mode_tag ~first_id:(base_id + offset + (idx * 100)) p_item)))
