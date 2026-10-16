(* The conv workloads' models, shared by the measured runner (bench_conv) and its segment diagnostic
   (bench_conv_diag), so that the diagnostic's census and per-segment times are of the graph the
   benchmark measures. Layouts are documented in gen_fixtures.py's build_conv and build_conv_resnet;
   the fixture's [arch] metadata selects the builder. Fixture weights are injected into the
   block-created inline params by debug-name token matching (Bench_harness.inject) through the
   [mapping] each builder returns. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules
module St = Safetensors
module H = Bench_harness

type t = {
  model : Tensor.t -> Tensor.t;  (** Images [batch | h, w, c] to logits. *)
  mapping : (string * string list) list;  (** Fixture key, required debug-name tokens. *)
}

(* LeNet-5 and the cifar-scale two-conv classifiers: conv -> relu -> pool, twice, then the head. *)
let two_conv st =
  let c1 = H.meta_int st "channels1" in
  let c2 = H.meta_int st "channels2" in
  let k = H.meta_int st "kernel_size" in
  let fc1_dim = H.meta_int st "fc1" in
  let fc2_dim = H.meta_int st "fc2" in
  (* Same-padding keeps the spatial extent divisible; valid convs (LeNet) shrink it. The flag
     applies to BOTH convs — gen_fixtures sizes fc1_w as if both preserve extent, and the Python
     runners pad both. Defaults to valid so pre-existing fixtures are unchanged. *)
  let use_padding =
    match List.Assoc.find (St.metadata st) ~equal:String.equal "use_padding" with
    | Some "true" -> true
    | _ -> false
  in
  (* Per-conv strides (cifar_stride's stride-2 stem, gh-ocannl-502); valid-only, see gen_fixtures.
     Absent on pre-existing fixtures. *)
  let s1 = H.meta_int_default st "stride1" ~default:1 in
  let s2 = H.meta_int_default st "stride2" ~default:1 in
  let conv1 =
    Nn_blocks.conv2d ~label:[ "conv1" ] ~kernel_size:k ~stride:s1 ~use_padding ~out_channels:c1 ()
  in
  let conv2 =
    Nn_blocks.conv2d ~label:[ "conv2" ] ~kernel_size:k ~stride:s2 ~use_padding ~out_channels:c2 ()
  in
  let pool1 = Nn_blocks.max_pool2d ~stride:2 ~window_size:2 () in
  let pool2 = Nn_blocks.max_pool2d ~stride:2 ~window_size:2 () in
  let fc1 = Nn_blocks.mlp_layer ~label:[ "fc1" ] ~hid_dim:fc1_dim () in
  let fc2 = Nn_blocks.mlp_layer ~label:[ "fc2" ] ~hid_dim:fc2_dim () in
  let%op model x =
    ({ w_logits } * fc2 (fc1 (pool2 (relu (conv2 (pool1 (relu (conv1 x)))))))) + { b_logits = 0. }
  in
  let mapping =
    [
      ("conv1_kernel", [ "conv1"; "kernel" ]);
      ("conv1_bias", [ "conv1"; "bias" ]);
      ("conv2_kernel", [ "conv2"; "kernel" ]);
      ("conv2_bias", [ "conv2"; "bias" ]);
      ("fc1_w", [ "fc1"; "w" ]);
      ("fc1_b", [ "fc1"; "b" ]);
      ("fc2_w", [ "fc2"; "w" ]);
      ("fc2_b", [ "fc2"; "b" ]);
      ("w_logits", [ "w"; "logits" ]);
      ("b_logits", [ "b"; "logits" ]);
    ]
  in
  { model; mapping }

(* The reduced ResNet-style classifier (gh-ocannl-1161): a padded stem conv, then one residual block
   per stage (s1, s2, ...), then a 2x2 pool and the two-conv workloads' head. Block s1 adds the
   identity; every later block halves the extent with a 2x2 pool on the main path and a stride-2 1x1
   conv on the shortcut. The names are the cells' site names: [s<i>c1]/[s<i>c2] the padded k x k
   convs (their kernels and biases), [s<i>sc] the strided 1x1 shortcut, [s<i>res] the residual add
   and [s<i>out] the relu after it, [s<i>c1out] the relu between the two k x k convs, so a kernel
   table's written nodes say which site a kernel belongs to (a virtual node writes nothing and so
   names nothing). The shortcut is spelled [use_padding:true] like [Nn_blocks.resnet_block]'s
   downsample conv: a 1x1 window has a zero margin, and the valid mode would need an odd extent
   (gen_fixtures.py's build_conv_resnet). *)
let resnet st =
  let k = H.meta_int st "kernel_size" in
  let fc1_dim = H.meta_int st "fc1" in
  let fc2_dim = H.meta_int st "fc2" in
  let chans = H.get_meta st "stage_channels" |> String.split ~on:',' |> List.map ~f:Int.of_string in
  let conv ~name ?(kernel_size = k) ?(stride = 1) out_channels =
    Nn_blocks.conv2d ~label:[ name ] ~kernel_size ~stride ~use_padding:true ~out_channels ()
  in
  let stem = conv ~name:"stem" (List.hd_exn chans) in
  let block i c =
    let s = Printf.sprintf "s%d" (i + 1) in
    let c1 = conv ~name:(s ^ "c1") c and c2 = conv ~name:(s ^ "c2") c in
    let residual main shortcut =
      TDSL.O.relu ~label:[ s ^ "out" ] (TDSL.add ~label:[ s ^ "res" ] main shortcut ())
    in
    let main x = c2 (TDSL.O.relu ~label:[ s ^ "c1out" ] (c1 x)) in
    if i = 0 then fun x -> residual (main x) x
    else
      let pool = Nn_blocks.max_pool2d ~stride:2 ~window_size:2 () in
      let sc = conv ~name:(s ^ "sc") ~kernel_size:1 ~stride:2 c in
      fun x -> residual (main (pool x)) (sc x)
  in
  let blocks = List.mapi chans ~f:block in
  let pool = Nn_blocks.max_pool2d ~stride:2 ~window_size:2 () in
  let fc1 = Nn_blocks.mlp_layer ~label:[ "fc1" ] ~hid_dim:fc1_dim () in
  let fc2 = Nn_blocks.mlp_layer ~label:[ "fc2" ] ~hid_dim:fc2_dim () in
  let features x =
    List.fold blocks ~init:(TDSL.O.relu ~label:[ "stemout" ] (stem x)) ~f:(fun h b -> b h)
  in
  let%op model x = ({ w_logits } * fc2 (fc1 (pool (features x)))) + { b_logits = 0. } in
  let conv_entries name =
    [ (name ^ "_kernel", [ name; "kernel" ]); (name ^ "_bias", [ name; "bias" ]) ]
  in
  let sites =
    "stem"
    :: List.concat_mapi chans ~f:(fun i _ ->
        let s = Printf.sprintf "s%d" (i + 1) in
        [ s ^ "c1"; s ^ "c2" ] @ if i = 0 then [] else [ s ^ "sc" ])
  in
  let mapping =
    List.concat_map sites ~f:conv_entries
    @ [
        ("fc1_w", [ "fc1"; "w" ]);
        ("fc1_b", [ "fc1"; "b" ]);
        ("fc2_w", [ "fc2"; "w" ]);
        ("fc2_b", [ "fc2"; "b" ]);
        ("w_logits", [ "w"; "logits" ]);
        ("b_logits", [ "b"; "logits" ]);
      ]
  in
  { model; mapping }

let build st =
  match List.Assoc.find (St.metadata st) ~equal:String.equal "arch" with
  | None | Some "two_conv" -> two_conv st
  | Some "resnet" -> resnet st
  | Some arch -> failwith ("bench_conv: unknown conv arch " ^ arch)
