open Base
open Ocannl
module TDSL = Operation.TDSL
let y0 =
  let hey1 = TDSL.param ?values:None "hey1" in
  let open! TDSL.O in
    ((+) ?label:(Some ["y0"]))
      ((( *. ) ?label:None) (TDSL.number (Float.of_int 2)) hey1)
      (TDSL.number (Float.of_int 3))
let y1 =
  let hey2 = TDSL.param ?values:None "hey2" in
  let open! TDSL.O in
    fun x ->
      ((+) ?label:(Some (["y1"] @ (x.Tensor.value).Ir.Tnode.label)))
        ((( * ) ?label:None) hey2 (TDSL.number (Float.of_int 2))) x
let y2 =
  let hey3 = TDSL.param ?values:None "hey3" in
  let open! TDSL.O in
    fun x1 ->
      fun x2 ->
        ((+) ?label:(Some (["y2"] @ (x1.Tensor.value).Ir.Tnode.label)))
          ((( *. ) ?label:None) x1 hey3) x2
let a =
  let open! TDSL.O in
    TDSL.ndarray ?label:(Some ["a"]) ~batch_dims:[] ~input_dims:[3]
      ~output_dims:[2]
      [|(Float.of_int 1);(Float.of_int 2);(Float.of_int 3);(Float.of_int 4);(
        Float.of_int 5);(Float.of_int 6)|]
let b =
  let open! TDSL.O in
    TDSL.ndarray ?label:(Some ["b"]) ~batch_dims:[2] ~input_dims:[]
      ~output_dims:[2]
      [|(Float.of_int 7);(Float.of_int 8);(Float.of_int 9);(Float.of_int 10)|]
let y =
  let hey4 = TDSL.param ?values:None "hey4" in
  let open! TDSL.O in
    ((+) ?label:(Some ["y"]))
      ((( * ) ?label:None) hey4 (TDSL.number ?label:None ~axis_label:"q" 2.0))
      (TDSL.number ?label:None ~axis_label:"p" 1.0)
let z =
  let hey5 = TDSL.param ?values:None "hey5"
  and hey6 = TDSL.param ?values:None "hey6" in
  let open! TDSL.O in
    ((+) ?label:(Some ["z"]))
      ((( * ) ?label:None) (TDSL.number ?label:None ~axis_label:"q" 2.0) hey5)
      ((( * ) ?label:None) hey6 (TDSL.number ?label:None ~axis_label:"p" 1.0))
let () = ignore (y0, y1, y2, a, b, y, z)
type mlp_layer_config = {
  label: string list ;
  hid_dim: int }
let mlp_layer =
  let open! TDSL.O in
    fun ~config ->
      let b =
        (TDSL.param ~more_label:(config.label)) ~output_dims:[config.hid_dim]
          "b"
      and w = (TDSL.param ~more_label:(config.label)) ?values:None "w" in
      fun x ->
        (relu ?label:(Some (["mlp_layer"] @ (x.Tensor.value).Ir.Tnode.label)))
          (((+) ?label:None) ((( * ) ?label:None) w x) b)
let _use_layer =
  let config_block__0 = mlp_layer ~config:{ label = ["L2"]; hid_dim = 3 }
  and config_block__1 = mlp_layer ~config:{ label = ["L"]; hid_dim = 3 } in
  let open! TDSL.O in fun x -> config_block__1 (config_block__0 x)
let _config_layer =
  let open! TDSL.O in
    fun ~config:_ ->
      let config_block__0 = mlp_layer ~config:{ label = ["L"]; hid_dim = 3 } in
      fun x -> config_block__0 x
type tlp_config = {
  label: string list ;
  dim1: int ;
  dim2: int ;
  dim3: int }
let _three_layer_perceptron =
  let open! TDSL.O in
    fun ~config:(config : tlp_config) ->
      let config_block__0 =
        mlp_layer
          ~config:{ label = ("L1" :: (config.label)); hid_dim = (config.dim1)
                  }
      and config_block__1 =
        mlp_layer
          ~config:{ label = ("L2" :: (config.label)); hid_dim = (config.dim2)
                  }
      and config_block__2 =
        mlp_layer
          ~config:{ label = ("L3" :: (config.label)); hid_dim = (config.dim3)
                  } in
      fun x -> config_block__2 (config_block__1 (config_block__0 x))
