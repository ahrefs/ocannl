(** Prior to OCANNL 0.5, this module is just a placeholder hinting at an intended design pattern
    for model components. *)

open! Base
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL

type mlp_layer_config = { label : string list; hid_dim : int }

let%op mlp_layer ~config x = ?/(("w" * x) + "b" config.hid_dim)

type mlp_config = { label : string list; hid_dims : int list }

let mlp ~config =
  let layers =
    List.mapi config.hid_dims ~f:(fun i hid_dim ->
        mlp_layer ~config:{ label = [ "L" ^ Int.to_string i ]; hid_dim })
  in
  fun x -> List.fold layers ~init:x ~f:(fun x layer -> layer x)
