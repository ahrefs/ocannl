open! Base
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module PDSL = Operation.PDSL

let%op mlp_layer ~label ~hid_dim () x = relu (({ w = uniform () } * x) + { b = 0.; o = [ hid_dim ] })

let mlp ~label ~hid_dims () =
  let layers =
    List.mapi hid_dims ~f:(fun i hid_dim ->
        mlp_layer ~label:(("L" ^ Int.to_string i) :: label) ~hid_dim ())
  in
  fun x -> List.fold layers ~init:x ~f:(fun x layer -> layer x)
