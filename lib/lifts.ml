(* Included from BER MetaOCaml to avoid build problems. See:
   https://github.com/metaocaml/ber-metaocaml/blob/ber-n111/ber-metaocaml-111/lifts.ml *)
(* Liftable types
   The types of present-stage values that are OK to use at a future
   stage (values safe to CSP, both in byte- and native-code modes)

   Although polymorphic lift is possible in MetaOCaml and works well
   in byte-code, it is not at all helpful in the native mode.
*)


open Codelib

(* Although the lift methods below all look the same, it is the
   type that matters.
   The type will make MetaOCaml generate the right code for lift.
*)

module Lift_int = struct
  type t = int
  let lift (x:int) = .<x>.
end

module Lift_char = struct
  type t = char
  let lift (x:char) = .<x>.
end

module Lift_float = struct
  type t = float
  let lift (x:float) = .<x>.
end

module Lift_string = struct
  type t = string
  let lift (x:string) = .<x>.
end


(* More can, and should be added *)

module Lift_option(L:lift) = struct
  type t = L.t option
  let lift = function 
    | None   -> .<None>.
    | Some x -> .<Some .~(L.lift x)>.
end

module Lift_list(L:lift) = struct
  type t = L.t list
  let lift l = 
    genlet @@
    List.fold_right (fun x acc -> .<.~(L.lift x) :: .~acc>.) l .<[]>.
end

let lift_array : 'a code array -> 'a array code = fun arr ->
  arr |> 
    Array.map (fun x -> (x : 'a code :> Trx.code_repr)) |>
    Trx.build_array Trx.loc_none |>  (* returns code_repr for array *)
    Ocannl_runtime.Node.Obj.magic |>
    genlet

module Lift_array(L:lift) = struct
  type t = L.t array
  let lift l = 
    l |> Array.map L.lift |> lift_array
end



