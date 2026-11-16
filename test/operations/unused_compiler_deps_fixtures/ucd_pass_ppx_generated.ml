(* Positive control: the source names no ppxlib.ast module; metaquot's expansion does. *)
let one ~loc = [%expr 1]
