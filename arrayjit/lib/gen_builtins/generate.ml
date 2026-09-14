(* The OCANNL_ namespace contains cc-only SIMD helpers. Everything else in the table is shared
   scalar C, including the native/emulated half abstraction and packed RNG types. Preserve table
   order, just as the kernel emitter does; macros may name later definitions. *)
let () =
  print_endline "/* Generated from builtins_cc.ml. Edit the definition table, not this file. */";
  print_endline "#ifndef ARRAYJIT_BUILTINS_SHARED_H";
  print_endline "#define ARRAYJIT_BUILTINS_SHARED_H";
  print_endline "#include <math.h>\n#include <stdint.h>\n#include <string.h>\n#include <stdlib.h>";
  let shared =
    List.filter
      (fun (key, _, _) -> not (String.starts_with ~prefix:"OCANNL_" key))
      Builtins_definitions.builtins
  in
  let keys = List.map (fun (key, _, _) -> key) shared in
  List.iter
    (fun (key, definition, dependencies) ->
      List.iter
        (fun dependency ->
          if not (List.mem dependency keys) then
            failwith (key ^ " depends on non-shared builtin " ^ dependency))
        dependencies;
      print_string definition;
      print_newline ())
    shared;
  print_endline "#endif"
