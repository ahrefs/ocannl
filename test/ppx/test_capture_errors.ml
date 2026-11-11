open Ocannl.Nn_blocks.DSL_modules

(* Expansion-only errors: an absent capture must be refused at its string literal, before inference.
   Every capture-bearing operator arm is covered, including the mixfix binary form. *)
let%op unary x = x ++ "ab => a" [ "ab" ]
let%op digit x = x ++ "f0 => f0" [ "f0" ]
let%op multichar x = x ++ "a, b => a" [ "ab" ]
let%op missing x = x ++ "ab => a" [ "c" ]
let%op binary x y = x +* "ab; bc => ac" [ "missing" ] y
let%op mixfix x y = x +* y "ab; bc => ac" [ "missing" ]
let%op concat x y = (x, y) ++^ "a; b => a^b" [ "missing" ]
let%cd cd_unary x = (x ++ "ab => a" [ "ab" ]).forward
let%cd cd_binary x y = (x +* "ab; bc => ac" [ "missing" ] y).forward
let%cd cd_mixfix x y = (x +* y "ab; bc => ac" [ "missing" ]).forward

(* The label inventory includes rows, labels in the result only, convolution constituents and
   concatenated components. Coefficient identifiers and fixed indices are not capture labels. *)
let%op rows x = x ++ "..batch..|..input..->..output.., a => a" [ "batch"; "input"; "output" ]
let%op result_only x = x ++ "a => b" [ "b" ]
let%op window ~stride ~dilation x = x ++ "stride*o<+dilation*k => o" [ "o"; "k" ]
let%op segments x = x ++ "left^middle^right => middle" [ "left"; "middle"; "right" ]
let%op multichar_ok x = x ++ "ab, => ab," [ "ab" ]
let%cd cd_multichar_ok x = (x ++ "ab, => ab," [ "ab" ]).forward
let%op row_only x = x ++ "..axes,.. => ..axes,.." [ "axes" ]
let%op coefficient x = x ++ "stride*o<+k => o" [ "stride" ]
let%op dilation_coefficient x = x ++ "o<+dilation*k => o" [ "dilation" ]

(* ~logic assignment captures share the operator checks and shape-logic references. *)
let%cd logic_unary tgt x = tgt =: id x ~logic:"a^c => a^b^c" [ "missing" ]
let%cd logic_binary tgt x y = tgt =: x * y ~logic:"a; a => a" [ "a" ]
let%cd logic_binary_tuple tgt x y = tgt =: mul (x, y) ~logic:"a; a => a" [ "a" ]
let%cd logic_ternary tgt x y z = tgt =: where x y z ~logic:"a; a; a => a" [ "a" ]
let%cd logic_ternary_tuple tgt x y z = tgt =: where (x, y, z) ~logic:"a; a; a => a" [ "a" ]
let%cd logic_unary_parens tgt x = tgt =: (id x) ~logic:"a => a" [ "a" ]
let%cd logic_empty tgt x = tgt =: id x ~logic:"a => a" []
