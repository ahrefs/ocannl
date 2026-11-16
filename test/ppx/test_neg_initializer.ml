open Ocannl.Nn_blocks.DSL_modules

let source = NTDSL.range 4
let parameter = [%op { w = -source }]

[%%extend_dsls let negative x = -x]
