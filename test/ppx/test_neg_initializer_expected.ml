open Ocannl.Nn_blocks.DSL_modules
let source = NTDSL.range 4
let parameter =
  let open! TDSL.O in
    let w =
      (TDSL.param ?more_label:None ?value:None ?values:None
         ?param_init:(Some (NTDSL.neg source)) "w") () in
    w
module DSL_modules =
  struct
    module Ir = Ir
    module Shape = Shape
    module Tensor = Tensor
    module TDSL =
      struct
        include TDSL
        let negative =
          let open! TDSL.O in
            let __ocannl_body x =
              Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                (fun () -> TDSL.neg x) in
            let __ocannl_scheme =
              Tensor.Scheme.For_ppx.deferred
                ~name:"%op function at ./test_neg_initializer.ml:6" ~arity:1
                (function | x::[] -> __ocannl_body x | _ -> assert false) in
            fun x ->
              Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
                (fun () -> __ocannl_body x)
        module O =
          struct
            include TDSL.O
            let negative ?label x = negative x ?label ()
          end
      end
    module NTDSL =
      struct
        include NTDSL
        let negative =
          let open! NTDSL.O in
            fun x ->
              Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                (fun () -> NTDSL.neg x)
        module O =
          struct
            include NTDSL.O
            let negative ?label x = negative x ?label ()
          end
      end
  end
