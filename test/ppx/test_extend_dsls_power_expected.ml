open Ocannl.Nn_blocks.DSL_modules
let counter = ref 2
let exponent = ref 2.0
module DSL_modules =
  struct
    module Ir = Ir
    module Shape = Shape
    module Tensor = Tensor
    module TDSL =
      struct
        include TDSL
        let squared =
          let open! TDSL.O in
            let __ocannl_body x =
              Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                (fun () -> TDSL.pointpow (Float.of_int 2) x) in
            let __ocannl_scheme =
              Tensor.Scheme.For_ppx.deferred
                ~name:"%op function at ./test_extend_dsls_power.ml:7"
                ~arity:1
                (function | x::[] -> __ocannl_body x | _ -> assert false) in
            fun x ->
              Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
                (fun () -> __ocannl_body x)
        and counted_power =
          let open! TDSL.O in
            let __ocannl_body x =
              let __exponent1 = Float.of_int (!counter) in
              Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                (fun () -> TDSL.pointpow __exponent1 (Stdlib.incr counter; x)) in
            let __ocannl_scheme =
              Tensor.Scheme.For_ppx.deferred
                ~name:"%op function at ./test_extend_dsls_power.ml:9"
                ~arity:1
                (function | x::[] -> __ocannl_body x | _ -> assert false) in
            fun x ->
              Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
                (fun () -> __ocannl_body x)
        and escaped_power =
          let open! TDSL.O in
            let __ocannl_body x =
              Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                (fun () -> TDSL.pointpow (!exponent) x) in
            let __ocannl_scheme =
              Tensor.Scheme.For_ppx.deferred
                ~name:"%op function at ./test_extend_dsls_power.ml:15"
                ~arity:1
                (function | x::[] -> __ocannl_body x | _ -> assert false) in
            fun x ->
              Tensor.Scheme.For_ppx.apply __ocannl_scheme [x]
                (fun () -> __ocannl_body x)
        and power_of_two =
          let open! TDSL.O in
            fun () ->
              let __exponent1 = Float.of_int (!counter) in
              Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                (fun () ->
                   TDSL.pointpow __exponent1 (TDSL.number ?label:None 2.0))
        module O =
          struct
            include TDSL.O
            let squared ?label x = squared x ?label ()
            and counted_power ?label x = counted_power x ?label ()
            and escaped_power ?label x = escaped_power x ?label ()
            and power_of_two ?label () = power_of_two () ?label ()
          end
      end
    module NTDSL =
      struct
        include NTDSL
        let squared =
          let open! NTDSL.O in
            fun x ->
              Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                (fun () -> NTDSL.pointpow (Float.of_int 2) x)
        and counted_power =
          let open! NTDSL.O in
            fun x ->
              let __exponent1 = Float.of_int (!counter) in
              Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                (fun () ->
                   NTDSL.pointpow __exponent1 (Stdlib.incr counter; x))
        and escaped_power =
          let open! NTDSL.O in
            fun x ->
              Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                (fun () -> NTDSL.pointpow (!exponent) x)
        and power_of_two =
          let open! NTDSL.O in
            fun () ->
              let __exponent1 = Float.of_int (!counter) in
              Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                (fun () ->
                   NTDSL.pointpow __exponent1 (NTDSL.number ?label:None 2.0))
        module O =
          struct
            include NTDSL.O
            let squared ?label x = squared x ?label ()
            and counted_power ?label x = counted_power x ?label ()
            and escaped_power ?label x = escaped_power x ?label ()
            and power_of_two ?label () = power_of_two () ?label ()
          end
      end
  end
