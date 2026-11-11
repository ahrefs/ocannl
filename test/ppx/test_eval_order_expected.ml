open Ocannl.Operation.DSL_modules
open Verdict.Claims
let evaluated = ref []
let mk name = evaluated := (name :: (!evaluated)); TDSL.range ~label:[name] 3
let id (t : Tensor.t) = (t.value).id
let in_source_order what names f =
  evaluated := [];
  (let (_ : Tensor.t) = f () in
   let order = List.rev (!evaluated) in
   Printf.printf "%s: evaluated %s\n" what (String.concat " " order);
   p (what ^ ": components are evaluated in source order") (order = names))
let ids_ascend what ts =
  p_alli (what ^ ": component ids ascend in source order") ts
    ~f:(fun i t -> (i = 0) || ((id (List.nth ts (i - 1))) < (id t)))
let () =
  in_source_order "list block" ["a"; "b"; "c"]
    (fun () ->
       let open! TDSL.O in
         let __op_component1 =
           Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> mk "a") in
         let __op_component2 =
           Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> mk "b") in
         stack ?label:(Some []) `Output
           [|__op_component1;__op_component2;(Tensor.Scheme.For_ppx.at_site
                                                Stdlib.__POS__
                                                (fun () -> mk "c"))|]);
  in_source_order "array block" ["a"; "b"; "c"]
    (fun () ->
       let open! TDSL.O in
         let __op_component1 =
           Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> mk "a") in
         let __op_component2 =
           Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> mk "b") in
         stack ?label:(Some []) `Batch
           [|__op_component1;__op_component2;(Tensor.Scheme.For_ppx.at_site
                                                Stdlib.__POS__
                                                (fun () -> mk "c"))|]);
  in_source_order "tuple block" ["a"; "b"; "c"]
    (fun () ->
       let open! TDSL.O in
         let __op_component1 =
           Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> mk "a") in
         let __op_component2 =
           Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> mk "b") in
         stack ?label:(Some []) `Input
           [|__op_component1;__op_component2;(Tensor.Scheme.For_ppx.at_site
                                                Stdlib.__POS__
                                                (fun () -> mk "c"))|]);
  in_source_order "nested block" ["a"; "b"; "c"; "d"]
    (fun () ->
       let open! TDSL.O in
         let __op_component_1 =
           let __op_component1 =
             Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> mk "a") in
           stack ?label:None `Input
             [|__op_component1;(Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                                  (fun () -> mk "b"))|] in
         stack ?label:(Some []) `Output
           [|__op_component_1;((let __op_component1 =
                                  Tensor.Scheme.For_ppx.at_site
                                    Stdlib.__POS__ (fun () -> mk "c") in
                                stack ?label:None `Input
                                  [|__op_component1;(Tensor.Scheme.For_ppx.at_site
                                                       Stdlib.__POS__
                                                       (fun () -> mk "d"))|]))|]);
  in_source_order "concat operands" ["a"; "b"; "c"]
    (fun () ->
       let open! TDSL.O in
         let __op_component1 =
           Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> mk "a") in
         let __op_component2 =
           Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> mk "b") in
         concat ?label:(Some []) "i; j; k => i^j^k"
           [|__op_component1;__op_component2;(Tensor.Scheme.For_ppx.at_site
                                                Stdlib.__POS__
                                                (fun () -> mk "c"))|]);
  in_source_order "concat operands with captures" ["a"; "b"]
    (fun () ->
       let open! TDSL.O in
         let i = Shape.get_variable_ref "i" in
         let j = Shape.get_variable_ref "j" in
         let __op_component1 =
           Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> mk "a") in
         concat ?label:(Some []) ~capture_dims:[i; j] "i; j => i^j"
           [|__op_component1;(Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                                (fun () -> mk "b"))|])
let () =
  let x = mk "x"
  and y = mk "y" in
  in_source_order "variables among effectful components" ["a"; "b"]
    (fun () ->
       let open! TDSL.O in
         let __op_component2 =
           Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> mk "a") in
         stack ?label:(Some []) `Output
           [|x;__op_component2;y;(Tensor.Scheme.For_ppx.at_site
                                    Stdlib.__POS__ (fun () -> mk "b"))|]);
  in_source_order "variables only" []
    (fun () -> let open! TDSL.O in stack ?label:(Some []) `Output [|x;y|])
let () =
  let __op_component1 = mk "y" in
  let stacked = ref [] in
  let keep t = stacked := (t :: (!stacked)); t in
  let (_ : Tensor.t) =
    let open! TDSL.O in
      let __op_component_1 =
        Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
          (fun () ->
             keep
               (Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                  (fun () -> mk "a"))) in
      stack ?label:(Some []) `Output
        [|__op_component_1;(Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                              (fun () -> keep __op_component1))|] in
  p "a variable named like a binding is not captured"
    (match !stacked with
     | second::first::[] -> (second == __op_component1) && (first != second)
     | _ -> false)
let () =
  let a = ref None
  and b = ref None
  and c = ref None in
  let keep r t = r := (Some t); t in
  let (_ : Tensor.t) =
    let open! TDSL.O in
      let __op_component1 =
        Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
          (fun () ->
             keep a
               (Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                  (fun () -> mk "a"))) in
      let __op_component2 =
        Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
          (fun () ->
             keep b
               (Tensor.Scheme.For_ppx.at_site Stdlib.__POS__
                  (fun () -> mk "b"))) in
      stack ?label:(Some []) `Output
        [|__op_component1;__op_component2;(Tensor.Scheme.For_ppx.at_site
                                             Stdlib.__POS__
                                             (fun () ->
                                                keep c
                                                  (Tensor.Scheme.For_ppx.at_site
                                                     Stdlib.__POS__
                                                     (fun () -> mk "c"))))|] in
  ids_ascend "list block" (List.filter_map (fun r -> !r) [a; b; c])
let () =
  let (i, j, k) =
    let i = let open! TDSL.O in Shape.get_variable_ref "i" in
    let j = let open! TDSL.O in Shape.get_variable_ref "j" in
    let k = let open! TDSL.O in Shape.get_variable_ref "k" in
    let r =
      let open! TDSL.O in
        let __op_component1 =
          Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> mk "a") in
        let __op_component2 =
          Tensor.Scheme.For_ppx.at_site Stdlib.__POS__ (fun () -> mk "b") in
        concat ?label:(Some ["r"]) ~capture_dims:[i; j; k] "i; j; k => i^j^k"
          [|__op_component1;__op_component2;(Tensor.Scheme.For_ppx.at_site
                                               Stdlib.__POS__
                                               (fun () -> mk "c"))|] in
    ignore (r : Tensor.t); (i, j, k) in
  p_alli "capture list: reference uids ascend in source order" [i; j; k]
    ~f:(fun n r ->
          (n = 0) || ((List.nth [i; j; k] (n - 1)).Shape.uid < r.Shape.uid))
